#!/usr/bin/env python3
"""
Batch evaluasi: grid (cover x payload) -> CSV PSNR/SSIM.

Paralel di dua tingkat:
  1. antar sel (cover x payload) via ProcessPoolExecutor spawn
  2. pencarian seed+r di dalam tiap sel (hmf.search, spawn)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from hmf import (
    choose_best_seed_and_matrix_r,
    embed_hamming_matchfirst_lsb,
    extract_hamming,
    huffman_expand_to_original_bits,
    load_image,
    matrix_capacity,
    maybe_huffman_compress_bits,
    psnr,
    read_payload_bits,
    resolve_mode,
    ssim,
)
from hmf.hamming import carrier_vector
from hmf.keys import KEY_TO_MODE

BASE = Path(__file__).resolve().parent

SEED_CANDIDATES = tuple(range(0, 16))
R_CANDIDATES = (3, 4, 5, 6, 7)
EMBED_SCOPE = "ALL_CHANNELS"
IMAGE_MODE = "RGB"

COVERS = [
    ("Tree", "cover_images/Tree.tiff"),
    ("House", "cover_images/House.tiff"),
    ("Baboon", "cover_images/Baboon_RGB.tiff"),
    ("Pepper", "cover_images/Pepper.tiff"),
    ("Medical", "cover_images/Medical.tiff"),
    ("San_Francisco", "cover_images/San_Fransisco.tiff"),
    ("Pentagon", "cover_images/Pentagon.tiff"),
]

PAYLOADS = [
    ("1kb", "payload/random-binary_1Kb.txt"),
    ("10kb", "payload/random-binary_10Kb.txt"),
    ("20kb", "payload/random-binary_20Kb.txt"),
    ("30kb", "payload/random-binary_30Kb.txt"),
    ("40kb", "payload/random-binary_40Kb.txt"),
    ("50kb", "payload/random-binary_50Kb.txt"),
    ("60kb", "payload/random-binary_60Kb.txt"),
    ("70kb", "payload/random-binary_70Kb.txt"),
    ("80kb", "payload/random-binary_80Kb.txt"),
    ("90kb", "payload/random-binary_90Kb.txt"),
    ("100kb", "payload/random-binary_100Kb.txt"),
]

FIELDNAMES = [
    "cover", "payload", "width", "height", "n_pixels",
    "payload_bits_raw", "payload_bits_embedded", "compression",
    "virtual_key", "virtual_mode",
    "seed", "matrix_r", "capacity_bits", "n_flips", "flips_per_bit",
    "n_no_flip_bits", "no_flip_bit_ratio", "n_flip_bits",
    "mse", "psnr_db", "ssim", "lossless", "seconds", "status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMF batch PSNR/SSIM")
    p.add_argument("--virtual-key", type=int, default=5, choices=sorted(KEY_TO_MODE))
    p.add_argument("--no-huffman", action="store_true")
    p.add_argument("--workers", type=int, default=None, help="worker antar-sel")
    p.add_argument("--search-workers", type=int, default=None, help="worker pencarian seed+r")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--serial", action="store_true", help="tanpa paralel antar-sel")
    return p.parse_args()


def _isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def _run_cell(task: Tuple, cfg: Dict) -> Dict:
    cover_label, cover_rel, payload_label, payload_rel = task
    t0 = time.time()
    row = {k: "" for k in FIELDNAMES}
    row["cover"] = cover_label
    row["payload"] = payload_label
    row["virtual_key"] = cfg["virtual_key"]
    row["virtual_mode"] = cfg["virtual_mode"]

    try:
        cover = load_image(BASE / cover_rel, mode=IMAGE_MODE)
        h, w = int(cover.shape[0]), int(cover.shape[1])
        carrier = carrier_vector(cover, embed_scope=EMBED_SCOPE)
        n_pixels, n_channels = int(carrier.shape[0]), int(carrier.shape[1])
        max_cap = max(matrix_capacity(n_pixels, r, n_channels) for r in R_CANDIDATES)

        payload_raw = read_payload_bits(BASE / payload_rel)
        if cfg["use_huffman"]:
            payload_bits, huff_meta = maybe_huffman_compress_bits(payload_raw)
        else:
            payload_bits, huff_meta = payload_raw, {"compression": "none"}

        row.update({
            "width": w,
            "height": h,
            "n_pixels": n_pixels,
            "payload_bits_raw": len(payload_raw),
            "payload_bits_embedded": len(payload_bits),
            "compression": str(huff_meta.get("compression", "none")),
        })

        if len(payload_bits) > max_cap:
            row["status"] = "EXCEEDS_CAPACITY"
            row["seconds"] = f"{time.time() - t0:.3f}"
            return row

        seed, matrix_r, _ = choose_best_seed_and_matrix_r(
            cover,
            payload_bits,
            seed_candidates=SEED_CANDIDATES,
            embed_scope=EMBED_SCOPE,
            virtual_mode=cfg["virtual_mode"],
            r_candidates=R_CANDIDATES,
            max_workers=cfg.get("search_workers"),
            parallel=cfg.get("parallel_search", True),
        )
        stego, stats = embed_hamming_matchfirst_lsb(
            cover,
            payload_bits,
            seed=seed,
            matrix_r=matrix_r,
            embed_scope=EMBED_SCOPE,
            virtual_mode=cfg["virtual_mode"],
        )
        cap = matrix_capacity(n_pixels, matrix_r, n_channels)
        mse, psnr_val = psnr(cover, stego)
        ssim_val = ssim(cover, stego)

        recovered_emb = extract_hamming(
            stego,
            len(payload_bits),
            seed=seed,
            matrix_r=matrix_r,
            embed_scope=EMBED_SCOPE,
            virtual_mode=cfg["virtual_mode"],
            virtual_choices=stats.virtual_choices,
        )
        recovered = huffman_expand_to_original_bits(recovered_emb, {
            "compression": huff_meta.get("compression", "none"),
            "original_bit_len": huff_meta.get("original_bit_len"),
            "huffman_freq": huff_meta.get("huffman_freq"),
        })
        lossless = recovered == payload_raw

        row.update({
            "seed": seed,
            "matrix_r": matrix_r,
            "capacity_bits": cap,
            "n_flips": stats.n_flips,
            "flips_per_bit": f"{stats.flips_per_bit:.8f}",
            "n_no_flip_bits": stats.n_no_flip_bits,
            "no_flip_bit_ratio": f"{stats.no_flip_bit_ratio:.8f}",
            "n_flip_bits": stats.n_flip_bits,
            "mse": f"{mse:.8f}",
            "psnr_db": "inf" if not _isfinite(psnr_val) else f"{psnr_val:.6f}",
            "ssim": f"{ssim_val:.8f}",
            "lossless": "OK" if lossless else "FAIL",
            "status": "OK" if lossless else "EXTRACT_MISMATCH",
        })
    except Exception as exc:
        row["status"] = f"ERROR: {type(exc).__name__}: {exc}"

    row["seconds"] = f"{time.time() - t0:.3f}"
    return row


def _matrix_value(row: Dict, field: str) -> str:
    status = row.get("status", "")
    if status == "EXCEEDS_CAPACITY":
        return "NA"
    if status.startswith("ERROR") or status.startswith("WORKER_FAIL"):
        return "ERR"
    val = row.get(field, "")
    return str(val) if val != "" else "ERR"


def _write_matrix(path: Path, corner: str, payload_labels: List[str], results: Dict, field: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow([corner] + payload_labels)
        for cover_label, _ in COVERS:
            wtr.writerow(
                [cover_label] + [_matrix_value(results.get((cover_label, pl), {}), field) for pl in payload_labels]
            )


def run_batch(cfg: Dict) -> None:
    tasks = [(cl, cr, pl, pr) for cl, cr in COVERS for pl, pr in PAYLOADS]
    total = len(tasks)
    results: Dict = {}
    out_dir = BASE / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "results_psnr_ssim.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as detail_file:
        writer = csv.DictWriter(detail_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        def record(row: Dict, idx: int) -> None:
            results[(row["cover"], row["payload"])] = row
            writer.writerow(row)
            detail_file.flush()
            print(
                f"[{idx}/{total}] {row['cover']} x {row['payload']}: "
                f"{row['status']} | seed={row['seed']} r={row['matrix_r']} "
                f"PSNR={row['psnr_db']} SSIM={row['ssim']} ({row['seconds']}s)",
                flush=True,
            )

        t_batch = time.time()

        if cfg.get("serial"):
            for i, task in enumerate(tasks, 1):
                record(_run_cell(task, cfg), i)
        else:
            max_workers = cfg.get("workers") or min(os.cpu_count() or 1, 8)
            ctx = mp.get_context("spawn")
            print(f"Batch: {total} sel, {max_workers} worker spawn", flush=True)
            done = 0
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_run_cell, t, cfg): t for t in tasks}
                for fut in as_completed(futures):
                    done += 1
                    try:
                        row = fut.result()
                    except Exception as exc:
                        cl, _, pl, _ = futures[fut]
                        row = {k: "" for k in FIELDNAMES}
                        row.update({"cover": cl, "payload": pl, "status": f"WORKER_FAIL: {type(exc).__name__}"})
                    record(row, done)

    payload_labels = [pl for pl, _ in PAYLOADS]
    _write_matrix(out_dir / "psnr_matrix.csv", "PSNR_dB (cover \\ payload)", payload_labels, results, "psnr_db")
    _write_matrix(out_dir / "ssim_matrix.csv", "SSIM (cover \\ payload)", payload_labels, results, "ssim")

    ok = sum(1 for r in results.values() if r.get("status") == "OK")
    na = sum(1 for r in results.values() if r.get("status") == "EXCEEDS_CAPACITY")
    err = len(results) - ok - na
    print(f"\nSelesai {time.time() - t_batch:.1f}s. OK={ok}, EXCEEDS={na}, lain={err}")
    print(f"Output: {detail_path}, psnr_matrix.csv, ssim_matrix.csv")


def main() -> None:
    args = parse_args()
    cfg = {
        "virtual_key": args.virtual_key,
        "virtual_mode": resolve_mode(args.virtual_key),
        "use_huffman": not args.no_huffman,
        "workers": args.workers,
        "search_workers": args.search_workers,
        "output_dir": args.output_dir,
        "serial": args.serial,
        "parallel_search": True,
    }
    run_batch(cfg)


if __name__ == "__main__":
    main()
