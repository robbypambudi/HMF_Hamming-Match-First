#!/usr/bin/env python3
"""Jalankan satu kali embed + extract untuk satu cover dan payload."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Batasi thread BLAS sebelum numpy diimpor (penting untuk multiprocessing).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np

from hmf import (
    choose_best_seed_and_matrix_r,
    embed_hamming_matchfirst_lsb,
    extract_hamming,
    huffman_expand_to_original_bits,
    load_image,
    load_key,
    matrix_block_n,
    matrix_capacity,
    maybe_huffman_compress_bits,
    psnr,
    read_payload_bits,
    resolve_mode,
    save_image,
    save_key,
    ssim,
    write_payload_bits,
)
from hmf.hamming import carrier_vector
from hmf.keys import KEY_LABELS, KEY_TO_MODE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMF embed + extract")
    p.add_argument("--cover", default="cover_images/Baboon_RGB.tiff")
    p.add_argument("--payload", default="payload/random-binary_10Kb.txt")
    p.add_argument("--stego", default="output/stego.tiff")
    p.add_argument("--key", default="output/key.json")
    p.add_argument("--extracted", default="output/extracted.txt")
    p.add_argument("--seed", type=int, default=None, help="seed tetap (lewati optimasi)")
    p.add_argument("--matrix-r", type=int, default=None, help="r tetap (lewati optimasi)")
    plugin_keys = sorted(KEY_TO_MODE)
    p.add_argument(
        "--virtual-key",
        type=int,
        default=5,
        choices=plugin_keys,
        help="key paper (default 5 = adaptive). "
        + ", ".join(f"{k}={KEY_LABELS[k]}" for k in plugin_keys),
    )
    p.add_argument("--image-mode", default="RGB", choices=["L", "RGB"])
    p.add_argument("--no-huffman", action="store_true")
    p.add_argument("--no-parallel", action="store_true", help="pencarian seed+r serial")
    p.add_argument("--seed-range", type=int, default=64, help="0..N-1 saat optimasi seed")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    cover_path = base / args.cover
    payload_path = base / args.payload
    stego_path = base / args.stego
    key_path = base / args.key
    extracted_path = base / args.extracted

    stego_path.parent.mkdir(parents=True, exist_ok=True)

    virtual_mode = resolve_mode(args.virtual_key)
    embed_scope = "ALL_CHANNELS"
    r_candidates = (3, 4, 5, 6, 7)

    cover = load_image(cover_path, mode=args.image_mode)
    payload_raw = read_payload_bits(payload_path)

    if args.no_huffman:
        payload_bits, huff_meta = payload_raw, {"compression": "none"}
    else:
        payload_bits, huff_meta = maybe_huffman_compress_bits(payload_raw)

    carrier = carrier_vector(cover, embed_scope=embed_scope)
    n_pixels, n_channels = carrier.shape

    if args.seed is not None and args.matrix_r is not None:
        seed, matrix_r = args.seed, args.matrix_r
    elif args.seed is not None:
        from hmf.search import choose_best_matrix_r

        matrix_r, _ = choose_best_matrix_r(
            cover, payload_bits, seed=args.seed,
            embed_scope=embed_scope, virtual_mode=virtual_mode, r_candidates=r_candidates,
        )
        seed = args.seed
    else:
        seed_candidates = range(args.seed_range)
        seed, matrix_r, eval_rows = choose_best_seed_and_matrix_r(
            cover,
            payload_bits,
            seed_candidates=seed_candidates,
            embed_scope=embed_scope,
            virtual_mode=virtual_mode,
            r_candidates=r_candidates,
            max_workers=args.workers,
            parallel=not args.no_parallel,
        )
        top = sorted(eval_rows, key=lambda x: (x[3], -x[1], x[0]))[:5]
        print(f"Mode key {args.virtual_key} ({KEY_LABELS[args.virtual_key]})")
        print("Kandidat terbaik (seed, r, flips):")
        for s, r, cap, flips, fpb in top:
            print(f"  seed={s:<3} r={r} cap={cap:,} flips={flips:,} fpb={fpb:.6f}")
        print(f"Terpilih: seed={seed}, r={matrix_r}\n")

    cap = matrix_capacity(n_pixels, matrix_r, n_channels)
    print(f"Cover   : {cover_path.name} {cover.shape}")
    print(f"Payload : {payload_path.name} ({len(payload_raw):,} bit raw, {len(payload_bits):,} bit embed)")
    print(f"Kapasitas: {cap:,} bit")

    stego, stats = embed_hamming_matchfirst_lsb(
        cover,
        payload_bits,
        seed=seed,
        matrix_r=matrix_r,
        embed_scope=embed_scope,
        virtual_mode=virtual_mode,
    )
    save_image(stego, stego_path)
    save_key(
        key_path,
        seed=seed,
        matrix_r=matrix_r,
        payload_len=len(payload_bits),
        shape=tuple(int(x) for x in cover.shape),
        mode=args.image_mode,
        embed_scope=embed_scope,
        virtual_mode=virtual_mode,
        virtual_key=args.virtual_key,
        virtual_choices=stats.virtual_choices,
        compression=str(huff_meta.get("compression", "none")),
        huffman_freq=huff_meta.get("huffman_freq"),
        original_bit_len=huff_meta.get("original_bit_len"),
    )

    loaded = load_key(key_path)
    recovered_emb = extract_hamming(
        stego,
        int(loaded["payload_len"]),
        seed=int(loaded["seed"]),
        matrix_r=int(loaded["matrix_r"]),
        embed_scope=str(loaded["embed_scope"]),
        virtual_mode=str(loaded["virtual_mode"]),
        virtual_choices=loaded.get("virtual_choices"),
    )
    recovered = huffman_expand_to_original_bits(recovered_emb, {
        "compression": loaded.get("compression", "none"),
        "original_bit_len": loaded.get("original_bit_len"),
        "huffman_freq": loaded.get("huffman_freq"),
    })
    write_payload_bits(extracted_path, recovered)

    mse, psnr_val = psnr(cover, stego)
    ssim_val = ssim(cover, stego)
    ok = recovered == payload_raw

    print("\n=== Hasil ===")
    print(f"Blok        : {stats.n_blocks:,} (tanpa flip: {stats.n_no_flip_blocks:,})")
    print(f"Flips       : {stats.n_flips:,} ({stats.flips_per_bit:.4%} per bit)")
    print(f"Matrix r    : {matrix_r} (n={matrix_block_n(matrix_r)} pixel, k={matrix_r})")
    print(f"Key mode    : {args.virtual_key} = {KEY_LABELS[args.virtual_key]}")
    print(f"MSE / PSNR  : {mse:.6f} / {psnr_val:.4f} dB" if np.isfinite(psnr_val) else f"MSE / PSNR  : {mse:.6f} / inf")
    print(f"SSIM        : {ssim_val:.6f}")
    print(f"Extract     : {'OK' if ok else 'GAGAL'}")
    print(f"Stego       : {stego_path}")
    print(f"Key         : {key_path}")


if __name__ == "__main__":
    main()
