"""
wave_quantization.py  [CPU]
Why a matmul of size 1536 can be slower per FLOP than one of size 2048.
A GEMM is split into output tiles; each SM computes one tile at a time. If the
number of tiles is not a multiple of the SM count, the last "wave" runs partly empty.

Run:  python wave_quantization.py --sms 132 --tile 128 128
"""
import argparse
import math


def efficiency(M, N, tile_m, tile_n, sms, k_pad=None):
    tiles_m, tiles_n = math.ceil(M / tile_m), math.ceil(N / tile_n)
    tiles = tiles_m * tiles_n
    waves = math.ceil(tiles / sms)
    wave_eff = tiles / (waves * sms)                       # fraction of SM-slots doing work
    tile_eff = (M * N) / (tiles_m * tile_m * tiles_n * tile_n)  # padding waste inside edge tiles
    return tiles, waves, wave_eff, tile_eff, wave_eff * tile_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", type=int, default=132)
    ap.add_argument("--tile", type=int, nargs=2, default=[128, 128])
    args = ap.parse_args()
    tm, tn = args.tile

    print(f"{args.sms} SMs, {tm}x{tn} output tiles\n")
    print(f"{'M=N':>6s} {'tiles':>6s} {'waves':>6s} {'wave eff':>9s} {'tile eff':>9s} {'overall':>8s}")
    for size in [1024, 1152, 1408, 1536, 1664, 2048, 2176, 4096, 4097, 4224, 8192]:
        t, w, we, te, o = efficiency(size, size, tm, tn, args.sms)
        print(f"{size:6d} {t:6d} {w:6d} {we:9.1%} {te:9.1%} {o:8.1%}")

    print("\nVocabulary padding example (tokens x vocab output, 8192 tokens):")
    for vocab in [50257, 50304, 128256, 128000]:
        t, w, we, te, o = efficiency(8192, vocab, tm, tn, args.sms)
        print(f"  vocab {vocab:6d}: tile eff {te:6.2%}, wave eff {we:6.2%}, divisible by 64: {vocab % 64 == 0}")


if __name__ == "__main__":
    main()
