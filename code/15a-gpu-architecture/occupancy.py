"""
occupancy.py  [CPU]
A simplified CUDA occupancy calculator for an H100-class SM: how many warps can be
resident, and which resource (threads, registers, shared memory, block slots) limits it.

Run:  python occupancy.py --threads 256 --regs 64 --smem-kb 48
"""
import argparse

# Per-SM limits for compute capability 9.0 (H100). Allocation granularities simplified.
SM = {
    "max_threads": 2048,
    "max_warps": 64,
    "max_blocks": 32,
    "registers": 65536,
    "smem_kb": 228,
    "reg_alloc_unit": 256,   # registers are allocated per warp in chunks of 256
}


def occupancy(threads_per_block, regs_per_thread, smem_kb_per_block):
    warps_per_block = -(-threads_per_block // 32)
    regs_per_warp = -(-(regs_per_thread * 32) // SM["reg_alloc_unit"]) * SM["reg_alloc_unit"]
    limits = {
        "threads": SM["max_threads"] // threads_per_block,
        "warps": SM["max_warps"] // warps_per_block,
        "block slots": SM["max_blocks"],
        "registers": SM["registers"] // (regs_per_warp * warps_per_block),
        "shared memory": int(SM["smem_kb"] // smem_kb_per_block) if smem_kb_per_block else SM["max_blocks"],
    }
    blocks = min(limits.values())
    limiter = min(limits, key=limits.get)
    warps = blocks * warps_per_block
    return blocks, warps, warps / SM["max_warps"], limiter, limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--regs", type=int, default=64)
    ap.add_argument("--smem-kb", type=float, default=0)
    a = ap.parse_args()
    blocks, warps, occ, limiter, limits = occupancy(a.threads, a.regs, a.smem_kb)
    print(f"block of {a.threads} threads, {a.regs} regs/thread, {a.smem_kb} KB smem")
    for k, v in limits.items():
        print(f"  blocks allowed by {k:14s}: {v}")
    print(f"=> {blocks} resident blocks, {warps} warps, occupancy {occ:.0%} (limited by {limiter})")

    print("\nHow register use changes occupancy (256 threads, no smem):")
    for r in [32, 64, 96, 128, 168, 255]:
        _, w, o, lim, _ = occupancy(256, r, 0)
        print(f"  {r:3d} regs/thread -> {w:2d} warps ({o:4.0%}), limited by {lim}")


if __name__ == "__main__":
    main()
