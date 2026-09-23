"""
microbench.py  [1 GPU recommended; runs on CPU too, slowly]
Measure what your hardware actually delivers, and compare with the datasheet:
  1. memory bandwidth (large tensor copy)
  2. matmul throughput across sizes and dtypes
  3. an elementwise op vs a matmul: memory-bound vs compute-bound
Results are printed and saved to microbench.csv.

Run:  python microbench.py            (auto-detects CUDA)
      python microbench.py --quick    (smaller sizes, for CPU/laptops)
"""
import argparse
import csv
import time

import torch


def timer(fn, iters, device):
    """Median time of fn() in seconds, with warm-up and proper GPU synchronisation."""
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        if device == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        else:
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"device: {name}\n")
    rows = []

    # 1. Memory bandwidth: y.copy_(x) reads N bytes and writes N bytes.
    n_bytes = (256 if args.quick else 2048) * 2**20
    x = torch.empty(n_bytes // 4, dtype=torch.float32, device=device).uniform_()
    y = torch.empty_like(x)
    t = timer(lambda: y.copy_(x), 10, device)
    bw = 2 * n_bytes / t
    print(f"[bandwidth] copy {n_bytes/2**20:.0f} MiB: {bw/1e9:8.1f} GB/s")
    rows.append(["bandwidth_copy", n_bytes, f"{bw/1e9:.1f} GB/s"])
    del x, y

    # 2. Matmul sweep
    dtypes = [torch.float32]
    if device == "cuda":
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        torch.backends.cuda.matmul.allow_tf32 = False
    sizes = [256, 512, 1024, 2048] if args.quick else [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    for dt in dtypes:
        for s in sizes:
            a = torch.randn(s, s, device=device, dtype=dt)
            b = torch.randn(s, s, device=device, dtype=dt)
            t = timer(lambda: a @ b, 10 if s <= 4096 else 5, device)
            tflops = 2 * s**3 / t / 1e12
            print(f"[matmul] {str(dt):15s} {s:5d}^3: {tflops:8.2f} TFLOP/s")
            rows.append([f"matmul_{dt}", s, f"{tflops:.2f} TFLOP/s"])

    # 3. Memory-bound vs compute-bound: same data, very different FLOP rates
    s = 2048 if args.quick else 8192
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    a = torch.randn(s, s, device=device, dtype=dt)
    b = torch.randn(s, s, device=device, dtype=dt)
    t_add = timer(lambda: a + b, 10, device)
    t_mm = timer(lambda: a @ b, 5, device)
    print(f"\n[contrast] {s}x{s} add:    {s*s/t_add/1e12:8.3f} TFLOP/s  (1 FLOP per 6 bytes -> memory-bound)")
    print(f"[contrast] {s}x{s} matmul: {2*s**3/t_mm/1e12:8.3f} TFLOP/s  (~{s/3:.0f} FLOPs per byte -> compute-bound)")
    rows.append(["add", s, f"{s*s/t_add/1e12:.3f} TFLOP/s"])
    rows.append(["matmul", s, f"{2*s**3/t_mm/1e12:.3f} TFLOP/s"])

    with open("microbench.csv", "w", newline="") as f:
        csv.writer(f).writerows([["test", "size", "result"]] + rows)
    print("\nsaved microbench.csv; compare with the datasheet peaks in gpu_specs.py")


if __name__ == "__main__":
    main()
