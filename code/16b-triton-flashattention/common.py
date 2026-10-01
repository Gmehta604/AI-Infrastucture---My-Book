"""
common.py: helpers shared by the scripts in this folder.

Every Triton script runs in two modes:
  * on an NVIDIA (or AMD ROCm) GPU: real compiled kernels, with benchmarks;
  * on a laptop without a GPU: set TRITON_INTERPRET=1 and Triton executes the kernel
    in NumPy on the CPU. Slow, but perfect for learning and debugging (you can even
    use print() and breakpoints inside kernels). Benchmarks are skipped in this mode.
"""
import os

import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() and not INTERPRET else "cpu"

if DEVICE == "cpu" and not INTERPRET:
    raise SystemExit(
        "No GPU found. Run in CPU interpreter mode instead:\n"
        "  TRITON_INTERPRET=1 python " + os.path.basename(__import__("sys").argv[0])
    )


def check(name, got, ref, atol=1e-2, rtol=1e-2):
    err = (got.float() - ref.float()).abs().max().item()
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: max abs error {err:.2e}")
    if not ok:
        raise SystemExit(1)


def bench(fn):
    """Median runtime in milliseconds on GPU; None in interpreter mode."""
    if DEVICE != "cuda":
        return None
    import triton.testing
    return triton.testing.do_bench(fn)


def report(rows, bytes_or_flops=None, unit="GB/s"):
    """rows: list of (name, ms). Prints a table; skipped on CPU."""
    if DEVICE != "cuda":
        print("(benchmarks skipped: run on a GPU to see speed)")
        return
    for name, ms in rows:
        extra = ""
        if bytes_or_flops is not None:
            scale = 1e6 if unit == "GB/s" else 1e9
            extra = f"  {bytes_or_flops / ms / scale:9.1f} {unit}"
        print(f"  {name:28s} {ms:8.3f} ms{extra}")
