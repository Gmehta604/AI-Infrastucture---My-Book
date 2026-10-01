"""
01_vector_add.py  [1 GPU, or CPU with TRITON_INTERPRET=1]
The smallest Triton kernel. Compare with vector_add.cu in code/16a: in Triton you
program one BLOCK of elements per program instance, not one element per thread.

Run:  python 01_vector_add.py            (GPU)
      TRITON_INTERPRET=1 python 01_vector_add.py   (CPU)
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, bench, check, report


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                 # which block this program handles
    offsets = pid * BLOCK + tl.arange(0, BLOCK) # a vector of BLOCK indices
    mask = offsets < n                          # guard the ragged last block
    x = tl.load(x_ptr + offsets, mask=mask)     # one vectorised, coalesced load
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)             # number of program instances
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    n = 98_432 if DEVICE == "cpu" else 1 << 26
    x = torch.randn(n, device=DEVICE)
    y = torch.randn(n, device=DEVICE)
    check("triton add", add(x, y), x + y, atol=1e-6)
    report([("triton", bench(lambda: add(x, y))), ("torch", bench(lambda: x + y))],
           bytes_or_flops=3 * n * 4)
