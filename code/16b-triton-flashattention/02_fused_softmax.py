"""
02_fused_softmax.py  [1 GPU, or CPU with TRITON_INTERPRET=1]
Row-wise softmax in one pass over memory. Each program loads a whole row into
registers (SRAM), computes max, exp, sum and divide, and writes the row once.
A softmax written as separate PyTorch ops reads and writes the matrix several times.

Run:  python 02_fused_softmax.py
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, bench, check, report


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                  # numerical stability
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, y, mask=mask)


def softmax(x):
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)       # a row must fit in one block
    num_warps = 4 if BLOCK <= 2048 else 8 if BLOCK <= 4096 else 16
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols, BLOCK=BLOCK, num_warps=num_warps)
    return out


def naive_softmax(x):
    """What you get writing softmax by hand in PyTorch: 5 kernels, ~8 passes over memory."""
    m = x.max(dim=1, keepdim=True).values      # read x
    z = x - m                                  # read x, write z
    num = torch.exp(z)                         # read z, write num
    den = num.sum(dim=1, keepdim=True)         # read num
    return num / den                           # read num, write out


if __name__ == "__main__":
    rows, cols = (64, 781) if DEVICE == "cpu" else (8192, 4096)
    x = torch.randn(rows, cols, device=DEVICE)
    check("triton softmax", softmax(x), torch.softmax(x, dim=1), atol=1e-5)
    min_bytes = 2 * x.numel() * 4              # read once + write once
    report([("triton fused", bench(lambda: softmax(x))),
            ("torch.softmax", bench(lambda: torch.softmax(x, dim=1))),
            ("naive PyTorch ops", bench(lambda: naive_softmax(x)))],
           bytes_or_flops=min_bytes)
