"""
03_rmsnorm.py  [1 GPU, or CPU with TRITON_INTERPRET=1]
Fused RMSNorm (used in Llama, Qwen, DeepSeek) with an optional fused residual add:
    h = x + residual            (optional)
    y = h / sqrt(mean(h^2) + eps) * weight
One read of each input and one write of each output, instead of ~6 passes in eager PyTorch.
Rows longer than BLOCK are handled with a loop, so any hidden size works.

Run:  python 03_rmsnorm.py
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, bench, check, report


@triton.jit
def rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, h_ptr, stride, n_cols, eps,
                   HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride
    # pass 1: sum of squares (and write h = x + residual if requested)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        h = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            h += tl.load(res_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(h_ptr + row * stride + cols, h, mask=mask)
        acc += h * h
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)
    # pass 2: normalise and scale (re-reads h; for rows that fit in one block this hits cache)
    src = h_ptr if HAS_RES else x_ptr
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        h = tl.load(src + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * stride + cols, (h * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, residual=None, eps=1e-6, BLOCK=1024):
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    rows, cols = x2.shape
    y = torch.empty_like(x2)
    h = torch.empty_like(x2) if residual is not None else x2
    res = residual.reshape(-1, cols).contiguous() if residual is not None else x2
    rmsnorm_kernel[(rows,)](x2, res, weight, y, h, x2.stride(0), cols, eps,
                            HAS_RES=residual is not None, BLOCK=min(BLOCK, triton.next_power_of_2(cols)))
    return (y.view_as(x), h.view_as(x)) if residual is not None else y.view_as(x)


def torch_rmsnorm(x, weight, residual=None, eps=1e-6):
    h = x + residual if residual is not None else x
    hf = h.float()
    y = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
    return (y, h) if residual is not None else y


if __name__ == "__main__":
    rows, d = (32, 1500) if DEVICE == "cpu" else (16384, 4096)
    dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    x = torch.randn(rows, d, device=DEVICE, dtype=dtype)
    r = torch.randn_like(x)
    w = torch.randn(d, device=DEVICE, dtype=dtype)
    check("rmsnorm", rmsnorm(x, w), torch_rmsnorm(x, w), atol=2e-2)
    y1, h1 = rmsnorm(x, w, residual=r)
    y2, h2 = torch_rmsnorm(x, w, residual=r)
    check("rmsnorm + residual (y)", y1, y2, atol=2e-2)
    check("rmsnorm + residual (h)", h1, h2, atol=2e-2)
    nbytes = 4 * x.numel() * x.element_size()   # read x, r; write y, h
    report([("triton fused", bench(lambda: rmsnorm(x, w, residual=r))),
            ("eager PyTorch", bench(lambda: torch_rmsnorm(x, w, residual=r)))],
           bytes_or_flops=nbytes)
