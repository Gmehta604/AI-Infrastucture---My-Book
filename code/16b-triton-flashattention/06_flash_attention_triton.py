"""
06_flash_attention_triton.py  [1 GPU, or CPU with TRITON_INTERPRET=1]
A FlashAttention forward kernel in Triton (the FlashAttention-2 structure):
  * one program per (block of BM query rows, batch x head),
  * loop over key/value blocks with online softmax, accumulators in registers,
  * causal masking that skips blocks entirely above the diagonal,
  * exp2 with a pre-multiplied log2(e) scale (cheaper than exp on GPUs).
Compared against PyTorch's scaled_dot_product_attention and a naive implementation.

Run:  python 06_flash_attention_triton.py
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import DEVICE, bench, check, report


@triton.jit
def attn_fwd(Q, K, V, O, stride_bh, stride_t, T, sm_scale,
             D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)                       # which block of query rows
    bh = tl.program_id(1)                            # which (batch, head)
    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    base = bh * stride_bh

    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)  # (BM, D), stays in registers

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634         # fold log2(e) in, then use exp2

    hi = T
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BM, T)       # no keys after the last query in this block
    for start_n in range(0, hi, BN):
        cols = start_n + offs_n
        k = tl.load(K + base + cols[None, :] * stride_t + offs_d[:, None],
                    mask=cols[None, :] < T, other=0.0)          # (D, BN): K transposed
        qk = tl.dot(q, k) * qk_scale                              # (BM, BN)
        valid = cols[None, :] < T
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + base + cols[:, None] * stride_t + offs_d[None, :],
                    mask=cols[:, None] < T, other=0.0)          # (BN, D)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)


def flash_attention(q, k, v, causal=False, BM=64, BN=64):
    """q, k, v: (batch, heads, T, D), contiguous, D a power of two >= 16."""
    B, H, T, D = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(T, BM), B * H)
    attn_fwd[grid](q, k, v, o, T * D, D, T, 1.0 / math.sqrt(D),
                   D=D, BM=BM, BN=BN, CAUSAL=causal, num_warps=4, num_stages=2)
    return o


def naive_attention(q, k, v, causal=False):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])     # materialises (T, T)
    if causal:
        T = q.shape[-2]
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1), float("-inf"))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v


if __name__ == "__main__":
    if DEVICE == "cpu":
        B, H, T, D, dtype = 1, 2, 200, 32, torch.float32
    else:
        B, H, T, D, dtype = 4, 32, 4096, 64, torch.float16
    q, k, v = (torch.randn(B, H, T, D, device=DEVICE, dtype=dtype) for _ in range(3))
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    for causal in [False, True]:
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        check(f"flash attention (causal={causal})", flash_attention(q, k, v, causal), ref, atol=tol)

    if DEVICE == "cuda":
        flops = 4 * B * H * T * T * D / 2                      # causal: half the matrix
        report([("triton flash (causal)", bench(lambda: flash_attention(q, k, v, True))),
                ("torch SDPA (causal)", bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))),
                ("naive (causal)", bench(lambda: naive_attention(q, k, v, True)))],
               bytes_or_flops=flops, unit="TFLOP/s")
        torch.cuda.reset_peak_memory_stats()
        naive_attention(q, k, v, True); naive_mem = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        flash_attention(q, k, v, True); flash_mem = torch.cuda.max_memory_allocated()
        print(f"peak memory: naive {naive_mem/1e9:.2f} GB, flash {flash_mem/1e9:.2f} GB")
    else:
        report([])
