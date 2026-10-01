"""
04_matmul.py  [1 GPU, or CPU with TRITON_INTERPRET=1]
A tensor-core matmul in ~40 lines of Triton, with:
  * block tiling (the compiler handles shared memory, pipelining and tensor cores),
  * grouped ("swizzled") program ordering for better L2 cache reuse,
  * autotuning over tile sizes, stages and warps,
  * a fused activation epilogue (bias-free GELU-tanh or leaky ReLU) applied in registers.

Run:  python 04_matmul.py
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, bench, check, report

CONFIGS = [
    triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 64, "BN": 256, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
]
if INTERPRET:                                          # keep CPU runs quick
    CONFIGS = [triton.Config({"BM": 32, "BN": 32, "BK": 32, "GROUP_M": 4})]


@triton.jit
def gelu_tanh(x):
    z = 0.7978845608 * (x + 0.044715 * x * x * x)     # sqrt(2/pi) * (x + 0.044715 x^3)
    return 0.5 * x * (2 * tl.sigmoid(2 * z))            # 1 + tanh(z) == 2 * sigmoid(2z)


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])   # re-tune when shapes change
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  ACTIVATION: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    # --- grouped ordering: programs that run together share rows of A and columns of B in L2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_m = group_id * GROUP_M
    rows_in_group = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group

    # --- pointers to the first K-tile of A and B for this output tile
    offs_m = (pid_m * BM + tl.arange(0, BM)) % M
    offs_n = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)          # accumulate in FP32
    for k in range(0, tl.cdiv(K, BK)):
        k_mask = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)                          # tensor cores on GPU
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    # --- fused epilogue: the activation is applied while the tile is still in registers
    if ACTIVATION == "gelu":
        acc = gelu_tanh(acc)
    elif ACTIVATION == "leaky_relu":
        acc = tl.where(acc >= 0, acc, 0.01 * acc)
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b, activation=""):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), ACTIVATION=activation)
    return c


if __name__ == "__main__":
    import torch.nn.functional as F
    if DEVICE == "cpu":
        M, N, K, dtype = 96, 80, 70, torch.float32
    else:
        M = N = K = 8192
        dtype = torch.float16
    a = torch.randn(M, K, device=DEVICE, dtype=dtype)
    b = torch.randn(K, N, device=DEVICE, dtype=dtype)
    tol = 1e-3 if dtype == torch.float32 else 1e-1
    check("matmul", matmul(a, b), a @ b, atol=tol * K ** 0.5, rtol=1e-2)
    check("matmul + fused GELU", matmul(a, b, "gelu"), F.gelu(a @ b, approximate="tanh"), atol=tol * K ** 0.5, rtol=1e-2)
    flops = 2 * M * N * K
    report([("triton matmul", bench(lambda: matmul(a, b))),
            ("cuBLAS (torch)", bench(lambda: a @ b)),
            ("triton matmul + GELU", bench(lambda: matmul(a, b, "gelu"))),
            ("torch matmul then GELU", bench(lambda: F.gelu(a @ b, approximate="tanh")))],
           bytes_or_flops=flops, unit="TFLOP/s")
    if DEVICE == "cuda":
        print("best config:", matmul_kernel.best_config)
