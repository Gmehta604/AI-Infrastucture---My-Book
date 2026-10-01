"""
fp8_matmul.py  [CPU simulation; uses real FP8 tensor cores on H100/H200/B200 if available]
Simulates an FP8 GEMM the way training frameworks do it: quantize inputs with scales,
multiply in FP8, accumulate in FP32, multiply the result by the scales.
Compares per-tensor scaling against DeepSeek-V3-style fine-grained scaling
(1x128 tiles for activations, 128x128 blocks for weights), relative to a BF16 GEMM.

Run:  python fp8_matmul.py
"""
import torch

torch.manual_seed(0)
E4M3_MAX = 448.0


def q_e4m3(x):
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def fp8_per_tensor(a, b):
    sa = a.abs().max() / E4M3_MAX
    sb = b.abs().max() / E4M3_MAX
    return (q_e4m3(a / sa) @ q_e4m3(b / sb)) * (sa * sb)


def fp8_blockwise(a, b, tile=128):
    """a: (M, K) activations, scaled per 1 x tile along K.
       b: (K, N) weights, scaled per tile x tile block.
       Each K-chunk's partial product is rescaled and accumulated in FP32 (like DeepSeek-V3,
       which promotes partial sums to FP32 every 128 elements of K)."""
    M, K = a.shape
    N = b.shape[1]
    out = torch.zeros(M, N)
    for k0 in range(0, K, tile):
        a_t = a[:, k0:k0 + tile]
        sa = a_t.abs().amax(1, keepdim=True).clamp(min=1e-12) / E4M3_MAX       # (M, 1)
        qa = q_e4m3(a_t / sa)
        part = torch.zeros(M, N)
        for n0 in range(0, N, tile):
            b_t = b[k0:k0 + tile, n0:n0 + tile]
            sb = b_t.abs().max().clamp(min=1e-12) / E4M3_MAX                    # scalar per block
            part[:, n0:n0 + tile] = (qa @ q_e4m3(b_t / sb)) * sb
        out += part * sa                                                        # FP32 accumulation
    return out


def rel(x, ref):
    return ((x - ref).norm() / ref.norm()).item()


if __name__ == "__main__":
    M, K, N = 512, 2048, 1024
    a = torch.randn(M, K)
    a[:, torch.randperm(K)[:6]] *= 50.0          # activation outlier channels
    b = torch.randn(K, N) * 0.02                 # weights: small, well behaved
    ref = a.double() @ b.double()
    bf16 = (a.bfloat16() @ b.bfloat16()).double()

    print(f"GEMM {M}x{K} @ {K}x{N}, relative error vs float64:")
    print(f"  BF16 inputs                    {rel(bf16, ref):.2e}")
    print(f"  FP8, per-tensor scales         {rel(fp8_per_tensor(a, b).double(), ref):.2e}")
    print(f"  FP8, 1x128 / 128x128 scales    {rel(fp8_blockwise(a, b).double(), ref):.2e}")

    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9):
        # Real FP8 tensor cores via torch._scaled_mm (a private API; its signature has changed between versions).
        dev = "cuda"
        A, B = a.to(dev), b.to(dev)
        sa = (A.abs().max() / E4M3_MAX).float()
        sb = (B.abs().max() / E4M3_MAX).float()
        qa = (A / sa).to(torch.float8_e4m3fn)
        qb = (B / sb).to(torch.float8_e4m3fn).t().contiguous().t()   # B must be column-major
        try:
            out = torch._scaled_mm(qa, qb, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            print(f"  FP8 tensor cores (_scaled_mm)  {rel(out.double().cpu(), ref):.2e}")
        except Exception as e:  # noqa: BLE001
            print("  torch._scaled_mm unavailable in this PyTorch version:", type(e).__name__)
    else:
        print("  (no FP8-capable GPU found: Ada, Hopper or Blackwell needed for real FP8 tensor cores)")
