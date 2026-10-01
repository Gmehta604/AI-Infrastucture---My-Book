"""
scaling.py  [CPU]
FP8 and FP4 have tiny ranges, so tensors are multiplied by a scale before casting.
This script compares scaling granularities on an activation-like tensor that has a few
large outlier channels (as real LLM activations do):

  per-tensor   one scale for the whole tensor
  per-row      one scale per token (row)
  1x128 tiles  DeepSeek-V3's activation scheme
  MXFP8        blocks of 32 values sharing a power-of-two (E8M0) scale, FP8 E4M3 elements
  MXFP4        blocks of 32, power-of-two scale, FP4 E2M1 elements
  NVFP4        blocks of 16, FP8 E4M3 scale per block (+ one FP32 tensor scale), FP4 elements

Run:  python scaling.py
"""
import torch

torch.manual_seed(0)
E4M3_MAX, E2M1_MAX = 448.0, 6.0
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def to_fp4(x):
    """Round to the nearest E2M1 value (sign kept separately)."""
    mag = x.abs().clamp(max=E2M1_MAX)
    idx = (mag.unsqueeze(-1) - FP4_GRID).abs().argmin(-1)
    return torch.sign(x) * FP4_GRID[idx]


def to_e4m3(x):
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def quant_blocks(x, block, elem, scale_kind):
    """Quantize the last dim in blocks. elem: 'e4m3' or 'fp4'. scale_kind: 'fp32', 'pow2', 'e4m3'."""
    rows, cols = x.shape
    xb = x.reshape(rows, cols // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    emax = E4M3_MAX if elem == "e4m3" else E2M1_MAX
    scale = amax / emax                                   # maps the block's max onto the format's max
    if scale_kind == "pow2":                              # MX: scale must be a power of two (E8M0)
        scale = 2.0 ** torch.ceil(torch.log2(scale))
    elif scale_kind == "e4m3":                            # NVFP4: scale itself stored in FP8 E4M3
        tensor_scale = scale.amax() / E4M3_MAX            # second-level FP32 scale keeps it in range
        scale = to_e4m3(scale / tensor_scale) * tensor_scale
    q = to_e4m3(xb / scale) if elem == "e4m3" else to_fp4(xb / scale)
    return (q * scale).reshape(rows, cols)


def per_tensor(x, elem):
    emax = E4M3_MAX if elem == "e4m3" else E2M1_MAX
    s = x.abs().max() / emax
    return (to_e4m3(x / s) if elem == "e4m3" else to_fp4(x / s)) * s


def per_row(x, elem):
    emax = E4M3_MAX if elem == "e4m3" else E2M1_MAX
    s = x.abs().amax(1, keepdim=True) / emax
    return (to_e4m3(x / s) if elem == "e4m3" else to_fp4(x / s)) * s


def rel_err(q, x):
    return ((q - x).norm() / x.norm()).item()


if __name__ == "__main__":
    tokens, channels = 256, 4096
    x = torch.randn(tokens, channels)
    outliers = torch.randperm(channels)[:8]
    x[:, outliers] *= 60.0                                 # a few "massive activation" channels
    x[torch.randint(0, tokens, (4,))] *= 5.0               # and a few unusually large tokens

    print("Relative error ||Q(x) - x|| / ||x||  (lower is better)\n")
    rows = [
        ("FP8 E4M3, per-tensor scale", per_tensor(x, "e4m3")),
        ("FP8 E4M3, per-row scale", per_row(x, "e4m3")),
        ("FP8 E4M3, 1x128 tiles (DeepSeek-V3)", quant_blocks(x, 128, "e4m3", "fp32")),
        ("MXFP8: 32-blocks, power-of-2 scale", quant_blocks(x, 32, "e4m3", "pow2")),
        ("FP4, per-tensor scale", per_tensor(x, "fp4")),
        ("FP4, per-row scale", per_row(x, "fp4")),
        ("MXFP4: 32-blocks, power-of-2 scale", quant_blocks(x, 32, "fp4", "pow2")),
        ("NVFP4: 16-blocks, E4M3 scale", quant_blocks(x, 16, "fp4", "e4m3")),
    ]
    for name, q in rows:
        print(f"  {name:38s} {rel_err(q, x):.4f}")

    print("\nBits per value including scale overhead:")
    print("  FP8 per-tensor ~8.0 | MXFP8 8 + 8/32 = 8.25 | MXFP4 4 + 8/32 = 4.25 | NVFP4 4 + 8/16 = 4.5")

    print("\nWhat outliers do to per-tensor FP8 on the NON-outlier channels:")
    normal = torch.ones(channels, dtype=torch.bool)
    normal[outliers] = False
    for name, q in rows[:3]:
        print(f"  {name:38s} error on ordinary channels {rel_err(q[:, normal], x[:, normal]):.4f}")
