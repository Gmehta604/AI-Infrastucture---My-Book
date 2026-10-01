"""
weight_quantization.py  [CPU]
Post-training weight quantization schemes used for serving, compared on a weight
matrix with LLM-like statistics (roughly Gaussian with occasional large values):

  INT8 per-tensor absmax       INT8 per-channel absmax
  INT4 per-channel              INT4 group-wise (groups of 128) symmetric and asymmetric
  NF4 (QLoRA's normal-float codebook, groups of 64)
  FP8 E4M3 per-channel          FP4 (NVFP4-style, blocks of 16)

Reports reconstruction error and the effect on the layer's OUTPUT for real-looking inputs,
which is what actually matters.

Run:  python weight_quantization.py
"""
import torch

torch.manual_seed(0)

# The 16 NF4 levels from the QLoRA paper (quantiles of a standard normal, normalised to [-1, 1]).
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def int_sym(w, bits, dim_groups):
    """Symmetric absmax integer quantization over groups along the input dimension."""
    out_f, in_f = w.shape
    g = dim_groups or in_f
    wg = w.reshape(out_f, in_f // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
    return (torch.round(wg / scale).clamp(-qmax - 1, qmax) * scale).reshape(out_f, in_f)


def int_asym(w, bits, g):
    """Asymmetric (min/max with zero-point) quantization, as in GPTQ/AWQ group formats."""
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // g, g)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-12) / qmax
    zero = torch.round(-lo / scale)
    q = torch.round(wg / scale + zero).clamp(0, qmax)
    return ((q - zero) * scale).reshape(out_f, in_f)


def codebook(w, levels, g):
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // g, g)
    absmax = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    idx = ((wg / absmax).unsqueeze(-1) - levels).abs().argmin(-1)
    return (levels[idx] * absmax).reshape(out_f, in_f)


def fp4_blocks(w, g=16):
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // g, g)
    scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 6.0
    scale = scale.clamp(max=448).to(torch.float8_e4m3fn).float()   # block scale stored in FP8
    mag = (wg / scale).abs().clamp(max=6.0)
    idx = (mag.unsqueeze(-1) - FP4_GRID).abs().argmin(-1)
    return (torch.sign(wg) * FP4_GRID[idx] * scale).reshape(out_f, in_f)


def fp8_channel(w):
    s = w.abs().amax(1, keepdim=True) / 448.0
    return (w / s).to(torch.float8_e4m3fn).float() * s


if __name__ == "__main__":
    out_f, in_f = 2048, 4096
    w = torch.randn(out_f, in_f) * 0.02
    mask = torch.rand_like(w) < 0.001
    w[mask] *= 8                                   # rare large weights
    x = torch.randn(512, in_f)                      # inputs to the layer
    x[:, torch.randperm(in_f)[:16]] *= 20           # with outlier channels
    y_ref = x @ w.T

    methods = [
        ("INT8 per-tensor", lambda w: int_sym(w, 8, None) if False else int_sym(w.flatten()[None], 8, None).reshape(w.shape), 8.0),
        ("INT8 per-channel", lambda w: int_sym(w, 8, None), 8.0),
        ("FP8 E4M3 per-channel", fp8_channel, 8.0),
        ("INT4 per-channel", lambda w: int_sym(w, 4, None), 4.0),
        ("INT4 group-128 symmetric", lambda w: int_sym(w, 4, 128), 4 + 16 / 128),
        ("INT4 group-128 asymmetric", lambda w: int_asym(w, 4, 128), 4 + 32 / 128),
        ("NF4 group-64 (QLoRA)", lambda w: codebook(w, NF4, 64), 4 + 16 / 64),
        ("FP4 E2M1 blocks of 16, FP8 scale", fp4_blocks, 4 + 8 / 16),
    ]
    print(f"{'method':34s} {'bits/weight':>11s} {'weight err':>11s} {'output err':>11s} {'memory (GB) for 70B':>20s}")
    for name, fn, bits in methods:
        wq = fn(w)
        we = ((wq - w).norm() / w.norm()).item()
        oe = ((x @ wq.T - y_ref).norm() / y_ref.norm()).item()
        print(f"{name:34s} {bits:11.2f} {we:11.4f} {oe:11.4f} {70e9 * bits / 8 / 1e9:20.1f}")
    print("\nGroup-wise scales cost a fraction of a bit per weight but cut error sharply.")
    print("Methods like GPTQ and AWQ further reduce OUTPUT error by using calibration data (Deep Dive 19b).")
