"""
float_formats.py  [CPU]
Floating-point formats from the bits up. Prints, for every format used in AI:
range, precision (epsilon), smallest normal and subnormal values, then decodes
example numbers bit by bit and lists every value FP4 (E2M1) and FP8 (E4M3) can hold.

Run:  python float_formats.py
"""
import math
import struct

import torch

# name: (exponent bits, mantissa bits, has_inf, special rule)
FORMATS = {
    "FP32":     (8, 23, True,  "ieee"),
    "TF32":     (8, 10, True,  "ieee"),   # stored in 32 bits; tensor cores use 19
    "BF16":     (8, 7,  True,  "ieee"),
    "FP16":     (5, 10, True,  "ieee"),
    "FP8 E5M2": (5, 2,  True,  "ieee"),
    "FP8 E4M3": (4, 3,  False, "e4m3fn"), # no inf; only S.1111.111 is NaN, so max is 448
    "FP6 E3M2": (3, 2,  False, "finite"),
    "FP6 E2M3": (2, 3,  False, "finite"),
    "FP4 E2M1": (2, 1,  False, "finite"),
}


def stats(e_bits, m_bits, rule):
    bias = 2 ** (e_bits - 1) - 1
    if rule == "ieee":
        max_exp = (2 ** e_bits - 2) - bias              # all-ones exponent reserved for inf/NaN
        max_val = (2 - 2 ** -m_bits) * 2 ** max_exp
    elif rule == "e4m3fn":
        max_exp = (2 ** e_bits - 1) - bias              # all-ones exponent usable...
        max_val = (2 - 2 * 2 ** -m_bits) * 2 ** max_exp # ...except mantissa 111 (NaN): 1.75 * 256 = 448
    else:
        max_exp = (2 ** e_bits - 1) - bias              # no specials at all
        max_val = (2 - 2 ** -m_bits) * 2 ** max_exp
    min_normal = 2.0 ** (1 - bias)
    min_sub = 2.0 ** (1 - bias - m_bits)
    eps = 2.0 ** -m_bits                                 # gap between 1 and the next number
    return bias, max_val, min_normal, min_sub, eps


def decode(bits, e_bits, m_bits):
    """Decode an integer bit pattern of a (1, e_bits, m_bits) float (finite values)."""
    bias = 2 ** (e_bits - 1) - 1
    sign = (bits >> (e_bits + m_bits)) & 1
    exp = (bits >> m_bits) & (2 ** e_bits - 1)
    man = bits & (2 ** m_bits - 1)
    if exp == 0:                                         # subnormal: no implicit leading 1
        val = (man / 2 ** m_bits) * 2 ** (1 - bias)
    else:
        val = (1 + man / 2 ** m_bits) * 2 ** (exp - bias)
    return (-1) ** sign * val, sign, exp, man


def show_bits(x):
    b = struct.unpack(">I", struct.pack(">f", x))[0]
    s = f"{b:032b}"
    fp32 = f"{s[0]} {s[1:9]} {s[9:]}"
    bf16 = f"{s[0]} {s[1:9]} {s[9:16]}"
    return fp32, bf16


if __name__ == "__main__":
    print(f"{'format':10s} {'bits':>4s} {'max':>12s} {'min normal':>11s} {'min subnormal':>14s} {'epsilon':>9s} {'~decimal digits':>15s}")
    for name, (e, m, _, rule) in FORMATS.items():
        bias, mx, mn, ms, eps = stats(e, m, rule)
        print(f"{name:10s} {1+e+m:4d} {mx:12.4g} {mn:11.3g} {ms:14.3g} {eps:9.3g} {-math.log10(eps):15.1f}")

    print("\nHow 0.1 is stored (sign | exponent | mantissa):")
    fp32, bf16 = show_bits(0.1)
    print("  FP32:", fp32, "=", f"{torch.tensor(0.1, dtype=torch.float32).item():.10f}")
    print("  BF16:", bf16, "        =", f"{torch.tensor(0.1).to(torch.bfloat16).float().item():.10f}  (top 16 bits of FP32)")
    print("  FP16:", f"{torch.tensor(0.1).to(torch.float16).float().item():.10f}")
    for dt, nm in [(torch.float8_e4m3fn, "E4M3"), (torch.float8_e5m2, "E5M2")]:
        print(f"  FP8 {nm}:", f"{torch.tensor(0.1).to(dt).float().item():.10f}")

    print("\nRounding to each format (value -> stored value):")
    for v in [1000.0, 70000.0, 1e-5, 1e-8, 3.14159265, 1 + 1 / 300]:
        row = [f"{v:>12g}"]
        for dt in [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]:
            stored = torch.tensor(v).to(dt).float().item()
            row.append(f"{stored:>12.6g}")
        print("  " + " ".join(row) + "   (BF16, FP16, E4M3, E5M2)")
    print("  note: E4M3 has no infinity; values above 448 become NaN on a plain cast in PyTorch,")
    print("  which is why FP8 code always scales first and clamps (saturates) to the format's max.")

    print("\nEvery non-negative FP4 E2M1 value:")
    vals = sorted({decode(b, 2, 1)[0] for b in range(8)})
    print("  ", vals)
    print("Every non-negative FP8 E4M3 value (126 finite positives + 0):")
    e4m3 = sorted({decode(b, 4, 3)[0] for b in range(127)})   # 0x7F is NaN
    print("   smallest:", e4m3[:6], " largest:", e4m3[-4:])
    gaps = [(b - a) / a for a, b in zip(e4m3[8:], e4m3[9:])]
    print(f"   relative gap between neighbours (normal range): {min(gaps):.3f} to {max(gaps):.3f}")
