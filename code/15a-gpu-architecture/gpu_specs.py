"""
gpu_specs.py  [CPU]
Approximate vendor peak figures for common AI accelerators, used by the other
scripts in this folder. Dense (non-sparse) throughput, in FLOP/s.

These are datasheet peaks, not what you will measure. Always check the current
datasheet for the exact SKU you rent (SXM vs PCIe, power cap, memory size).
"""

TB = 1e12
PF = 1e15

GPUS = {
    #            memory    HBM bandwidth   dense BF16      dense FP8       dense FP4       power
    "A100-80GB": {"mem_gb": 80,  "bw": 2.0 * TB,  "bf16": 312e12,  "fp8": None,    "fp4": None,    "watts": 400,  "sms": 108},
    "H100-SXM":  {"mem_gb": 80,  "bw": 3.35 * TB, "bf16": 989e12,  "fp8": 1979e12, "fp4": None,    "watts": 700,  "sms": 132},
    "H200":      {"mem_gb": 141, "bw": 4.8 * TB,  "bf16": 989e12,  "fp8": 1979e12, "fp4": None,    "watts": 700,  "sms": 132},
    "B200":      {"mem_gb": 180, "bw": 8.0 * TB,  "bf16": 2.25 * PF, "fp8": 4.5 * PF, "fp4": 9 * PF,  "watts": 1000, "sms": 148},
    "B300":      {"mem_gb": 288, "bw": 8.0 * TB,  "bf16": 2.25 * PF, "fp8": 4.5 * PF, "fp4": 15 * PF, "watts": 1100, "sms": 160},
    "MI300X":    {"mem_gb": 192, "bw": 5.3 * TB,  "bf16": 1.307 * PF, "fp8": 2.615 * PF, "fp4": None, "watts": 750, "sms": 304},
    "MI355X":    {"mem_gb": 288, "bw": 8.0 * TB,  "bf16": 2.5 * PF, "fp8": 5.0 * PF, "fp4": 10 * PF, "watts": 1400, "sms": 256},
    "Rubin (vendor, 2026)": {"mem_gb": 288, "bw": 22 * TB, "bf16": None, "fp8": None, "fp4": 50 * PF, "watts": None, "sms": None},
}

# Typical rental prices change monthly; fill in what you actually pay (USD per GPU-hour).
PRICE_PER_HOUR = {
    "A100-80GB": 1.40,
    "H100-SXM": 2.50,
    "H200": 3.50,
    "B200": 5.00,
}


def ridge_point(name, dtype="bf16"):
    """FLOPs per byte at which the chip switches from memory-bound to compute-bound."""
    g = GPUS[name]
    if g.get(dtype) is None:
        return None
    return g[dtype] / g["bw"]


if __name__ == "__main__":
    print(f"{'GPU':22s} {'mem':>6s} {'TB/s':>6s} {'BF16 TF':>8s} {'ridge BF16':>11s} {'ridge FP8':>10s} {'ridge FP4':>10s}")
    for name, g in GPUS.items():
        def fmt(x, w, d=0):
            return f"{x:{w}.{d}f}" if x is not None else f"{'-':>{w}s}"
        print(f"{name:22s} {g['mem_gb']:6d} {g['bw']/TB:6.2f} {fmt(g['bf16'] and g['bf16']/1e12, 8)} "
              f"{fmt(ridge_point(name,'bf16'), 11)} {fmt(ridge_point(name,'fp8'), 10)} {fmt(ridge_point(name,'fp4'), 10)}")
