"""
accumulation.py  [CPU]
Why every low-precision kernel accumulates in FP32.
  1. Summing many small numbers in BF16/FP16: once the running total is large,
     each new term is smaller than half a unit in the last place and is lost ("swamping").
  2. The same sum with an FP32 accumulator, with Kahan compensated summation, and with
     stochastic rounding (the trick some low-precision training recipes use).
  3. A weight update that is too small to change a BF16 weight at all.

Run:  python accumulation.py
"""
import numpy as np
import torch

torch.manual_seed(0)


def running_sum(values, dtype):
    acc = torch.zeros((), dtype=dtype)
    for v in values.to(dtype):
        acc = (acc + v).to(dtype)
    return acc.float().item()


def kahan_sum(values, dtype):
    s = torch.zeros((), dtype=dtype)
    c = torch.zeros((), dtype=dtype)               # running compensation for lost low bits
    for v in values.to(dtype):
        y = (v - c).to(dtype)
        t = (s + y).to(dtype)
        c = ((t - s) - y).to(dtype)
        s = t
    return s.float().item()


def stochastic_round_bf16(x):
    """Round FP32 -> BF16 up or down at random, with probability proportional to distance.
    Unbiased: the expected value of the rounded number equals the input."""
    bits = x.view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32)
    return ((bits + noise) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


def stochastic_sum(values):
    acc = torch.zeros((), dtype=torch.float32)
    for v in values:
        acc = stochastic_round_bf16((acc + v).float().reshape(1))[0].float()
    return acc.item()


if __name__ == "__main__":
    n = 20_000
    vals = torch.full((n,), 0.01)                  # true sum = 200
    print(f"Sum of {n:,} copies of 0.01 (true answer 200):")
    print(f"  FP32 accumulator           {running_sum(vals, torch.float32):10.4f}")
    print(f"  FP16 accumulator           {running_sum(vals, torch.float16):10.4f}")
    print(f"  BF16 accumulator           {running_sum(vals, torch.bfloat16):10.4f}   <- stalls once each step rounds away")
    print(f"  BF16 + Kahan compensation  {kahan_sum(vals, torch.bfloat16):10.4f}")
    print(f"  BF16 + stochastic rounding {stochastic_sum(vals):10.4f}   (unbiased on average)")

    print("\nA BF16 weight of 1.0 receiving updates of size lr * grad:")
    for upd in [1e-2, 4e-3, 3e-3, 1e-3, 1e-4]:
        w = torch.tensor(1.0, dtype=torch.bfloat16)
        w_new = (w - upd).to(torch.bfloat16)
        print(f"  update {upd:7.0e}: new weight {w_new.item():.6f}  {'(unchanged!)' if w_new == w else ''}")
    print("  BF16's spacing just below 1.0 is 2^-8 ~ 0.0039: smaller updates vanish. This is why")
    print("  optimizers keep an FP32 master copy of the weights (or use stochastic rounding).")

    print("\nDot product of length 4096 (random inputs), relative error vs float64:")
    a = np.random.default_rng(1).standard_normal(4096)
    b = np.random.default_rng(2).standard_normal(4096)
    ref = float(np.dot(a, b))
    ta, tb = torch.tensor(a), torch.tensor(b)
    for name, dt, acc_dt in [("BF16 inputs, FP32 accumulate", torch.bfloat16, torch.float32),
                             ("BF16 inputs, BF16 accumulate", torch.bfloat16, torch.bfloat16),
                             ("FP8 E4M3 inputs, FP32 accumulate", torch.float8_e4m3fn, torch.float32)]:
        prod = ta.to(dt).float() * tb.to(dt).float()
        acc = torch.zeros((), dtype=acc_dt)
        for p in prod:
            acc = (acc + p.to(acc_dt)).to(acc_dt)
        print(f"  {name:34s} {abs(acc.item() - ref) / abs(ref):.2e}")
