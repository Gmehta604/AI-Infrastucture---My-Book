"""
07_torch_compile.py  [CPU or GPU]
What torch.compile actually does to a small Transformer block:
  1. counts graph breaks with torch._dynamo.explain,
  2. shows a function that breaks the graph (a .item() call and a print) and a fixed version,
  3. times eager vs compiled,
  4. tells you how to dump the generated Triton (GPU) or C++ (CPU) code.

Run:  python 07_torch_compile.py
See generated code:  TORCH_LOGS="output_code" python 07_torch_compile.py 2>&1 | less
See graph breaks:    TORCH_LOGS="graph_breaks" python 07_torch_compile.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Block(nn.Module):
    def __init__(self, d=1024, hidden=2816):
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(d))
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm_w   # RMSNorm: 4 ops
        return x + self.down(F.silu(self.gate(h)) * self.up(h))                     # SwiGLU + residual


def leaky_step(x):
    """Two classic graph breaks: a data-dependent Python branch and a print."""
    scale = x.abs().max().item()          # .item() pulls a value to Python -> graph break
    print("scale", round(scale, 2))       # side effect -> graph break
    if scale > 3:
        x = x / scale
    return F.gelu(x) * 2


def fixed_step(x):
    """Same maths with tensor ops only: no break, fully fusible."""
    scale = x.abs().amax()
    x = torch.where(scale > 3, x / scale, x)
    return F.gelu(x) * 2


def timeit(fn, x, iters=20):
    for _ in range(3):
        fn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(512, 1024, device=device)
    exp = torch._dynamo.explain(leaky_step)(x)
    print(f"leaky_step: {exp.graph_count} graphs, {exp.graph_break_count} graph breaks")
    for reason in exp.break_reasons:
        print("   break:", str(reason.reason).splitlines()[0][:100])
    exp = torch._dynamo.explain(fixed_step)(x)
    print(f"fixed_step: {exp.graph_count} graph, {exp.graph_break_count} graph breaks\n")

    block = Block().to(device).eval()
    x = torch.randn(8, 512, 1024, device=device)
    with torch.no_grad():
        compiled = torch.compile(block)
        t0 = time.perf_counter()
        y = compiled(x)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"first compiled call (includes compilation): {time.perf_counter() - t0:.1f} s")
        print("max difference vs eager:", (y - block(x)).abs().max().item())
        print(f"eager:    {timeit(block, x):7.2f} ms")
        print(f"compiled: {timeit(compiled, x):7.2f} ms")
        if device == "cuda":
            ro = torch.compile(block, mode="reduce-overhead")   # adds CUDA graphs
            print(f"reduce-overhead (CUDA graphs): {timeit(ro, x):7.2f} ms")
    print("\nTo see the fused kernels Inductor generated, rerun with TORCH_LOGS=\"output_code\".")
