"""
fused_bias_gelu.py  [1 GPU, needs nvcc (CUDA toolkit) for the JIT build]
Write a custom CUDA kernel, compile it from Python with torch.utils.cpp_extension.load_inline,
and call it like any PyTorch op. The kernel fuses a bias add and a GELU into one pass over
memory; PyTorch eager runs them as two kernels (two round trips to HBM).

Run:  python fused_bias_gelu.py
The first run compiles the extension (about a minute); later runs use the cache.
"""
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void bias_gelu_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                 float* __restrict__ y, long n, int cols) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i] + bias[i % cols];
    const float k0 = 0.7978845608f;            // sqrt(2 / pi)
    y[i] = 0.5f * v * (1.f + tanhf(k0 * (v + 0.044715f * v * v * v)));
  }
}

torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32, "float32 only");
  TORCH_CHECK(x.is_contiguous() && bias.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(bias.numel() == x.size(-1), "bias must match the last dimension of x");
  auto y = torch::empty_like(x);
  long n = x.numel();
  int cols = x.size(-1);
  int threads = 256;
  long blocks = (n + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();   // respect PyTorch's current stream
  bias_gelu_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), bias.data_ptr<float>(),
                                                   y.data_ptr<float>(), n, cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""
cpp_src = "torch::Tensor bias_gelu(torch::Tensor x, torch::Tensor bias);"

ext = load_inline(name="bias_gelu_ext", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["bias_gelu"], extra_cuda_cflags=["-O3"], verbose=False)


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    x = torch.randn(8192, 14336, device="cuda")      # an MLP activation for 8K tokens
    b = torch.randn(14336, device="cuda")

    eager = lambda: F.gelu(x + b, approximate="tanh")
    compiled_fn = torch.compile(lambda x, b: F.gelu(x + b, approximate="tanh"))
    compiled = lambda: compiled_fn(x, b)
    custom = lambda: ext.bias_gelu(x, b)

    ref = eager()
    print("max error vs PyTorch:", (custom() - ref).abs().max().item())
    bytes_min = 2 * x.numel() * 4                     # read x once, write y once
    for name, fn in [("eager (2 kernels)", eager), ("torch.compile", compiled), ("custom CUDA", custom)]:
        ms = bench(fn)
        print(f"{name:20s} {ms:7.3f} ms   {bytes_min / ms / 1e6:7.0f} GB/s effective")


if __name__ == "__main__":
    main()
