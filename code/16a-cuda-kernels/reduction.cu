// reduction.cu  [1 GPU]
// Summing an array: the canonical memory-bound kernel. Three versions:
//   1. every thread does atomicAdd on one global counter  (serialises: very slow)
//   2. shared-memory tree reduction per block, one atomic per block
//   3. grid-stride accumulation + warp shuffles, one atomic per warp  (near peak bandwidth)
//
// Build:  make reduction         Run:  ./reduction
#include <algorithm>
#include "common.cuh"

__global__ void reduce_atomic(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, x[i]);
}

template <int THREADS>
__global__ void reduce_shared(const float* x, float* out, int n) {
  __shared__ float s[THREADS];
  int i = blockIdx.x * THREADS + threadIdx.x;
  s[threadIdx.x] = i < n ? x[i] : 0.f;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {   // halve active threads each step
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

__inline__ __device__ float warp_sum(float v) {
  // Each step adds the value held by the lane `offset` positions higher.
  // Registers are exchanged directly between the 32 lanes: no shared memory.
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__global__ void reduce_warp(const float* x, float* out, int n) {
  float v = 0.f;
  // float4 loads + grid-stride loop: each thread sums many elements first
  const float4* x4 = reinterpret_cast<const float4*>(x);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += blockDim.x * gridDim.x) {
    float4 f = x4[i];
    v += (f.x + f.y) + (f.z + f.w);
  }
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, v);     // lane 0 of each warp
}

int main() {
  const int n = 1 << 26;                               // 64M floats, 256 MB
  std::vector<float> h(n);
  fill_random(h, 3);
  double ref = 0; for (float f : h) ref += f;

  float *d, *out;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  int sms; CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

  auto run = [&](const char* name, auto launch) {
    CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
    launch();
    CUDA_CHECK_LAST();
    float got; CUDA_CHECK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));
    float ms = time_ms([&] { cudaMemsetAsync(out, 0, sizeof(float)); launch(); }, 10);
    printf("%-28s %8.3f ms %8.0f GB/s   sum %.2f (ref %.2f)\n", name, ms,
           n * sizeof(float) / ms / 1e6, got, ref);
  };
  run("1 atomic per element", [&] { reduce_atomic<<<ceil_div(n, 256), 256>>>(d, out, n); });
  run("2 shared-memory tree", [&] { reduce_shared<256><<<ceil_div(n, 256), 256>>>(d, out, n); });
  run("3 warp shuffle + float4", [&] { reduce_warp<<<sms * 4, 512>>>(d, out, n); });
  printf("(small differences in the sum are expected: float addition order differs)\n");
  cudaFree(d); cudaFree(out);
  return 0;
}
