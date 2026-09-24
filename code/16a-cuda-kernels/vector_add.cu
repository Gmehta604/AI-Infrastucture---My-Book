// vector_add.cu  [1 GPU]
// The "hello world" of CUDA: allocate, copy, launch, copy back, check errors.
// Also measures effective bandwidth, since vector add is purely memory-bound.
//
// Build:  make vector_add        Run:  ./vector_add
#include <algorithm>
#include "common.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's global index
  if (i < n) c[i] = a[i] + b[i];                   // guard: the last block may overhang
}

// Grid-stride loop: a fixed number of threads walks the whole array.
// Works for any n and lets you size the grid to the GPU rather than to the data.
__global__ void vector_add_gridstride(const float* a, const float* b, float* c, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 26;                           // 64M floats = 256 MB per array
  const size_t bytes = n * sizeof(float);
  std::vector<float> ha(n), hb(n), hc(n);
  fill_random(ha, 1); fill_random(hb, 2);

  float *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dc, bytes));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = ceil_div(n, threads);
  vector_add<<<blocks, threads>>>(da, db, dc, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    if (fabsf(hc[i] - (ha[i] + hb[i])) > 1e-6f) { printf("mismatch at %d\n", i); return 1; }
  printf("vector_add correct\n");

  int sms; CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  float ms1 = time_ms([&] { vector_add<<<blocks, threads>>>(da, db, dc, n); });
  float ms2 = time_ms([&] { vector_add_gridstride<<<sms * 8, threads>>>(da, db, dc, n); });
  // bytes moved: read a, read b, write c
  printf("one thread per element: %.3f ms, %.0f GB/s\n", ms1, 3.0 * bytes / ms1 / 1e6);
  printf("grid-stride (%d blocks): %.3f ms, %.0f GB/s\n", sms * 8, ms2, 3.0 * bytes / ms2 / 1e6);

  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
  return 0;
}
