// softmax.cu  [1 GPU]
// Row-wise softmax over a (rows x cols) matrix, as used in attention and the LM head.
//   1. naive: one thread per row, three passes over the row  (uncoalesced, little parallelism)
//   2. one warp per row, ONLINE softmax: max and sum computed in a single pass,
//      combined across lanes with warp shuffles, then one pass to write.
//
// Build:  make softmax           Run:  ./softmax 8192 4096
#include <algorithm>
#include <cfloat>
#include "common.cuh"

__global__ void softmax_naive(const float* x, float* y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + (size_t)r * cols;
  float* yr = y + (size_t)r * cols;
  float m = -FLT_MAX;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) s += expf(xr[c] - m);
  for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) / s;
}

__global__ void softmax_warp_online(const float* x, float* y, int rows, int cols) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (warp >= rows) return;
  const float* xr = x + (size_t)warp * cols;
  float* yr = y + (size_t)warp * cols;

  // Pass 1 (online): keep a running max m and a running sum s rescaled whenever m grows.
  float m = -FLT_MAX, s = 0.f;
  for (int c = lane; c < cols; c += 32) {           // lanes read adjacent elements: coalesced
    float v = xr[c];
    float m_new = fmaxf(m, v);
    s = s * expf(m - m_new) + expf(v - m_new);
    m = m_new;
  }
  // Combine the 32 partial (m, s) pairs across the warp.
  for (int off = 16; off > 0; off >>= 1) {
    float m_o = __shfl_xor_sync(0xffffffff, m, off);
    float s_o = __shfl_xor_sync(0xffffffff, s, off);
    float m_new = fmaxf(m, m_o);
    s = s * expf(m - m_new) + s_o * expf(m_o - m_new);
    m = m_new;
  }
  // Pass 2: write normalised outputs.
  const float inv = 1.f / s;
  for (int c = lane; c < cols; c += 32) yr[c] = expf(xr[c] - m) * inv;
}

int main(int argc, char** argv) {
  const int rows = argc > 1 ? atoi(argv[1]) : 8192;
  const int cols = argc > 2 ? atoi(argv[2]) : 4096;
  const size_t n = (size_t)rows * cols;
  std::vector<float> hx(n), hy(n), href(n);
  fill_random(hx, 4);
  for (int r = 0; r < rows; ++r) {                   // CPU reference
    const float* xr = &hx[(size_t)r * cols];
    float m = -FLT_MAX; for (int c = 0; c < cols; ++c) m = std::max(m, xr[c]);
    double s = 0; for (int c = 0; c < cols; ++c) s += std::exp(xr[c] - m);
    for (int c = 0; c < cols; ++c) href[(size_t)r * cols + c] = std::exp(xr[c] - m) / s;
  }
  float *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  auto check = [&](const char* name, auto launch) {
    launch(); CUDA_CHECK_LAST(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    float ms = time_ms(launch, 10);
    // minimum traffic: read x once, write y once
    printf("%-26s %8.3f ms %8.0f GB/s effective, max err %.2e\n", name, ms,
           2.0 * n * sizeof(float) / ms / 1e6, max_abs_diff(hy, href));
  };
  check("1 naive thread-per-row", [&] { softmax_naive<<<ceil_div(rows, 128), 128>>>(dx, dy, rows, cols); });
  check("2 warp-per-row online", [&] { softmax_warp_online<<<ceil_div(rows * 32, 256), 256>>>(dx, dy, rows, cols); });
  cudaFree(dx); cudaFree(dy);
  return 0;
}
