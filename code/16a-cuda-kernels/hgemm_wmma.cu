// hgemm_wmma.cu  [1 GPU, Volta or newer]
// Kernel 7: the same matmul on TENSOR CORES using the WMMA API.
// Inputs in FP16, accumulation in FP32. Each warp computes one 16x16 output tile.
// This simple version reads fragments straight from global memory; production kernels
// (CUTLASS, cuBLAS) stage tiles through shared memory with async copies (cp.async / TMA)
// and use larger warp tiles, reaching several times this speed.
//
// Build:  make hgemm_wmma        Run:  ./hgemm_wmma 4096
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common.cuh"

using namespace nvcuda;

constexpr int WM = 16, WN = 16, WK = 16;   // WMMA tile shape for FP16

__global__ void hgemm_wmma(int M, int N, int K, const half* A, const half* B, float* C) {
  // blockDim = (128, 4): 4 warps along M (x) times 4 warps along N (y) = a 64x64 block tile
  const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int warpN = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = warpM * WM, col = warpN * WN;
  if (row >= M || col >= N) return;

  wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
  wmma::fill_fragment(acc, 0.f);

  for (int k = 0; k < K; k += WK) {
    wmma::load_matrix_sync(a, A + row * K + k, K);    // 16x16 slice of A, leading dim K
    wmma::load_matrix_sync(b, B + k * N + col, N);    // 16x16 slice of B, leading dim N
    wmma::mma_sync(acc, a, b, acc);                   // one warp-wide tensor-core MMA
  }
  wmma::store_matrix_sync(C + row * N + col, acc, N, wmma::mem_row_major);
}

__global__ void to_half(const float* in, half* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

int main(int argc, char** argv) {
  const int S = argc > 1 ? atoi(argv[1]) : 4096;
  if (S % 64 != 0) { fprintf(stderr, "size must be a multiple of 64\n"); return 1; }
  const int M = S, N = S, K = S;

  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
  fill_random(hA, 1); fill_random(hB, 2);
  float *fA, *fB, *dC, *dRef;
  half *hAd, *hBd;
  CUDA_CHECK(cudaMalloc(&fA, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&fB, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc(&hAd, sizeof(half) * M * K));
  CUDA_CHECK(cudaMalloc(&hBd, sizeof(half) * K * N));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc(&dRef, sizeof(float) * M * N));
  CUDA_CHECK(cudaMemcpy(fA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(fB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));
  to_half<<<ceil_div(M * K, 256), 256>>>(fA, hAd, M * K);
  to_half<<<ceil_div(K * N, 256), 256>>>(fB, hBd, K * N);
  CUDA_CHECK_LAST();

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.f, beta = 0.f;
  auto cublas = [&] {  // row-major trick: C^T = B^T A^T in column-major terms
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, hBd, CUDA_R_16F, N,
                 hAd, CUDA_R_16F, K, &beta, dRef, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
  };
  dim3 block(128, 4), grid(ceil_div(M, 64), ceil_div(N, 64));
  auto mine = [&] { hgemm_wmma<<<grid, block>>>(M, N, K, hAd, hBd, dC); };

  cublas(); mine();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  const double flops = 2.0 * M * N * K;
  float t_cublas = time_ms(cublas), t_mine = time_ms(mine);
  printf("size %d, FP16 inputs, FP32 accumulate\n", S);
  printf("cuBLAS (tensor cores): %8.3f ms  %8.1f TFLOP/s\n", t_cublas, flops / t_cublas / 1e9);
  printf("simple WMMA kernel:    %8.3f ms  %8.1f TFLOP/s  (%.0f%% of cuBLAS), max err %.2e\n",
         t_mine, flops / t_mine / 1e9, 100.0 * t_cublas / t_mine, max_abs_diff(hC, hRef));
  cublasDestroy(handle);
  return 0;
}
