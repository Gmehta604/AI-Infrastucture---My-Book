// sgemm.cu  [1 GPU]
// Six FP32 matrix-multiply kernels, each fixing the main bottleneck of the one before,
// checked for correctness and timed against cuBLAS.
//
// C = alpha * A @ B + beta * C, all matrices row-major.  A: M x K, B: K x N, C: M x N.
// For simplicity M, N and K must be multiples of 128.
//
// Build:  make sgemm             Run:  ./sgemm            (all kernels, 4096)
//                                      ./sgemm 5 2048     (kernel 5 only, size 2048)
#include <algorithm>
#include <cstring>
#include <cublas_v2.h>
#include "common.cuh"

// ---------------------------------------------------------------------------------
// Kernel 1: naive. One thread per output element.
// threadIdx.x walks down ROWS, so neighbouring threads touch addresses K or N floats
// apart: loads of B and stores of C are not coalesced.
// ---------------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ---------------------------------------------------------------------------------
// Kernel 2: global memory coalescing. Same work, but consecutive threads now compute
// consecutive COLUMNS, so a warp reads 32 adjacent floats of B and writes 32 adjacent
// floats of C: one wide transaction instead of 32.
// ---------------------------------------------------------------------------------
template <int BS>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float* A,
                                const float* B, float beta, float* C) {
  const int row = blockIdx.x * BS + threadIdx.x / BS;
  const int col = blockIdx.y * BS + threadIdx.x % BS;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ---------------------------------------------------------------------------------
// Kernel 3: shared-memory tiling. The block cooperatively loads a BSxBS tile of A and
// of B into shared memory, then every thread reuses those tiles BS times.
// Global traffic drops by a factor of ~BS.
// ---------------------------------------------------------------------------------
template <int BS>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
  __shared__ float As[BS * BS];
  __shared__ float Bs[BS * BS];
  const int cRow = blockIdx.x, cCol = blockIdx.y;
  const int tRow = threadIdx.x / BS, tCol = threadIdx.x % BS;
  A += cRow * BS * K;                       // move pointers to this block's tiles
  B += cCol * BS;
  C += cRow * BS * N + cCol * BS;

  float acc = 0.f;
  for (int bk = 0; bk < K; bk += BS) {
    As[tRow * BS + tCol] = A[tRow * K + tCol];      // coalesced loads (tCol is fastest)
    Bs[tRow * BS + tCol] = B[tRow * N + tCol];
    __syncthreads();                                // tile fully loaded
    A += BS;
    B += BS * N;
    for (int d = 0; d < BS; ++d) acc += As[tRow * BS + d] * Bs[d * BS + tCol];
    __syncthreads();                                // done reading before overwrite
  }
  C[tRow * N + tCol] = alpha * acc + beta * C[tRow * N + tCol];
}

// ---------------------------------------------------------------------------------
// Kernel 4: 1D block tiling. Each thread computes TM outputs in a column, keeping them
// in registers. Every value loaded from shared memory (b) is reused TM times.
// ---------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_1d_blocktile(int M, int N, int K, float alpha, const float* A,
                                   const float* B, float beta, float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int tCol = threadIdx.x % BN;        // which output column
  const int tRow = threadIdx.x / BN;        // which group of TM output rows
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;

  float acc[TM] = {0.f};
  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
    #pragma unroll
    for (int d = 0; d < BK; ++d) {
      const float b = Bs[d * BN + tCol];                 // one shared-memory load...
      #pragma unroll
      for (int r = 0; r < TM; ++r)
        acc[r] += As[(tRow * TM + r) * BK + d] * b;      // ...used TM times
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < TM; ++r) {
    const int idx = (tRow * TM + r) * N + tCol;
    C[idx] = alpha * acc[r] + beta * C[idx];
  }
}

// ---------------------------------------------------------------------------------
// Kernel 5: 2D block tiling. Each thread computes a TMxTN tile of outputs as an outer
// product of TM values of A and TN values of B held in registers.
// Per inner step: TM + TN shared loads feed TM*TN FMAs (8+8 loads for 64 FMAs).
// ---------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_2d_blocktile(int M, int N, int K, float alpha, const float* A, const float* B,
                   float beta, float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  constexpr int numThreads = (BM * BN) / (TM * TN);
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int tCol = threadIdx.x % (BN / TN);
  const int tRow = threadIdx.x / (BN / TN);
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  constexpr int strideA = numThreads / BK;          // rows of A loaded per pass
  const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
  constexpr int strideB = numThreads / BN;          // rows of B loaded per pass

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];
  for (int bk = 0; bk < K; bk += BK) {
    #pragma unroll
    for (int o = 0; o < BM; o += strideA)
      As[(innerRowA + o) * BK + innerColA] = A[(innerRowA + o) * K + innerColA];
    #pragma unroll
    for (int o = 0; o < BK; o += strideB)
      Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
    #pragma unroll
    for (int d = 0; d < BK; ++d) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[(tRow * TM + i) * BK + d];
      #pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = Bs[d * BN + tCol * TN + i];
      #pragma unroll
      for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) acc[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int m = 0; m < TM; ++m)
    #pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int idx = (tRow * TM + m) * N + tCol * TN + n;
      C[idx] = alpha * acc[m * TN + n] + beta * C[idx];
    }
}

// ---------------------------------------------------------------------------------
// Kernel 6: vectorised memory access. Same tiling as kernel 5, but:
//  - global loads and stores use float4 (128-bit) instructions;
//  - the A tile is stored TRANSPOSED in shared memory so the inner loop reads
//    consecutive addresses for both A and B.
// Requires N and K to be multiples of 4 (true here) and 16-byte aligned pointers.
// ---------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_vectorized(int M, int N, int K, float alpha, const float* A, const float* B,
                 float beta, float* C) {
  __shared__ float As[BK * BM];                      // transposed: As[k][m]
  __shared__ float Bs[BK * BN];
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int tCol = threadIdx.x % (BN / TN);
  const int tRow = threadIdx.x / (BN / TN);
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];
  for (int bk = 0; bk < K; bk += BK) {
    float4 a4 = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
    reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;
    #pragma unroll
    for (int d = 0; d < BK; ++d) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[d * BM + tRow * TM + i];
      #pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = Bs[d * BN + tCol * TN + i];
      #pragma unroll
      for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) acc[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int m = 0; m < TM; ++m)
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4* cptr = reinterpret_cast<float4*>(&C[(tRow * TM + m) * N + tCol * TN + n]);
      float4 c4 = cptr[0];
      c4.x = alpha * acc[m * TN + n + 0] + beta * c4.x;
      c4.y = alpha * acc[m * TN + n + 1] + beta * c4.y;
      c4.z = alpha * acc[m * TN + n + 2] + beta * c4.z;
      c4.w = alpha * acc[m * TN + n + 3] + beta * c4.w;
      cptr[0] = c4;
    }
}

// ---------------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------------
void run_kernel(int id, int M, int N, int K, float alpha, const float* A, const float* B,
                float beta, float* C, cublasHandle_t handle) {
  switch (id) {
    case 0:  // cuBLAS is column-major; for row-major C = A B compute C^T = B^T A^T
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
      break;
    case 1: {
      dim3 block(32, 32), grid(ceil_div(M, 32), ceil_div(N, 32));
      sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 2: {
      dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
      sgemm_coalesced<32><<<grid, 32 * 32>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 3: {
      dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
      sgemm_smem<32><<<grid, 32 * 32>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 4: {
      constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
      dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
      sgemm_1d_blocktile<BM, BN, BK, TM><<<grid, (BM * BN) / TM>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 5: {
      constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
      dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
      sgemm_2d_blocktile<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 6: {
      constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
      dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
      sgemm_vectorized<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    default:
      fprintf(stderr, "unknown kernel %d\n", id);
      exit(1);
  }
  CUDA_CHECK_LAST();
}

const char* NAMES[] = {"cuBLAS", "1 naive", "2 coalesced", "3 shared-mem tiling",
                       "4 1D block tiling", "5 2D block tiling", "6 vectorized"};

int main(int argc, char** argv) {
  int only = argc > 1 && strcmp(argv[1], "all") != 0 ? atoi(argv[1]) : -1;
  int S = argc > 2 ? atoi(argv[2]) : 4096;
  if (S % 128 != 0) { fprintf(stderr, "size must be a multiple of 128\n"); return 1; }
  const int M = S, N = S, K = S;
  const float alpha = 1.f, beta = 0.f;

  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
  fill_random(hA, 1); fill_random(hB, 2);
  float *dA, *dB, *dC, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc(&dRef, sizeof(float) * M * N));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);   // true FP32, no TF32 shortcut

  // Reference result from cuBLAS
  run_kernel(0, M, N, K, alpha, dA, dB, beta, dRef, handle);
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  const double flops = 2.0 * M * N * K;
  float cublas_ms = time_ms([&] { run_kernel(0, M, N, K, alpha, dA, dB, beta, dC, handle); });
  printf("size %d x %d x %d (FP32)\n", M, N, K);
  printf("%-22s %9s %10s %10s %10s\n", "kernel", "ms", "GFLOP/s", "% cuBLAS", "max err");
  printf("%-22s %9.3f %10.0f %10.1f %10s\n", NAMES[0], cublas_ms, flops / cublas_ms / 1e6, 100.0, "-");

  for (int id = 1; id <= 6; ++id) {
    if (only > 0 && id != only) continue;
    if (id == 1 && S > 4096) { printf("%-22s skipped (too slow at this size)\n", NAMES[1]); continue; }
    CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * M * N));
    run_kernel(id, M, N, K, alpha, dA, dB, beta, dC, handle);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(hC, hRef);
    int reps = id == 1 ? 3 : 20;
    float ms = time_ms([&] { run_kernel(id, M, N, K, alpha, dA, dB, beta, dC, handle); }, reps, 1);
    printf("%-22s %9.3f %10.0f %10.1f %10.2e%s\n", NAMES[id], ms, flops / ms / 1e6,
           100.0 * cublas_ms / ms, err, err > 1e-2f ? "  <-- WRONG" : "");
  }
  cublasDestroy(handle);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  return 0;
}
