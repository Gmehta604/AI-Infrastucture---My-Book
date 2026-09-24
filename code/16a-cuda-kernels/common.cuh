// common.cuh: small helpers shared by every program in this folder.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// Every CUDA API call returns an error code. Check it, always.
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
              cudaGetErrorString(err__));                                       \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Kernel launches don't return errors directly: check launch errors, then
// (in debug builds) synchronise to surface errors that happen while running.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline void fill_random(std::vector<float>& v, unsigned seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : v) x = dist(gen);
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

// Times `fn` with CUDA events: warm-up, then the median of `reps` runs, in milliseconds.
template <typename F>
float time_ms(F fn, int reps = 20, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> t(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(t.begin(), t.end());
  return t[reps / 2];
}
