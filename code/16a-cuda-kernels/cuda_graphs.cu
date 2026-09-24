// cuda_graphs.cu  [1 GPU]
// Launch overhead made visible. 2,000 tiny kernels launched one by one, versus the
// same sequence captured once into a CUDA graph and replayed with a single call.
// LLM decode steps are hundreds of small kernels, which is why serving engines
// (vLLM, SGLang, TensorRT-LLM) capture decode steps as CUDA graphs.
//
// Build:  make cuda_graphs       Run:  ./cuda_graphs
#include "common.cuh"

__global__ void tiny(float* x) { x[threadIdx.x] += 1.f; }

int main() {
  const int kernels = 2000;
  float* d;
  CUDA_CHECK(cudaMalloc(&d, 32 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d, 0, 32 * sizeof(float)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  auto eager = [&] {
    for (int i = 0; i < kernels; ++i) tiny<<<1, 32, 0, s>>>(d);
  };

  // Capture the same sequence into a graph.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  eager();
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  auto replay = [&] { CUDA_CHECK(cudaGraphLaunch(exec, s)); };

  auto time_on_stream = [&](auto fn) {
    for (int i = 0; i < 3; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < 10; ++i) fn();
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / 10;
  };
  float t_eager = time_on_stream(eager);
  float t_graph = time_on_stream(replay);
  printf("%d tiny kernels, launched one by one: %7.3f ms  (%.2f us per kernel)\n",
         kernels, t_eager, 1000.f * t_eager / kernels);
  printf("%d tiny kernels, one CUDA graph:      %7.3f ms  (%.2f us per kernel)\n",
         kernels, t_graph, 1000.f * t_graph / kernels);
  printf("speed-up: %.1fx\n", t_eager / t_graph);

  cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(s); cudaFree(d);
  return 0;
}
