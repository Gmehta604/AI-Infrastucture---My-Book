# Code for Deep Dive 16a: CUDA Programming and Kernel Optimisation

All programs need an NVIDIA GPU and the CUDA toolkit (`nvcc`). Google Colab, Kaggle, RunPod, Lambda and Modal all provide both.

| File | What it does |
|---|---|
| `common.cuh` | Error-checking macro, CUDA-event timer, helpers |
| `vector_add.cu` | First kernel: allocate, copy, launch, check; one-thread-per-element vs grid-stride |
| `sgemm.cu` | Six FP32 matmul kernels (naive → vectorised 2D tiling) checked and timed against cuBLAS |
| `hgemm_wmma.cu` | Kernel 7: FP16 matmul on tensor cores with the WMMA API, vs cuBLAS |
| `reduction.cu` | Array sum: atomics vs shared-memory tree vs warp shuffles |
| `softmax.cu` | Row softmax: naive vs warp-per-row online softmax |
| `cuda_graphs.cu` | Launch overhead: 2,000 kernels one by one vs one CUDA graph |
| `fused_bias_gelu.py` | Custom CUDA op compiled from Python and called from PyTorch |
| `Makefile` | Builds everything; set `ARCH` for your GPU |

## Build and run

```bash
cd code/16a-cuda-kernels
make ARCH=sm_90          # H100/H200. Use sm_80 for A100, sm_89 for L4/RTX 40xx, sm_86 for A10/RTX 30xx, sm_75 for T4
./vector_add
./sgemm                  # all kernels at 4096; ./sgemm 5 2048 runs kernel 5 at 2048
./hgemm_wmma 4096
./reduction
./softmax 8192 4096
./cuda_graphs
python fused_bias_gelu.py
```

On Google Colab, prefix shell commands with `!` in a notebook cell, and pick the T4 (`ARCH=sm_75`) or A100 (`ARCH=sm_80`) runtime.

## Profiling

```bash
nsys profile -o timeline ./sgemm 5 4096                   # timeline of the whole program
ncu --set full -k regex:sgemm_2d -o k5 ./sgemm 5 4096      # deep report for one kernel
ncu --set full -k regex:sgemm_vectorized -o k6 ./sgemm 6 4096
compute-sanitizer --tool memcheck ./sgemm 4 512            # out-of-bounds and misaligned accesses
compute-sanitizer --tool racecheck ./sgemm 3 512           # shared-memory races (e.g. a missing __syncthreads)
```

Open `.ncu-rep` files in the Nsight Compute GUI to compare kernels side by side.

## Checking register use without a GPU

```bash
nvcc -O3 -arch=sm_90 --ptxas-options=-v -c sgemm.cu -o /dev/null
```

prints registers, shared memory and spills for each kernel.
