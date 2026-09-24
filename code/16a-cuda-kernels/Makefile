NVCC  ?= nvcc
# Set ARCH for your GPU: sm_80 (A100), sm_86 (RTX 30xx, A10), sm_89 (L4, RTX 40xx), sm_90 (H100/H200), sm_100 (B200)
ARCH  ?= sm_80
FLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo
BINS   = vector_add sgemm hgemm_wmma reduction softmax cuda_graphs

all: $(BINS)

%: %.cu common.cuh
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

clean:
	rm -f $(BINS) *.ncu-rep *.nsys-rep

.PHONY: all clean
