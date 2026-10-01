# Code for Deep Dive 16b: Triton, Compilers and FlashAttention

Every Triton script runs on an NVIDIA GPU (with benchmarks), or on any laptop in Triton's
CPU interpreter mode (correctness only):

```bash
pip install torch triton numpy        # triton ships with PyTorch's Linux CUDA wheels
python 02_fused_softmax.py            # on a GPU machine
TRITON_INTERPRET=1 python 02_fused_softmax.py   # on a laptop, no GPU needed
```

The interpreter lets you put `print()` calls and breakpoints inside kernels, which is the
easiest way to learn how a kernel's blocks and masks behave.

| Script | Hardware | What it does |
|---|---|---|
| `common.py` | – | Device detection, correctness check, benchmarking helper |
| `01_vector_add.py` | GPU / interpreter | First kernel: program IDs, `tl.arange`, masks |
| `02_fused_softmax.py` | GPU / interpreter | One-pass row softmax vs `torch.softmax` vs hand-written PyTorch |
| `03_rmsnorm.py` | GPU / interpreter | Fused RMSNorm with optional fused residual add |
| `04_matmul.py` | GPU / interpreter | Autotuned tensor-core matmul with grouped ordering and a fused GELU epilogue |
| `05_flash_attention_numpy.py` | CPU | FlashAttention's algorithm in NumPy, checked against standard attention; memory table |
| `06_flash_attention_triton.py` | GPU / interpreter | FlashAttention forward kernel (causal and non-causal) vs PyTorch SDPA |
| `07_torch_compile.py` | CPU or GPU | Graph breaks, eager vs compiled timing, how to dump generated kernels |

All eight were run successfully in interpreter mode on CPU. GPU speed numbers depend on your
hardware; compare them with the expectations in the chapter.
