# Code for Deep Dive 16c: Numerics — BF16, FP8 and FP4

Everything here runs on a laptop CPU (`pip install torch numpy`). `fp8_matmul.py` additionally uses real FP8 tensor cores if you have an Ada, Hopper or Blackwell GPU.

| Script | What it shows |
|---|---|
| `float_formats.py` | Range, precision and special values of FP32 … FP4; bit-level decoding; every FP4 and FP8 value |
| `accumulation.py` | Why sums must accumulate in FP32; Kahan and stochastic rounding; weight updates lost in BF16 |
| `scaling.py` | Per-tensor vs per-row vs 1×128 tiles vs MXFP8 / MXFP4 / NVFP4 block scaling on outlier-heavy activations |
| `fp8_matmul.py` | A simulated FP8 GEMM with per-tensor vs DeepSeek-V3-style fine-grained scales |
| `mixed_precision_training.py` | FP32 vs BF16 vs FP16 with and without loss scaling, showing gradient underflow |
| `weight_quantization.py` | INT8, FP8, INT4 (per-channel, group-wise, asymmetric), NF4 and block FP4 weight quantization |

```bash
cd code/16c-numerics
python float_formats.py
python accumulation.py
python scaling.py
python fp8_matmul.py
python mixed_precision_training.py
python weight_quantization.py
```

All six were run on CPU while writing the chapter; the numbers quoted in the chapter come from those runs.
