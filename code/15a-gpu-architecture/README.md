# Code for Deep Dive 15a: GPU Architecture and the Performance Model

| Script | Hardware | What it does |
|---|---|---|
| `gpu_specs.py` | CPU | Datasheet figures and ridge points for A100, H100, H200, B200, B300, MI300X, MI355X, Rubin. Edit prices to match yours. |
| `occupancy.py` | CPU | Occupancy calculator: which resource limits resident warps. |
| `wave_quantization.py` | CPU | Tile and wave quantization for GEMMs of different sizes. |
| `roofline.py` | CPU | Places Transformer operations on a GPU's roofline; saves `roofline.png`. |
| `memory_planner.py` | CPU | GPUs needed, max batch, decode speed and $/M tokens for a model. |
| `microbench.py` | 1 GPU (CPU works, slowly) | Measures copy bandwidth and matmul TFLOP/s; saves `microbench.csv`. |

```bash
cd code/15a-gpu-architecture
python gpu_specs.py
python occupancy.py --threads 256 --regs 64 --smem-kb 48
python wave_quantization.py
python roofline.py --gpu H100-SXM --tokens 1 256 4096
python memory_planner.py --params 70 --layers 80 --kv-heads 8 --gpu H100-SXM
python microbench.py            # on a GPU machine
```
