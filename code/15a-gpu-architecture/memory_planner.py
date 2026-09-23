"""
memory_planner.py  [CPU]
Will a model fit, and how many GPUs does it need? Estimates memory for
inference (weights + KV cache) and training (weights, grads, optimizer, activations),
and the bandwidth-bound decode speed.

Examples:
  python memory_planner.py --params 70 --layers 80 --kv-heads 8 --head-dim 128 --gpu H100-SXM
  python memory_planner.py --params 8 --layers 32 --kv-heads 8 --head-dim 128 --gpu H100-SXM --weight-bytes 1
"""
import argparse
import math

from gpu_specs import GPUS, PRICE_PER_HOUR

GB = 1e9


def kv_bytes_per_token(layers, kv_heads, head_dim, kv_bytes=2):
    return 2 * layers * kv_heads * head_dim * kv_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True, help="parameters in billions")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--gpu", default="H100-SXM", choices=list(GPUS))
    ap.add_argument("--weight-bytes", type=float, default=2, help="2=BF16, 1=FP8/INT8, 0.5=4-bit")
    ap.add_argument("--kv-bytes", type=float, default=2)
    ap.add_argument("--context", type=int, default=8192, help="tokens per sequence")
    ap.add_argument("--batch", type=int, default=32, help="concurrent sequences")
    ap.add_argument("--usable", type=float, default=0.90, help="fraction of GPU memory usable")
    args = ap.parse_args()

    g = GPUS[args.gpu]
    gpu_mem = g["mem_gb"] * GB * args.usable

    # ---------- inference ----------
    weights = args.params * 1e9 * args.weight_bytes
    kv_tok = kv_bytes_per_token(args.layers, args.kv_heads, args.head_dim, args.kv_bytes)
    kv_total = kv_tok * args.context * args.batch
    need = weights + kv_total
    n_gpus = max(1, math.ceil(need / gpu_mem))
    # tensor parallel sizes are usually powers of two
    tp = 1 << (n_gpus - 1).bit_length()

    print(f"== Inference on {args.gpu} ({g['mem_gb']} GB, {g['bw']/1e12:.2f} TB/s) ==")
    print(f"weights                 {weights/GB:8.1f} GB")
    print(f"KV cache per token      {kv_tok/1024:8.1f} KB")
    print(f"KV cache ({args.batch} seq x {args.context} tok) {kv_total/GB:8.1f} GB")
    print(f"total                   {need/GB:8.1f} GB  ->  minimum {n_gpus} GPU(s); use TP={tp}")

    free_for_kv = tp * gpu_mem - weights
    max_seqs = int(free_for_kv // (kv_tok * args.context)) if free_for_kv > 0 else 0
    print(f"max concurrent {args.context}-token sequences on TP={tp}: {max_seqs}")

    # bandwidth-bound decode: every step reads all weights once (shared by the batch)
    # plus every sequence's KV cache (not shared).
    bw_total = g["bw"] * tp
    for b in [1, 8, 32, max(1, max_seqs)]:
        step_bytes = weights + b * kv_tok * args.context / 2   # average context = half full
        step_s = step_bytes / (bw_total * 0.8)                 # ~80% of peak bandwidth is realistic
        tok_s = b / step_s
        line = f"  batch {b:4d}: ~{1/step_s:6.1f} tokens/s per sequence, ~{tok_s:8.0f} tokens/s total"
        price = PRICE_PER_HOUR.get(args.gpu)
        if price:
            line += f", ~${price * tp / (tok_s * 3600) * 1e6:6.2f} per 1M output tokens"
        print(line)

    # ---------- training ----------
    print(f"\n== Full training with mixed-precision AdamW ==")
    per_param = 2 + 2 + 12  # bf16 weights + bf16 grads + fp32 master/m/v
    states = args.params * 1e9 * per_param
    print(f"weights+grads+optimizer {states/GB:8.1f} GB (16 bytes/param), before activations")
    for n in [8, 16, 64, 256]:
        print(f"  sharded over {n:3d} GPUs (ZeRO-3/FSDP): {states/n/GB:7.1f} GB per GPU + activations")


if __name__ == "__main__":
    main()
