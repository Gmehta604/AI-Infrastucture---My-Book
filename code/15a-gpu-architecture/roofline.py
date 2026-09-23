"""
roofline.py  [CPU]
Compute arithmetic intensity for the operations inside a Transformer layer and
place them on the roofline of a chosen GPU. Saves roofline.png.

Run:  python roofline.py --gpu H100-SXM --d 4096 --tokens 1 16 256 4096
"""
import argparse
import math

from gpu_specs import GPUS


def matmul(M, N, K, bytes_per=2):
    """C[M,N] = A[M,K] @ B[K,N]. Returns (flops, bytes) assuming each operand is read once."""
    flops = 2 * M * N * K
    bytes_moved = bytes_per * (M * K + K * N + M * N)
    return flops, bytes_moved


def elementwise(n, reads=2, writes=1, flops_per=1, bytes_per=2):
    """e.g. residual add: read x and y, write z."""
    return flops_per * n, bytes_per * n * (reads + writes)


def rmsnorm(tokens, d, bytes_per=2):
    # square, mean, rsqrt, multiply, scale ~ 5 FLOPs per element; read x and weight, write y
    return 5 * tokens * d, bytes_per * (2 * tokens * d + d)


def softmax(rows, cols, bytes_per=2):
    # max, subtract, exp, sum, divide ~ 5 FLOPs per element; read once, write once
    return 5 * rows * cols, bytes_per * 2 * rows * cols


def layer_ops(tokens, d, d_ff):
    """Main operations of one Transformer block for `tokens` tokens processed at once."""
    return {
        "QKV projection": matmul(tokens, 3 * d, d),
        "MLP up+gate": matmul(tokens, 2 * d_ff, d),
        "MLP down": matmul(tokens, d, d_ff),
        "RMSNorm": rmsnorm(tokens, d),
        "Residual add": elementwise(tokens * d),
        "Softmax (4k ctx, 32 heads)": softmax(tokens * 32, 4096),
    }


def attainable(intensity, peak, bw):
    return min(peak, bw * intensity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100-SXM", choices=list(GPUS))
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--dff", type=int, default=14336)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 256, 4096])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    g = GPUS[args.gpu]
    peak, bw = g["bf16"], g["bw"]
    ridge = peak / bw
    print(f"{args.gpu}: peak {peak/1e12:.0f} TFLOP/s BF16, {bw/1e12:.2f} TB/s, ridge {ridge:.0f} FLOPs/byte\n")

    points = []
    for T in args.tokens:
        print(f"--- {T} token(s) at once ---")
        for name, (f, b) in layer_ops(T, args.d, args.dff).items():
            ai = f / b
            perf = attainable(ai, peak, bw)
            bound = "compute" if ai >= ridge else "memory"
            print(f"  {name:28s} intensity {ai:8.1f}  ->  {perf/1e12:7.1f} TFLOP/s max  ({bound}-bound)")
            points.append((f"{name} (T={T})", ai, perf))
        print()

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    xs = [10 ** (i / 20) for i in range(-40, 101)]  # 0.01 .. 100000 FLOPs/byte
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(xs, [attainable(x, peak, bw) / 1e12 for x in xs], color="#1C6B50", lw=2.5, label=f"{args.gpu} roofline")
    ax.axvline(ridge, color="#A5591F", ls="--", lw=1)
    ax.text(ridge * 1.1, peak / 1e12 / 30, f"ridge ≈ {ridge:.0f}", color="#A5591F")
    for label, ai, perf in points:
        ax.scatter([ai], [perf / 1e12], s=22, color="#141B18", zorder=3)
        if "MLP down" in label or "RMSNorm" in label:
            ax.annotate(label, (ai, perf / 1e12), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("Arithmetic intensity (FLOPs per byte)")
    ax.set_ylabel("Attainable TFLOP/s")
    ax.set_title("Transformer operations on the roofline")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("roofline.png", dpi=150)
    print("saved roofline.png")


if __name__ == "__main__":
    main()
