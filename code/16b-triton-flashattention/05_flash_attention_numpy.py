"""
05_flash_attention_numpy.py  [CPU]
FlashAttention's algorithm in plain NumPy, next to standard attention, so you can
step through it with a debugger. Checks that both give the same output, then prints
how much memory and HBM traffic each needs as the sequence grows.

Run:  python 05_flash_attention_numpy.py
"""
import numpy as np


def standard_attention(Q, K, V, causal=False):
    """Materialises the full T x T score and probability matrices."""
    T, d = Q.shape
    S = Q @ K.T / np.sqrt(d)                               # (T, T)  <- the memory problem
    if causal:
        S = np.where(np.tril(np.ones((T, T), dtype=bool)), S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)                      # (T, T)
    return P @ V


def flash_attention(Q, K, V, causal=False, Br=64, Bc=64):
    """Tiled attention with online softmax. Never stores more than a Br x Bc tile of scores."""
    T, d = Q.shape
    O = np.zeros_like(Q)
    scale = 1.0 / np.sqrt(d)
    for i in range(0, T, Br):                              # outer loop: a block of query rows
        q = Q[i:i + Br]                                    # "load into SRAM"
        rows = q.shape[0]
        m = np.full(rows, -np.inf)                         # running row max
        l = np.zeros(rows)                                 # running softmax denominator
        acc = np.zeros((rows, d))                          # running (unnormalised) output
        last = min(i + Br, T) if causal else T             # causal: skip blocks above the diagonal
        for j in range(0, last, Bc):                       # inner loop: blocks of keys/values
            k, v = K[j:j + Bc], V[j:j + Bc]
            s = q @ k.T * scale                            # (rows, Bc) tile of scores
            if causal:
                qi = np.arange(i, i + rows)[:, None]
                kj = np.arange(j, j + k.shape[0])[None, :]
                s = np.where(qi >= kj, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])                 # tile probabilities (unnormalised)
            alpha = np.exp(m - m_new)                      # rescale factor for old state
            l = l * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + p @ v
            m = m_new
        O[i:i + Br] = acc / l[:, None]                     # normalise once at the end
    return O


def traffic_table(d=128, heads=32, bytes_per=2, Br=128):
    print(f"\nPer layer, {heads} heads, head dim {d}, BF16:")
    print(f"{'seq len':>8s} {'scores+probs stored':>20s} {'standard HBM traffic':>21s} {'flash (worst case)':>19s}")
    for T in [1024, 4096, 16384, 65536, 131072]:
        qkvo = 4 * T * d * bytes_per * heads
        s_mem = 2 * T * T * bytes_per * heads               # S and P matrices
        std = qkvo + 2 * s_mem                             # write S, read S, write P, read P (roughly)
        flash = qkvo + 2 * T * d * bytes_per * heads * (T // Br)  # K and V re-read once per query block
        print(f"{T:8d} {s_mem/1e9:17.2f} GB {std/1e9:18.2f} GB {flash/1e9:16.2f} GB")
    print("Worst case assumes K and V are re-read from HBM once per query block (L2 often catches some).")
    print("The key difference: flash never stores the T x T matrices, so its memory grows linearly with T.")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d = 300, 64                                         # 300 is deliberately not a multiple of 64
    Q, K, V = (rng.standard_normal((T, d)) for _ in range(3))
    for causal in [False, True]:
        ref = standard_attention(Q, K, V, causal)
        out = flash_attention(Q, K, V, causal)
        print(f"causal={causal}: max abs difference {np.abs(ref - out).max():.2e}")
    traffic_table()
