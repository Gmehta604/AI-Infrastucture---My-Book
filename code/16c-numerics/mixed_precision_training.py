"""
mixed_precision_training.py  [CPU or GPU]
Trains the same small network four ways and prints the loss after each epoch:
  fp32                  baseline
  bf16 autocast         the modern default: no loss scaling needed
  fp16, no scaling      small gradients underflow to zero -> training stalls or degrades
  fp16 + GradScaler     dynamic loss scaling rescues it

To make FP16 underflow visible on a toy problem, the loss is multiplied by a small
constant (as happens implicitly with long sequences averaged over many tokens).

Run:  python mixed_precision_training.py
"""
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
X = torch.randn(4096, 64, device=device)
true_w = torch.randn(64, 1, device=device)
y = (X @ true_w).tanh() + 0.05 * torch.randn(4096, 1, device=device)
LOSS_SCALE_DOWN = 1e-6      # simulates tiny per-token gradients


def make_model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1)).to(device)


def train(mode, epochs=8):
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scaler = torch.amp.GradScaler(device, enabled=(mode == "fp16+scaler"))
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16+scaler": torch.float16}[mode]
    history, zero_frac = [], 0.0
    for epoch in range(epochs):
        for i in range(0, len(X), 256):
            xb, yb = X[i:i + 256], y[i:i + 256]
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                loss = ((model(xb).float() - yb) ** 2).mean() * LOSS_SCALE_DOWN
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # measure how many gradient entries are exactly zero (underflowed)
            grads = torch.cat([p.grad.flatten() for p in model.parameters()])
            zero_frac = (grads == 0).float().mean().item()
            for p in model.parameters():          # undo the toy scale-down for the update itself
                p.grad /= LOSS_SCALE_DOWN
            scaler.step(opt)
            scaler.update()
        with torch.no_grad():
            history.append(((model(X) - y) ** 2).mean().item())
    return history, zero_frac


if __name__ == "__main__":
    if device == "cpu":
        print("Running on CPU: autocast works for bf16; fp16 autocast on CPU is supported in recent PyTorch.\n")
    print(f"{'mode':12s} " + " ".join(f"ep{e+1:<5d}" for e in range(8)) + "  zero-grad fraction")
    for mode in ["fp32", "bf16", "fp16", "fp16+scaler"]:
        try:
            hist, zf = train(mode)
            print(f"{mode:12s} " + " ".join(f"{h:7.4f}" for h in hist) + f"   {zf:6.1%}")
        except RuntimeError as e:
            print(f"{mode:12s} not supported on this device: {str(e).splitlines()[0][:70]}")
