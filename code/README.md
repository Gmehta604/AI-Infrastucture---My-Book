# Code for the deep dives

Each folder holds the runnable scripts for one deep dive, named after its parent chapter (for example `15a-gpu-architecture/`). Every script starts with a comment saying what it does, what hardware it needs, and how to run it.

## Setup

```bash
cd gradient-to-gigawatt
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Hardware tiers

Scripts are labelled with the hardware they need:

| Label | Needs | Where to run it |
|---|---|---|
| `CPU` | Any laptop | Locally |
| `1 GPU` | One NVIDIA GPU (a T4 or better) | Google Colab, Kaggle, RunPod, Lambda, Modal |
| `MULTI-GPU` | 2–8 GPUs in one machine | A rented multi-GPU node |
| `MULTI-NODE` | Several machines | A small rented cluster; optional |

Most chapters include CPU-friendly versions of their key ideas, so you can learn the concepts before paying for GPUs.

## Cost hygiene

- Shut down rented GPUs as soon as you finish.
- Set a budget alert with your provider.
- Develop and debug on a small GPU or CPU; move to large GPUs only to take measurements.
