"""Export trained PyTorch NNUE checkpoint → weights/nnue.npz for the agent.

W1 is quantised to int16 (scale factor 64) to reduce file size (~21 MB vs ~42 MB).
All other weights are stored as float32 (they are tiny: ~72 KB total).

Usage:
    python -m train.export_weights --checkpoint train/nnue.pt --out weights/nnue.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train.train_nnue import NNUE

_QUANT_SCALE = 64


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="train/nnue.pt")
    p.add_argument("--out", default="weights/nnue.npz")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    model = NNUE()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    w1_f32: np.ndarray = model.ft.weight.detach().numpy()  # (40960, N1)
    w1_int16 = np.clip(np.round(w1_f32 * _QUANT_SCALE), -32768, 32767).astype(np.int16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        w1=w1_int16,
        b1=model.b1.detach().numpy().astype(np.float32),
        w2=model.l2.weight.detach().numpy().T.astype(np.float32),  # (2*N1, N2)
        b2=model.l2.bias.detach().numpy().astype(np.float32),
        w3=model.l3.weight.detach().numpy().T.astype(np.float32),  # (N2, N3)
        b3=model.l3.bias.detach().numpy().astype(np.float32),
        w4=model.l4.weight.detach().numpy().T.astype(np.float32),  # (N3, 1)
        b4=model.l4.bias.detach().numpy().astype(np.float32),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"exported → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
