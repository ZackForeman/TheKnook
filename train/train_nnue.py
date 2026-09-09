"""Train the NNUE network on Stockfish-labelled positions.

Usage:
    python -m train.train_nnue --data train/data.npz --out train/nnue.pt \
        --epochs 20 --batch 4096 --lr 1e-3

Run locally with a GPU if available; CPU is also fine for smaller datasets.
Requires: torch, numpy, python-chess (all available in the competition venv).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import cast

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# HalfKP feature extraction (mirrors src/nnue.py logic, pure Python for training)
# ---------------------------------------------------------------------------
_N1 = 16
_FEAT = 40_960  # 64 king_sq x 10 piece/colour x 64 sq
# forward()'s three clamp(0, 1) activations bound the network's raw output to a small
# range near its init scale; reaching raw centipawn magnitudes (targets run to ~+-2000)
# would need an impractical number of steps for the last layer's weights to grow that
# large. Train against cp / _SCORE_SCALE instead, and multiply back up at inference
# (src/nnue.py's nnue_evaluate) — this constant must match there exactly.
_SCORE_SCALE = 400.0


def _halfkp_indices(board: chess.Board, perspective: int) -> list[int]:
    """Return active HalfKP feature indices for one perspective."""
    ks_sq: chess.Square | None = board.king(chess.WHITE if perspective == 0 else chess.BLACK)
    assert ks_sq is not None
    ks = int(ks_sq)
    indices: list[int] = []
    for actual_color in range(2):
        color_co = chess.WHITE if actual_color == 0 else chess.BLACK
        for pt in range(chess.PAWN, chess.QUEEN + 1):
            pt_idx = pt - 1
            for sq in board.pieces(pt, color_co):
                if perspective == 0:
                    pc = actual_color
                    sq_use = int(sq)
                    ks_use = ks
                else:
                    pc = 1 - actual_color
                    sq_use = int(sq) ^ 56
                    ks_use = ks ^ 56
                indices.append(ks_use * 640 + pc * 320 + pt_idx * 64 + sq_use)
    return indices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HalfKPDataset(Dataset[tuple[list[int], list[int], float]]):
    def __init__(self, fens: list[str], scores: list[float]) -> None:
        self.fens = fens
        self.scores = scores

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], float]:
        board = chess.Board(self.fens[idx])
        stm = 0 if board.turn == chess.WHITE else 1
        idx0 = _halfkp_indices(board, 0)
        idx1 = _halfkp_indices(board, 1)
        # Score is from white's perspective; convert to stm perspective, then to the
        # network's trained output scale.
        cp = self.scores[idx] if stm == 0 else -self.scores[idx]
        return idx0, idx1, cp / _SCORE_SCALE


def _flatten(indices: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ragged per-sample index lists into EmbeddingBag's (input, offsets) form."""
    offsets = torch.zeros(len(indices), dtype=torch.long)
    torch.cumsum(
        torch.tensor([len(idx) for idx in indices[:-1]], dtype=torch.long),
        dim=0,
        out=offsets[1:],
    )
    flat = torch.tensor([i for idx in indices for i in idx], dtype=torch.long)
    return flat, offsets


def _collate(
    batch: list[tuple[list[int], list[int], float]],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor
]:
    idx0_list = [item[0] for item in batch]
    idx1_list = [item[1] for item in batch]
    scores = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return _flatten(idx0_list), _flatten(idx1_list), scores


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NNUE(nn.Module):
    def __init__(self, n1: int = _N1, n2: int = 32, n3: int = 32) -> None:
        super().__init__()
        self.ft = nn.EmbeddingBag(_FEAT, n1, mode="sum")   # W1 as embedding lookup
        self.b1 = nn.Parameter(torch.zeros(n1))
        self.l2 = nn.Linear(2 * n1, n2)
        self.l3 = nn.Linear(n2, n3)
        self.l4 = nn.Linear(n3, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        # clamp(0, 1) after every layer has zero gradient outside that window, so a unit
        # whose pre-activation lands outside it at init is dead forever — gradient descent
        # can never pull it back. With ~15-30 active features summed per accumulator slot,
        # a small (e.g. std=0.01) embedding init keeps the pre-clamp distribution far too
        # narrow relative to [0, 1]: it's centered at 0, so ~50% of units start negative
        # and die immediately (verified empirically: 42-59% dead per layer at init, and
        # training never recovered past that regardless of data volume, weight decay, or
        # model width — all three were tried and ruled out before finding this).
        # A wider embedding std plus a mid-window bias keeps the pre-clamp distribution
        # centered inside (0, 1) with most of its spread away from either clamp edge.
        nn.init.normal_(self.ft.weight, std=0.05)
        nn.init.constant_(self.b1, 0.5)
        nn.init.constant_(self.l2.bias, 0.5)
        nn.init.constant_(self.l3.bias, 0.5)

    def accumulate(self, flat_indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Sum feature embeddings (= W1 row-sum per sample) + bias."""
        device = self.ft.weight.device
        return cast(torch.Tensor, self.ft(flat_indices.to(device), offsets.to(device))) + self.b1

    def forward(
        self,
        idx_stm: tuple[torch.Tensor, torch.Tensor],
        idx_opp: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        acc_stm = self.accumulate(*idx_stm)
        acc_opp = self.accumulate(*idx_opp)
        z1 = torch.cat([acc_stm, acc_opp], dim=1).clamp(0.0, 1.0)
        z2 = self.l2(z1).clamp(0.0, 1.0)
        z3 = self.l3(z2).clamp(0.0, 1.0)
        return cast(torch.Tensor, self.l4(z3)).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train/data.npz")
    p.add_argument("--out", default="train/nnue.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker processes for feature extraction")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay; the main lever against embedding-table "
                         "overfitting when active features touch most rows only a few times")
    p.add_argument("--patience", type=int, default=10,
                    help="Stop after this many epochs with no val improvement")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    fens: list[str] = list(data["fens"])
    scores: list[float] = list(data["scores"].astype(float))

    idx = list(range(len(fens)))
    random.Random(args.seed).shuffle(idx)
    split = int(len(idx) * (1 - args.val_frac))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = HalfKPDataset([fens[i] for i in train_idx], [scores[i] for i in train_idx])
    val_ds = HalfKPDataset([fens[i] for i in val_idx], [scores[i] for i in val_idx])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=_collate, num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=_collate, num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNUE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_without_improvement = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for idx0, idx1, targets in train_dl:
            targets = targets.to(device)
            preds = model(idx0, idx1)
            loss = loss_fn(preds, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(targets)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx0, idx1, targets in val_dl:
                targets = targets.to(device)
                preds = model(idx0, idx1)
                val_loss += loss_fn(preds, targets).item() * len(targets)

        train_mse = total_loss / len(train_ds)
        val_mse = val_loss / len(val_ds)
        # Reported in cp^2 (undoing _SCORE_SCALE) so the printed MSE is interpretable;
        # the comparison below stays in the network's native (scaled) units.
        cp2 = _SCORE_SCALE ** 2
        print(f"epoch {epoch:>3}  train={train_mse * cp2:.1f}  val={val_mse * cp2:.1f}")

        if val_mse < best_val:
            best_val = val_mse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"no val improvement in {args.patience} epochs, stopping early")
                break

    print(f"best val MSE={best_val * _SCORE_SCALE ** 2:.1f}, saved → {out_path}")


if __name__ == "__main__":
    main()
