"""Generate NNUE training data: sample positions from PGN games and label with Stockfish.

Usage:
    python -m train.generate_data --pgn games.pgn --stockfish /usr/bin/stockfish \
        --depth 12 --out train/data.npz --limit 500000

Requires: Stockfish binary, python-chess with engine support.
Run locally; not part of the agent zip.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", required=True, help="PGN file of games to sample from")
    p.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--out", default="train/data.npz")
    p.add_argument("--limit", type=int, default=500_000, help="Max positions to collect")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _is_quiet(board: chess.Board) -> bool:
    return (
        not board.is_check()
        and not board.is_game_over()
        and board.halfmove_clock < 100
    )


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    fens: list[str] = []
    scores: list[float] = []

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    limit = chess.engine.Limit(depth=args.depth)

    pgn_path = Path(args.pgn)
    collected = 0

    with pgn_path.open() as fh:
        while collected < args.limit:
            game = chess.pgn.read_game(fh)
            if game is None:
                break

            board = game.board()
            positions: list[str] = []
            for move in game.mainline_moves():
                board.push(move)
                if _is_quiet(board):
                    positions.append(board.fen())

            sample = rng.sample(positions, min(8, len(positions)))
            for fen in sample:
                b = chess.Board(fen)
                info = engine.analyse(b, limit)
                pov = info["score"].white()
                if pov.is_mate():
                    continue
                cp = pov.score()
                if cp is None or abs(cp) > 2_000:
                    continue
                fens.append(fen)
                scores.append(float(cp))
                collected += 1
                if collected >= args.limit:
                    break

            if collected % 10_000 == 0:
                print(f"collected {collected}/{args.limit}")

    engine.quit()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        fens=np.array(fens, dtype=object),
        scores=np.array(scores, dtype=np.float32),
    )
    print(f"saved {len(fens)} positions → {out_path}")


if __name__ == "__main__":
    main()
