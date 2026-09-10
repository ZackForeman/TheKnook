"""Generate NNUE training data: sample positions from PGN games and label with Stockfish.

Usage:
    python -m train.generate_data --pgn games.pgn --stockfish /usr/bin/stockfish \
        --depth 12 --out train/data.npz --limit 500000

Requires: Stockfish binary, python-chess with engine support.
Run locally; not part of the agent zip.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
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
    p.add_argument("--samples-per-game", type=int, default=8, help="Quiet positions per game")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help="Parallel Stockfish engine processes for labeling")
    return p.parse_args()


def _is_quiet(board: chess.Board) -> bool:
    return (
        not board.is_check()
        and not board.is_game_over()
        and board.halfmove_clock < 100
    )


def _collect_fens(pgn_path: Path, samples_per_game: int, limit: int, seed: int) -> list[str]:
    """Read the PGN and sample quiet positions per game — no engine needed here."""
    rng = random.Random(seed)
    fens: list[str] = []
    with pgn_path.open() as fh:
        while len(fens) < limit:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            positions: list[str] = []
            for move in game.mainline_moves():
                board.push(move)
                if _is_quiet(board):
                    positions.append(board.fen())
            fens.extend(rng.sample(positions, min(samples_per_game, len(positions))))
    return fens[:limit]


_engine: chess.engine.SimpleEngine | None = None
_limit: chess.engine.Limit | None = None


def _init_worker(stockfish_path: str, depth: int) -> None:
    """Runs once per worker process: open one persistent engine instance per worker."""
    global _engine, _limit
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _limit = chess.engine.Limit(depth=depth)


def _analyse(fen: str) -> tuple[str, float | None]:
    assert _engine is not None and _limit is not None
    info = _engine.analyse(chess.Board(fen), _limit)
    pov = info["score"].white()
    if pov.is_mate():
        return fen, None
    cp = pov.score()
    if cp is None or abs(cp) > 2_000:
        return fen, None
    return fen, float(cp)


def main() -> None:
    args = _parse_args()

    candidate_fens = _collect_fens(Path(args.pgn), args.samples_per_game, args.limit, args.seed)
    print(f"sampled {len(candidate_fens)} candidates, labeling with {args.workers} workers")

    fens: list[str] = []
    scores: list[float] = []
    with mp.Pool(
        args.workers, initializer=_init_worker, initargs=(args.stockfish, args.depth),
    ) as pool:
        for i, (fen, cp) in enumerate(pool.imap_unordered(_analyse, candidate_fens, chunksize=64)):
            if cp is not None:
                fens.append(fen)
                scores.append(cp)
            if (i + 1) % 10_000 == 0:
                print(f"labeled {i + 1}/{len(candidate_fens)}")

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
