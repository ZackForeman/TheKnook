"""Generate a diverse PGN corpus via local self-play, for train.generate_data to label.

Plays every unordered pair of agents in --agents against each other, alternating colours,
from randomized short openings (so a handful of deterministic baselines don't just repeat
the same lines from the standard start). Output feeds straight into:

    python -m train.generate_data --pgn train/selfplay.pgn --stockfish <path> ...

Usage:
    python -m train.selfplay --games-per-pair 100 --out train/selfplay.pgn

Run locally; not part of the agent zip.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import chess

from harness.referee import play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

DEFAULT_AGENTS = [
    "baselines/greedy",
    "baselines/minimax",
    "baselines/numba",
    "baselines/random",
    "baselines/v1",
    ".",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    p.add_argument("--games-per-pair", type=int, default=100)
    p.add_argument("--base-ms", type=int, default=3_000)
    p.add_argument("--increment-ms", type=int, default=50)
    p.add_argument("--ply-cap", type=int, default=PLY_CAP)
    p.add_argument("--max-opening-plies", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="train/selfplay.pgn")
    return p.parse_args()


def _random_opening_fen(rng: random.Random, max_plies: int) -> str:
    board = chess.Board()
    for _ in range(rng.randint(0, max_plies)):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        board.push(move)
    return board.fen()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    pairs = list(itertools.combinations(args.agents, 2))
    total = len(pairs) * args.games_per_pair
    print(f"{len(pairs)} pairings x {args.games_per_pair} games = {total} games")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    played = 0
    with out_path.open("w") as fh:
        for a, b in pairs:
            path_a, path_b = Path(a).resolve(), Path(b).resolve()
            for game in range(args.games_per_pair):
                white, black = (path_a, path_b) if game % 2 == 0 else (path_b, path_a)
                start_fen = _random_opening_fen(rng, args.max_opening_plies)
                outcome = play_match(
                    local(white),
                    local(black),
                    args.base_ms,
                    args.increment_ms,
                    ply_cap=args.ply_cap,
                    start_fen=start_fen,
                )
                fh.write(outcome.pgn + "\n\n")
                played += 1
                if played % 20 == 0:
                    print(f"played {played}/{total}")

    print(f"saved {played} games -> {out_path}")


if __name__ == "__main__":
    main()
