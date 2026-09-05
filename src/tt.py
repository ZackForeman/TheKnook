"""Transposition table."""

from enum import IntEnum

import chess


class Flag(IntEnum):
    EXACT = 0        # stored score is exact
    LOWER_BOUND = 1  # score >= beta (caused a cutoff; true score may be higher)
    UPPER_BOUND = 2  # score <= alpha (failed to improve alpha; true score may be lower)


class TTEntry:
    __slots__ = ("depth", "flag", "move", "score")

    def __init__(self, depth: int, score: float, flag: Flag, move: chess.Move | None) -> None:
        self.depth = depth
        self.score = score
        self.flag = flag
        self.move = move


_table: dict[int, TTEntry] = {}


def probe(key: int) -> TTEntry | None:
    return _table.get(key)


def store(key: int, entry: TTEntry) -> None:
    existing = _table.get(key)
    if existing is None or entry.depth >= existing.depth:
        _table[key] = entry


def clear() -> None:
    _table.clear()
