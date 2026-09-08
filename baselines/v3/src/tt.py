"""Transposition table: fixed-size, zobrist-keyed, depth-preferred replacement."""

from enum import IntEnum

import chess
import numpy as np


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


# Bounded, array-backed table (no unbounded growth over a long game). ~4.2M slots at
# _TABLE_BITS=22 costs well under the platform's 2 GB/process budget.
_TABLE_BITS = 22
_TABLE_SIZE = 1 << _TABLE_BITS
_MASK = _TABLE_SIZE - 1
_EMPTY_KEY = np.uint64(2**64 - 1)  # sentinel; a real zobrist hash matching this is negligible

_keys: np.ndarray = np.full(_TABLE_SIZE, _EMPTY_KEY, dtype=np.uint64)
_depths: np.ndarray = np.zeros(_TABLE_SIZE, dtype=np.int16)
_scores: np.ndarray = np.zeros(_TABLE_SIZE, dtype=np.float32)
_flags: np.ndarray = np.zeros(_TABLE_SIZE, dtype=np.int8)
_moves: list[chess.Move | None] = [None] * _TABLE_SIZE


def probe(key: int) -> TTEntry | None:
    idx = key & _MASK
    if _keys[idx] != np.uint64(key):
        return None
    return TTEntry(int(_depths[idx]), float(_scores[idx]), Flag(int(_flags[idx])), _moves[idx])


def store(key: int, entry: TTEntry) -> None:
    idx = key & _MASK
    k = np.uint64(key)
    same_key = _keys[idx] == k
    if not same_key or entry.depth >= int(_depths[idx]):
        _keys[idx] = k
        _depths[idx] = entry.depth
        _scores[idx] = entry.score
        _flags[idx] = int(entry.flag)
        _moves[idx] = entry.move


def clear() -> None:
    _keys.fill(_EMPTY_KEY)
    _depths.fill(0)
    _scores.fill(0.0)
    _flags.fill(0)
    _moves[:] = [None] * _TABLE_SIZE
