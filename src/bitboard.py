"""Bitboard encoding and JIT evaluation."""

import chess
import numpy as np
from numba import njit

MATERIAL = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
# index = piece_type - 1: 0=pawn 1=knight 2=bishop 3=rook 4=queen 5=king
MOBILITY_WEIGHT = 4


@njit(cache=False)
def popcount(bb: np.uint64) -> int:
    count = 0
    while bb:
        bb &= bb - np.uint64(1)
        count += 1
    return count


def encode(board: chess.Board) -> tuple[np.ndarray, int]:
    """Return (bbs, stm): bbs shape (2,6) uint64, stm 0=white 1=black."""
    bbs = np.zeros((2, 6), dtype=np.uint64)
    for color in (chess.WHITE, chess.BLACK):
        ci = 0 if color == chess.WHITE else 1
        for pt in range(chess.PAWN, chess.KING + 1):
            bbs[ci][pt - 1] = np.uint64(int(board.pieces(pt, color)))
    stm = 0 if board.turn == chess.WHITE else 1
    return bbs, stm


@njit(cache=False)
def evaluate(bbs: np.ndarray, stm: int, mobility: int) -> int:
    opp = 1 - stm
    score = 0
    for pt in range(6):
        score += MATERIAL[pt] * (popcount(bbs[stm][pt]) - popcount(bbs[opp][pt]))
    return score + MOBILITY_WEIGHT * mobility


# Warm both JIT functions at import — compilation lands in the 90 s init budget.
_bbs, _stm = encode(chess.Board())
popcount(np.uint64(0xFFFF))
evaluate(_bbs, _stm, 20)
