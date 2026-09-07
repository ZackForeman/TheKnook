"""Regression test: mobility masking must exclude the piece's own occupied squares."""

import numpy as np

from src.bitboard import popcount
from src.movegen import knight_attacks_bb


def test_knight_mobility_excludes_own_occupied_squares() -> None:
    # Knight on a1 can only reach b3 and c2; if both are occupied by its own side,
    # masked mobility must be 0, not 2.
    knight_sq = 0  # a1
    own_occ = np.uint64((1 << 17) | (1 << 10))  # b3, c2 occupied by own pieces
    attacks = knight_attacks_bb(knight_sq)
    assert popcount(attacks) == 2
    assert popcount(attacks & ~own_occ) == 0
