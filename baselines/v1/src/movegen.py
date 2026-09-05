"""Pre-computed bitboard attack tables and pseudo-legal move generation.

Square numbering: 0=a1, 63=h8, LSB-first (matches python-chess).
  file = sq % 8  (0=a .. 7=h)
  rank = sq // 8 (0=rank1 .. 7=rank8)
"""

import chess
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Construction helpers (Python only, called at module load)
# ---------------------------------------------------------------------------

def _set_bit(bb: int, sq: int) -> int:
    return bb | (1 << sq)


def _expand_bits(index: int, mask: int) -> int:
    """Scatter the low bits of index into the positions set in mask (inverse of pext)."""
    result = 0
    bit = 0
    while mask:
        lsb = mask & -mask
        if index & (1 << bit):
            result |= lsb
        mask ^= lsb
        bit += 1
    return result


def _pext_py(val: int, mask: int) -> int:
    """Python-only pext: extract bits of val at mask positions, compact to low bits."""
    result = 0
    bit = 0
    while mask:
        lsb = mask & -mask
        if val & lsb:
            result |= 1 << bit
        mask ^= lsb
        bit += 1
    return result


def _ray_attacks_rook(sq: int, occ: int) -> int:
    attacks = 0
    f, r = sq % 8, sq // 8
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nf, nr = f + df, r + dr
        while 0 <= nf <= 7 and 0 <= nr <= 7:
            nsq = nr * 8 + nf
            attacks = _set_bit(attacks, nsq)
            if occ & (1 << nsq):
                break
            nf += df
            nr += dr
    return attacks


def _ray_attacks_bishop(sq: int, occ: int) -> int:
    attacks = 0
    f, r = sq % 8, sq // 8
    for df, dr in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        nf, nr = f + df, r + dr
        while 0 <= nf <= 7 and 0 <= nr <= 7:
            nsq = nr * 8 + nf
            attacks = _set_bit(attacks, nsq)
            if occ & (1 << nsq):
                break
            nf += df
            nr += dr
    return attacks


# ---------------------------------------------------------------------------
# Non-sliding piece tables
# ---------------------------------------------------------------------------

KNIGHT_ATTACKS: np.ndarray = np.zeros(64, dtype=np.uint64)
KING_ATTACKS: np.ndarray = np.zeros(64, dtype=np.uint64)
PAWN_ATTACKS: np.ndarray = np.zeros((2, 64), dtype=np.uint64)
PAWN_SINGLE_PUSH: np.ndarray = np.zeros((2, 64), dtype=np.uint64)
PAWN_DOUBLE_PUSH_MASK: np.ndarray = np.zeros((2, 64), dtype=np.uint64)

for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8

    # Knight
    _bb = 0
    for _df, _dr in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        _nf, _nr = _f + _df, _r + _dr
        if 0 <= _nf <= 7 and 0 <= _nr <= 7:
            _bb = _set_bit(_bb, _nr * 8 + _nf)
    KNIGHT_ATTACKS[_sq] = np.uint64(_bb)

    # King
    _bb = 0
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            if _df == 0 and _dr == 0:
                continue
            _nf, _nr = _f + _df, _r + _dr
            if 0 <= _nf <= 7 and 0 <= _nr <= 7:
                _bb = _set_bit(_bb, _nr * 8 + _nf)
    KING_ATTACKS[_sq] = np.uint64(_bb)

    # Pawn attacks: [0]=white attacks rank+1, [1]=black attacks rank-1
    for _df in (-1, 1):
        _nf = _f + _df
        if 0 <= _nf <= 7:
            if _r + 1 <= 7:
                _wsq = (_r + 1) * 8 + _nf
                PAWN_ATTACKS[0, _sq] = np.uint64(int(PAWN_ATTACKS[0, _sq]) | (1 << _wsq))
            if _r - 1 >= 0:
                _bsq = (_r - 1) * 8 + _nf
                PAWN_ATTACKS[1, _sq] = np.uint64(int(PAWN_ATTACKS[1, _sq]) | (1 << _bsq))

    # Pawn single push
    if _r + 1 <= 7:
        PAWN_SINGLE_PUSH[0, _sq] = np.uint64(1 << ((_r + 1) * 8 + _f))
    if _r - 1 >= 0:
        PAWN_SINGLE_PUSH[1, _sq] = np.uint64(1 << ((_r - 1) * 8 + _f))

    # Pawn double push (only from starting rank)
    if _r == 1:  # white starting rank
        PAWN_DOUBLE_PUSH_MASK[0, _sq] = np.uint64(1 << ((_r + 2) * 8 + _f))
    if _r == 6:  # black starting rank
        PAWN_DOUBLE_PUSH_MASK[1, _sq] = np.uint64(1 << ((_r - 2) * 8 + _f))


# ---------------------------------------------------------------------------
# Sliding piece occupancy masks
# ---------------------------------------------------------------------------

ROOK_MASKS: np.ndarray = np.zeros(64, dtype=np.uint64)
BISHOP_MASKS: np.ndarray = np.zeros(64, dtype=np.uint64)

for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8
    _bb = 0
    # Rook: rank/file squares excluding edges
    for _df in range(1, 7):       # files b..g
        if _df != _f:
            _bb = _set_bit(_bb, _r * 8 + _df)
    for _dr in range(1, 7):       # ranks 2..7
        if _dr != _r:
            _bb = _set_bit(_bb, _dr * 8 + _f)
    ROOK_MASKS[_sq] = np.uint64(_bb)

    _bb = 0
    # Bishop: diagonals excluding edges
    for _df, _dr in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        _nf, _nr = _f + _df, _r + _dr
        while 0 <= _nf <= 7 and 0 <= _nr <= 7:
            # exclude border squares
            if 1 <= _nf <= 6 and 1 <= _nr <= 6:
                _bb = _set_bit(_bb, _nr * 8 + _nf)
            _nf += _df
            _nr += _dr
    BISHOP_MASKS[_sq] = np.uint64(_bb)


# ---------------------------------------------------------------------------
# Sliding piece attack tables
# ---------------------------------------------------------------------------

ROOK_ATTACKS: np.ndarray = np.zeros((64, 4096), dtype=np.uint64)
BISHOP_ATTACKS: np.ndarray = np.zeros((64, 512), dtype=np.uint64)

for _sq in range(64):
    _mask = int(ROOK_MASKS[_sq])
    _n = bin(_mask).count("1")
    for _i in range(1 << _n):
        _occ = _expand_bits(_i, _mask)
        _idx = _pext_py(_occ, _mask)
        ROOK_ATTACKS[_sq, _idx] = np.uint64(_ray_attacks_rook(_sq, _occ))

for _sq in range(64):
    _mask = int(BISHOP_MASKS[_sq])
    _n = bin(_mask).count("1")
    for _i in range(1 << _n):
        _occ = _expand_bits(_i, _mask)
        _idx = _pext_py(_occ, _mask)
        BISHOP_ATTACKS[_sq, _idx] = np.uint64(_ray_attacks_bishop(_sq, _occ))


# ---------------------------------------------------------------------------
# JIT lookup functions
# ---------------------------------------------------------------------------

@njit(cache=False)
def pext(val: np.uint64, mask: np.uint64) -> int:
    result, bit = np.uint64(0), np.uint64(0)
    while mask:
        lsb = mask & (~mask + np.uint64(1))
        if val & lsb:
            result |= np.uint64(1) << bit
        mask ^= lsb
        bit += np.uint64(1)
    return int(result)


@njit(cache=False)
def knight_attacks_bb(sq: int) -> np.uint64:
    return np.uint64(KNIGHT_ATTACKS[sq])


@njit(cache=False)
def king_attacks_bb(sq: int) -> np.uint64:
    return np.uint64(KING_ATTACKS[sq])


@njit(cache=False)
def pawn_attacks_bb(color: int, sq: int) -> np.uint64:
    return np.uint64(PAWN_ATTACKS[color, sq])


@njit(cache=False)
def rook_attacks_bb(sq: int, occupied: np.uint64) -> np.uint64:
    mask = ROOK_MASKS[sq]
    return np.uint64(ROOK_ATTACKS[sq, pext(occupied & mask, mask)])


@njit(cache=False)
def bishop_attacks_bb(sq: int, occupied: np.uint64) -> np.uint64:
    mask = BISHOP_MASKS[sq]
    return np.uint64(BISHOP_ATTACKS[sq, pext(occupied & mask, mask)])


@njit(cache=False)
def queen_attacks_bb(sq: int, occupied: np.uint64) -> np.uint64:
    return rook_attacks_bb(sq, occupied) | bishop_attacks_bb(sq, occupied)


# ---------------------------------------------------------------------------
# Pseudo-legal move generation
# ---------------------------------------------------------------------------

_PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def generate_pseudo_legal(
    bbs: np.ndarray,
    stm: int,
    ep_sq: int,
    castling_rights: int,
) -> list[chess.Move]:
    """Return pseudo-legal moves; caller must filter with board.is_legal()."""
    opp = 1 - stm

    my_occ = np.uint64(0)
    for pt in range(6):
        my_occ |= bbs[stm, pt]
    their_occ = np.uint64(0)
    for pt in range(6):
        their_occ |= bbs[opp, pt]
    occupied = my_occ | their_occ

    ep_bit = np.uint64(0) if ep_sq < 0 else np.uint64(1 << ep_sq)
    back_rank = 7 if stm == 0 else 0   # white promotes on rank 7, black on 0
    start_rank = 1 if stm == 0 else 6  # white double-push from rank 1

    moves: list[chess.Move] = []

    # Piece type indices: 0=pawn 1=knight 2=bishop 3=rook 4=queen 5=king
    for pt in range(6):
        bb = int(bbs[stm, pt])
        while bb:
            from_sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1

            if pt == 0:  # pawn
                # Attacks (captures + en passant)
                atk = int(pawn_attacks_bb(stm, from_sq)) & (int(their_occ) | int(ep_bit))
                while atk:
                    to_sq = (atk & -atk).bit_length() - 1
                    atk &= atk - 1
                    if to_sq // 8 == back_rank:
                        for promo in _PROMO_PIECES:
                            moves.append(chess.Move(from_sq, to_sq, promotion=promo))
                    else:
                        moves.append(chess.Move(from_sq, to_sq))

                # Single push
                single = int(PAWN_SINGLE_PUSH[stm, from_sq]) & ~int(occupied)
                if single:
                    to_sq = (single & -single).bit_length() - 1
                    if to_sq // 8 == back_rank:
                        for promo in _PROMO_PIECES:
                            moves.append(chess.Move(from_sq, to_sq, promotion=promo))
                    else:
                        moves.append(chess.Move(from_sq, to_sq))
                        # Double push only if single-push square was clear
                        if from_sq // 8 == start_rank:
                            double = int(PAWN_DOUBLE_PUSH_MASK[stm, from_sq]) & ~int(occupied)
                            if double:
                                to_sq2 = (double & -double).bit_length() - 1
                                moves.append(chess.Move(from_sq, to_sq2))
            else:
                if pt == 1:
                    atk = int(knight_attacks_bb(from_sq))
                elif pt == 2:
                    atk = int(bishop_attacks_bb(from_sq, occupied))
                elif pt == 3:
                    atk = int(rook_attacks_bb(from_sq, occupied))
                elif pt == 4:
                    atk = int(queen_attacks_bb(from_sq, occupied))
                else:  # king
                    atk = int(king_attacks_bb(from_sq))

                atk &= ~int(my_occ)
                while atk:
                    to_sq = (atk & -atk).bit_length() - 1
                    atk &= atk - 1
                    moves.append(chess.Move(from_sq, to_sq))

    # Castling — emit candidate; board.is_legal handles check-through validation
    if stm == 0:  # white
        if castling_rights & chess.BB_H1 and not (occupied & np.uint64(chess.BB_F1 | chess.BB_G1)):
            moves.append(chess.Move(chess.E1, chess.G1))
        if castling_rights & chess.BB_A1 and not (
            occupied & np.uint64(chess.BB_B1 | chess.BB_C1 | chess.BB_D1)
        ):
            moves.append(chess.Move(chess.E1, chess.C1))
    else:  # black
        if castling_rights & chess.BB_H8 and not (occupied & np.uint64(chess.BB_F8 | chess.BB_G8)):
            moves.append(chess.Move(chess.E8, chess.G8))
        if castling_rights & chess.BB_A8 and not (
            occupied & np.uint64(chess.BB_B8 | chess.BB_C8 | chess.BB_D8)
        ):
            moves.append(chess.Move(chess.E8, chess.C8))

    return moves


# ---------------------------------------------------------------------------
# Warm-up — trigger JIT compilation inside the 90 s init budget
# ---------------------------------------------------------------------------

_SQ = 28  # e4
_OCC = np.uint64(0x00FF00000000FF00)

pext(np.uint64(0xABCD), np.uint64(0xFFFF))
knight_attacks_bb(_SQ)
king_attacks_bb(_SQ)
pawn_attacks_bb(0, _SQ)
pawn_attacks_bb(1, _SQ)
rook_attacks_bb(_SQ, _OCC)
bishop_attacks_bb(_SQ, _OCC)
queen_attacks_bb(_SQ, _OCC)
