"""Bitboard encoding and JIT evaluation."""

import chess
import numpy as np
from numba import njit

MATERIAL = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
# index = piece_type - 1: 0=pawn 1=knight 2=bishop 3=rook 4=queen 5=king

MOBILITY_WEIGHT = 4
BISHOP_PAIR_BONUS: np.int32 = np.int32(30)

# ---------------------------------------------------------------------------
# Piece-square tables
# Defined from White's perspective in board-view order:
#   first 8 values = rank 8 (a8..h8), last 8 values = rank 1 (a1..h1).
# PST[pt][sq]      → bonus for White piece of type pt on square sq
# PST[pt][sq ^ 56] → bonus for Black piece (mirrors rank vertically)
# ---------------------------------------------------------------------------

_PST_VIEW: list[list[int]] = [
    # PAWN (pt=0) — centre control and advancement
    [
         0,  0,  0,  0,  0,  0,  0,  0,   # rank 8
        50, 50, 50, 50, 50, 50, 50, 50,   # rank 7
        10, 10, 20, 30, 30, 20, 10, 10,   # rank 6
         5,  5, 10, 25, 25, 10,  5,  5,   # rank 5
         0,  0,  0, 20, 20,  0,  0,  0,   # rank 4
         5, -5,-10,  0,  0,-10, -5,  5,   # rank 3
         5, 10, 10,-20,-20, 10, 10,  5,   # rank 2
         0,  0,  0,  0,  0,  0,  0,  0,   # rank 1
    ],
    # KNIGHT (pt=1)
    [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    # BISHOP (pt=2)
    [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    # ROOK (pt=3) — 7th rank, central files
    [
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    # QUEEN (pt=4)
    [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    # KING middlegame (pt=5) — castle, avoid centre
    [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
]

# Endgame king — centralise and chase enemy pawns.
_KING_EG_VIEW: list[int] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

# Build PST[pt, sq]: board-view index i → sq = (7 - i//8)*8 + i%8
PST: np.ndarray = np.zeros((6, 64), dtype=np.int32)
for _pt, _view in enumerate(_PST_VIEW):
    for _i, _val in enumerate(_view):
        _sq = (7 - _i // 8) * 8 + (_i % 8)
        PST[_pt, _sq] = _val

PST_EG_KING: np.ndarray = np.zeros(64, dtype=np.int32)
for _i, _val in enumerate(_KING_EG_VIEW):
    PST_EG_KING[(7 - _i // 8) * 8 + (_i % 8)] = _val

# Phase weights: N=1, B=1, R=2, Q=4; max = 4+4+8+8 = 24
_PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)

# Passed-pawn bonus indexed by advancement (0 = just left start rank, 5 = one step from promotion)
PASSED_PAWN_BONUS: np.ndarray = np.array([0, 10, 20, 35, 55, 80], dtype=np.int32)

# ---------------------------------------------------------------------------
# Passed-pawn masks
# PASSED_PAWN_MASK[color, sq]: squares that must be free of enemy pawns
#   color=0 (white): ranks sq_rank+1..7 on files sq_file±1
#   color=1 (black): ranks 0..sq_rank-1 on files sq_file±1
# ---------------------------------------------------------------------------

PASSED_PAWN_MASK: np.ndarray = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8
    _wm, _bm = 0, 0
    for _df in range(max(0, _f - 1), min(8, _f + 2)):
        for _nr in range(_r + 1, 8):
            _wm |= 1 << (_nr * 8 + _df)
        for _nr in range(0, _r):
            _bm |= 1 << (_nr * 8 + _df)
    PASSED_PAWN_MASK[0, _sq] = np.uint64(_wm)
    PASSED_PAWN_MASK[1, _sq] = np.uint64(_bm)

# ---------------------------------------------------------------------------
# Pawn structure / king safety tables
# ---------------------------------------------------------------------------

DOUBLED_PAWN_PENALTY: np.int32 = np.int32(20)
ISOLATED_PAWN_PENALTY: np.int32 = np.int32(15)
ROOK_OPEN_FILE_BONUS: np.int32 = np.int32(25)
ROOK_SEMI_OPEN_BONUS: np.int32 = np.int32(12)
KING_PAWN_SHIELD_BONUS: np.int32 = np.int32(10)
KING_OPEN_FILE_PENALTY: np.int32 = np.int32(20)

FILE_MASKS: np.ndarray = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    _m = 0
    for _r in range(8):
        _m |= 1 << (_r * 8 + _f)
    FILE_MASKS[_f] = np.uint64(_m)

ADJACENT_FILES: np.ndarray = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    _m = 0
    if _f > 0:
        _m |= int(FILE_MASKS[_f - 1])
    if _f < 7:
        _m |= int(FILE_MASKS[_f + 1])
    ADJACENT_FILES[_f] = np.uint64(_m)

# Three squares directly in front of the king (rank+1 for white, rank-1 for black).
KING_SHIELD_MASK: np.ndarray = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8
    _wm, _bm = 0, 0
    for _df in (-1, 0, 1):
        _nf = _f + _df
        if not (0 <= _nf <= 7):
            continue
        if _r + 1 <= 7:
            _wm |= 1 << ((_r + 1) * 8 + _nf)
        if _r - 1 >= 0:
            _bm |= 1 << ((_r - 1) * 8 + _nf)
    KING_SHIELD_MASK[0, _sq] = np.uint64(_wm)
    KING_SHIELD_MASK[1, _sq] = np.uint64(_bm)


@njit(cache=False)
def popcount(bb: np.uint64) -> int:
    count = 0
    while bb:
        bb &= bb - np.uint64(1)
        count += 1
    return count


@njit(cache=False)
def _game_phase(bbs: np.ndarray) -> int:
    """Return game phase 0..24: 24 = full material, 0 = bare kings."""
    phase = 0
    for color in range(2):
        for pt in range(6):
            phase += _PHASE_WEIGHT[pt] * popcount(bbs[color][pt])
    if phase > 24:
        phase = 24
    return phase


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
    phase = _game_phase(bbs)
    score = 0

    # Pawns through queens: material + static PST
    for pt in range(5):
        score += MATERIAL[pt] * (popcount(bbs[stm][pt]) - popcount(bbs[opp][pt]))
        bb = bbs[stm][pt]
        while bb:
            lsb = bb & (~bb + np.uint64(1))
            sq = popcount(lsb - np.uint64(1))
            score += int(PST[pt, sq])
            bb ^= lsb
        bb = bbs[opp][pt]
        while bb:
            lsb = bb & (~bb + np.uint64(1))
            sq = popcount(lsb - np.uint64(1))
            score -= int(PST[pt, sq ^ 56])
            bb ^= lsb

    # King: tapered between MG and EG tables
    bb = bbs[stm][5]
    while bb:
        lsb = bb & (~bb + np.uint64(1))
        sq = popcount(lsb - np.uint64(1))
        score += (phase * int(PST[5, sq]) + (24 - phase) * int(PST_EG_KING[sq])) // 24
        bb ^= lsb
    bb = bbs[opp][5]
    while bb:
        lsb = bb & (~bb + np.uint64(1))
        sq = popcount(lsb - np.uint64(1))
        score -= (phase * int(PST[5, sq ^ 56]) + (24 - phase) * int(PST_EG_KING[sq ^ 56])) // 24
        bb ^= lsb

    # Passed pawns
    their_pawns = bbs[opp][0]
    bb = bbs[stm][0]
    while bb:
        lsb = bb & (~bb + np.uint64(1))
        sq = popcount(lsb - np.uint64(1))
        if (their_pawns & PASSED_PAWN_MASK[stm, sq]) == np.uint64(0):
            rank = sq // 8
            advance = (rank - 1) if stm == 0 else (6 - rank)
            if 0 <= advance < 6:
                score += int(PASSED_PAWN_BONUS[advance])
        bb ^= lsb

    our_pawns = bbs[stm][0]
    bb = bbs[opp][0]
    while bb:
        lsb = bb & (~bb + np.uint64(1))
        sq = popcount(lsb - np.uint64(1))
        if (our_pawns & PASSED_PAWN_MASK[opp, sq]) == np.uint64(0):
            rank = sq // 8
            advance = (rank - 1) if opp == 0 else (6 - rank)
            if 0 <= advance < 6:
                score -= int(PASSED_PAWN_BONUS[advance])
        bb ^= lsb

    # Bishop pair
    if popcount(bbs[stm][2]) >= 2:
        score += int(BISHOP_PAIR_BONUS)
    if popcount(bbs[opp][2]) >= 2:
        score -= int(BISHOP_PAIR_BONUS)

    # ---- Pawn structure: doubled and isolated ----
    # our_pawns / their_pawns already defined above in the passed-pawn section.
    for fi in range(8):
        file_mask = FILE_MASKS[fi]
        adj_mask = ADJACENT_FILES[fi]
        n_stm = popcount(our_pawns & file_mask)
        n_opp = popcount(their_pawns & file_mask)
        if n_stm > 1:
            score -= int(DOUBLED_PAWN_PENALTY) * (n_stm - 1)
        if n_opp > 1:
            score += int(DOUBLED_PAWN_PENALTY) * (n_opp - 1)
        if n_stm > 0 and popcount(our_pawns & adj_mask) == 0:
            score -= int(ISOLATED_PAWN_PENALTY) * n_stm
        if n_opp > 0 and popcount(their_pawns & adj_mask) == 0:
            score += int(ISOLATED_PAWN_PENALTY) * n_opp

    # ---- Rooks on open / semi-open files ----
    bb = bbs[stm][3]
    while bb:
        lsb = bb & (~bb + np.uint64(1))
        sq = popcount(lsb - np.uint64(1))
        fi = sq % 8
        fm = FILE_MASKS[fi]
        if popcount(our_pawns & fm) == 0:
            if popcount(their_pawns & fm) == 0:
                score += int(ROOK_OPEN_FILE_BONUS)
            else:
                score += int(ROOK_SEMI_OPEN_BONUS)
        bb ^= lsb

    bb = bbs[opp][3]
    while bb:
        lsb = bb & (~bb + np.uint64(1))
        sq = popcount(lsb - np.uint64(1))
        fi = sq % 8
        fm = FILE_MASKS[fi]
        if popcount(their_pawns & fm) == 0:
            if popcount(our_pawns & fm) == 0:
                score -= int(ROOK_OPEN_FILE_BONUS)
            else:
                score -= int(ROOK_SEMI_OPEN_BONUS)
        bb ^= lsb

    # ---- King safety: pawn shield + open files near king (tapered by phase) ----
    bb = bbs[stm][5]
    if bb:
        lsb = bb & (~bb + np.uint64(1))
        ksq = popcount(lsb - np.uint64(1))
        pawn_cover = popcount(our_pawns & KING_SHIELD_MASK[stm, ksq])
        kf = ksq % 8
        open_near = 0
        for dfi in range(-1, 2):
            fi = kf + dfi
            if fi >= 0 and fi <= 7 and popcount(our_pawns & FILE_MASKS[fi]) == 0:
                open_near += 1
        ks = pawn_cover * int(KING_PAWN_SHIELD_BONUS) - open_near * int(KING_OPEN_FILE_PENALTY)
        score += ks * phase // 24

    bb = bbs[opp][5]
    if bb:
        lsb = bb & (~bb + np.uint64(1))
        ksq = popcount(lsb - np.uint64(1))
        pawn_cover = popcount(their_pawns & KING_SHIELD_MASK[opp, ksq])
        kf = ksq % 8
        open_near = 0
        for dfi in range(-1, 2):
            fi = kf + dfi
            if fi >= 0 and fi <= 7 and popcount(their_pawns & FILE_MASKS[fi]) == 0:
                open_near += 1
        ks = pawn_cover * int(KING_PAWN_SHIELD_BONUS) - open_near * int(KING_OPEN_FILE_PENALTY)
        score -= ks * phase // 24

    return score + MOBILITY_WEIGHT * mobility


# Warm all JIT functions at import — compilation lands in the 90 s init budget.
_bbs, _stm = encode(chess.Board())
popcount(np.uint64(0xFFFF))
_game_phase(_bbs)
evaluate(_bbs, _stm, 20)
