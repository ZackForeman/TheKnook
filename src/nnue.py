"""NNUE evaluation: HalfKP features, incremental accumulators, 4-layer network.

Architecture (chess HalfKP):
  Features per perspective: 64 king_sq x 10 (5 piece_types x 2 colours) x 64 sq = 40 960
  L1 accumulator: (2, N1) = (2, 16) float32  — one per side, shared W1/b1
  L2-L4: [512 → 32 → 32 → 1] with clipped-ReLU after L2 and L3
  Output: centipawns from side-to-move perspective
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
from numba import njit

from .bitboard import encode

# ---------------------------------------------------------------------------
# Hyper-parameters (must match training config exactly)
# ---------------------------------------------------------------------------
N1: int = 16  # accumulator width per side; L2 input is 2*N1 = 32
# The network is trained against cp / _SCORE_SCALE (see train/train_nnue.py) because
# its clamp(0, 1) hidden activations bound the raw output to a small range near its
# init scale — reaching real centipawn magnitudes directly would need an impractical
# number of training steps. Undo that scaling here to get centipawns back out.
_SCORE_SCALE: float = 400.0

# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
_WEIGHTS_PATH: Path = Path(__file__).parent.parent / "weights" / "nnue.npz"
NNUE_AVAILABLE: bool = _WEIGHTS_PATH.exists()


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    w1_raw: np.ndarray = d["w1"]
    w1 = w1_raw.astype(np.float32) / 64.0 if w1_raw.dtype == np.int16 else w1_raw.astype(np.float32)
    return (
        w1,
        d["b1"].astype(np.float32),
        d["w2"].astype(np.float32),
        d["b2"].astype(np.float32),
        d["w3"].astype(np.float32),
        d["b3"].astype(np.float32),
        d["w4"].astype(np.float32),
        d["b4"].astype(np.float32),
    )


_n2, _n3 = 32, 32

if NNUE_AVAILABLE:
    _W1, _b1, _W2, _b2, _W3, _b3, _W4, _b4 = _load(_WEIGHTS_PATH)
else:
    _W1 = np.zeros((40960, N1), dtype=np.float32)
    _b1 = np.zeros(N1, dtype=np.float32)
    _W2 = np.zeros((2 * N1, _n2), dtype=np.float32)
    _b2 = np.zeros(_n2, dtype=np.float32)
    _W3 = np.zeros((_n2, _n3), dtype=np.float32)
    _b3 = np.zeros(_n3, dtype=np.float32)
    _W4 = np.zeros((_n3, 1), dtype=np.float32)
    _b4 = np.zeros(1, dtype=np.float32)

# ---------------------------------------------------------------------------
# JIT-compiled kernels
# ---------------------------------------------------------------------------

@njit(cache=False)
def _popcount_nnue(bb: np.uint64) -> int:
    count = 0
    while bb:
        bb &= bb - np.uint64(1)
        count += 1
    return count


@njit(cache=False)
def _halfkp_idx(
    king_sq: int,
    actual_color: int,
    perspective: int,
    pt_idx: int,
    piece_sq: int,
) -> int:
    """HalfKP index 0..40959 for one piece from a given perspective.

    perspective=0 (white): board as-is, king_sq=white king's sq, pc=actual_color
    perspective=1 (black): flip rank (^56), swap colours
    pt_idx: 0=pawn 1=knight 2=bishop 3=rook 4=queen
    """
    if perspective == 0:
        ks = king_sq
        pc = actual_color
        sq = piece_sq
    else:
        ks = king_sq ^ 56
        pc = 1 - actual_color
        sq = piece_sq ^ 56
    return ks * 640 + pc * 320 + pt_idx * 64 + sq


@njit(cache=False)
def refresh_accumulator(
    bbs: np.ndarray,
    white_ks: int,
    black_ks: int,
    w1: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    """Full recompute of both accumulators from the (2,6) bitboard array."""
    n1 = b1.shape[0]
    acc = np.empty((2, n1), dtype=np.float32)
    for perspective in range(2):
        ks = white_ks if perspective == 0 else black_ks
        for k in range(n1):
            acc[perspective, k] = b1[k]
        for actual_color in range(2):
            for pt_idx in range(5):
                bb = bbs[actual_color, pt_idx]
                while bb:
                    lsb = bb & (~bb + np.uint64(1))
                    sq = _popcount_nnue(lsb - np.uint64(1))
                    idx = _halfkp_idx(ks, actual_color, perspective, pt_idx, sq)
                    for k in range(n1):
                        acc[perspective, k] += w1[idx, k]
                    bb ^= lsb
    return acc


@njit(cache=False)
def _add_piece(
    acc: np.ndarray,
    white_ks: int,
    black_ks: int,
    actual_color: int,
    pt_idx: int,
    piece_sq: int,
    w1: np.ndarray,
) -> None:
    n1 = acc.shape[1]
    for perspective in range(2):
        ks = white_ks if perspective == 0 else black_ks
        idx = _halfkp_idx(ks, actual_color, perspective, pt_idx, piece_sq)
        for k in range(n1):
            acc[perspective, k] += w1[idx, k]


@njit(cache=False)
def _sub_piece(
    acc: np.ndarray,
    white_ks: int,
    black_ks: int,
    actual_color: int,
    pt_idx: int,
    piece_sq: int,
    w1: np.ndarray,
) -> None:
    n1 = acc.shape[1]
    for perspective in range(2):
        ks = white_ks if perspective == 0 else black_ks
        idx = _halfkp_idx(ks, actual_color, perspective, pt_idx, piece_sq)
        for k in range(n1):
            acc[perspective, k] -= w1[idx, k]


@njit(cache=False)
def nnue_forward(
    acc: np.ndarray,
    stm: int,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    w4: np.ndarray,
    b4: np.ndarray,
) -> float:
    """Forward pass layers 2-4. Returns centipawn score from stm's perspective."""
    n1 = acc.shape[1]
    n2 = b2.shape[0]
    n3 = b3.shape[0]

    # Clipped-ReLU on concatenated accumulators; active side first (eq. 17)
    z1 = np.empty(2 * n1, dtype=np.float32)
    for i in range(n1):
        v = acc[stm, i]
        z1[i] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        v2 = acc[1 - stm, i]
        z1[n1 + i] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)

    # L2
    a2 = np.empty(n2, dtype=np.float32)
    for j in range(n2):
        s = b2[j]
        for i in range(2 * n1):
            s += z1[i] * w2[i, j]
        a2[j] = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)

    # L3
    a3 = np.empty(n3, dtype=np.float32)
    for j in range(n3):
        s = b3[j]
        for i in range(n2):
            s += a2[i] * w3[i, j]
        a3[j] = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)

    # L4 — no activation
    out = b4[0]
    for i in range(n3):
        out += a3[i] * w4[i, 0]
    return float(out)


# ---------------------------------------------------------------------------
# Python-level incremental update (calls numba kernels)
# ---------------------------------------------------------------------------

def _incremental_update(
    acc: np.ndarray,
    board: chess.Board,
    move: chess.Move,
    w1: np.ndarray,
) -> None:
    """Apply one non-king move's feature delta to acc in-place. board is pre-move state."""
    piece = board.piece_at(move.from_square)
    assert piece is not None

    moving_color: int = 0 if piece.color == chess.WHITE else 1
    pt_from: int = piece.piece_type - 1
    pt_to: int = (move.promotion - 1) if move.promotion is not None else pt_from

    wk: chess.Square | None = board.king(chess.WHITE)
    bk: chess.Square | None = board.king(chess.BLACK)
    assert wk is not None and bk is not None
    white_ks, black_ks = int(wk), int(bk)

    _sub_piece(acc, white_ks, black_ks, moving_color, pt_from, move.from_square, w1)
    _add_piece(acc, white_ks, black_ks, moving_color, pt_to, move.to_square, w1)

    captured = board.piece_at(move.to_square)
    if captured is not None:
        cap_color = 0 if captured.color == chess.WHITE else 1
        _sub_piece(acc, white_ks, black_ks, cap_color, captured.piece_type - 1,
                   move.to_square, w1)

    if board.is_en_passant(move):
        ep_sq: int = move.to_square + (-8 if board.turn == chess.WHITE else 8)
        ep_color: int = 1 if board.turn == chess.WHITE else 0
        _sub_piece(acc, white_ks, black_ks, ep_color, 0, ep_sq, w1)


# ---------------------------------------------------------------------------
# AccumulatorStack — maintained in parallel with board.push / board.pop
# ---------------------------------------------------------------------------

class AccumulatorStack:
    """Keeps a stack of (2, N1) accumulators that mirrors the search ply stack.

    None entries are used as a lazy sentinel when a king moves; the accumulator
    is recomputed on the first evaluate call at that ply.
    """

    __slots__ = ("_stack",)

    def __init__(self, board: chess.Board) -> None:
        bbs, _ = encode(board)
        wk: chess.Square | None = board.king(chess.WHITE)
        bk: chess.Square | None = board.king(chess.BLACK)
        assert wk is not None and bk is not None
        initial = refresh_accumulator(bbs, int(wk), int(bk), _W1, _b1)
        self._stack: list[np.ndarray | None] = [initial]

    def push(self, move: chess.Move, board: chess.Board) -> None:
        """Call BEFORE board.push(move). board must be in the pre-move state."""
        if move == chess.Move.null():
            # Null move: no piece changes; stm flips but accumulator is unchanged
            self._stack.append(self._stack[-1])
            return

        piece = board.piece_at(move.from_square)
        assert piece is not None

        if piece.piece_type == chess.KING or board.is_castling(move):
            # King's square changes → all feature indices for that perspective change.
            # Push None; get_or_refresh() will recompute lazily.
            self._stack.append(None)
            return

        cur = self.get_or_refresh(board)
        new_acc = cur.copy()
        _incremental_update(new_acc, board, move, _W1)
        self._stack.append(new_acc)

    def pop(self) -> None:
        self._stack.pop()

    def get_or_refresh(self, board: chess.Board) -> np.ndarray:
        """Return current accumulator, computing a full refresh if king moved."""
        acc = self._stack[-1]
        if acc is None:
            bbs, _ = encode(board)
            wk: chess.Square | None = board.king(chess.WHITE)
            bk: chess.Square | None = board.king(chess.BLACK)
            assert wk is not None and bk is not None
            acc = refresh_accumulator(bbs, int(wk), int(bk), _W1, _b1)
            self._stack[-1] = acc
        return acc


# ---------------------------------------------------------------------------
# Public evaluation entry-point
# ---------------------------------------------------------------------------

def nnue_evaluate(board: chess.Board, acc_stack: AccumulatorStack) -> int:
    """Return NNUE centipawn score from side-to-move perspective."""
    stm = 0 if board.turn == chess.WHITE else 1
    acc = acc_stack.get_or_refresh(board)
    return int(nnue_forward(acc, stm, _W2, _b2, _W3, _b3, _W4, _b4) * _SCORE_SCALE)


# ---------------------------------------------------------------------------
# JIT warm-up — runs during the 90 s import budget
# ---------------------------------------------------------------------------
_warmup_board = chess.Board()
_warmup_bbs, _ = encode(_warmup_board)
_warmup_wk: chess.Square | None = _warmup_board.king(chess.WHITE)
_warmup_bk: chess.Square | None = _warmup_board.king(chess.BLACK)
assert _warmup_wk is not None and _warmup_bk is not None
_warmup_acc = refresh_accumulator(_warmup_bbs, int(_warmup_wk), int(_warmup_bk), _W1, _b1)
_add_piece(_warmup_acc, int(_warmup_wk), int(_warmup_bk), 0, 3, 0, _W1)
_sub_piece(_warmup_acc, int(_warmup_wk), int(_warmup_bk), 0, 3, 0, _W1)
nnue_forward(_warmup_acc, 0, _W2, _b2, _W3, _b3, _W4, _b4)
del _warmup_board, _warmup_bbs, _warmup_acc, _warmup_wk, _warmup_bk
