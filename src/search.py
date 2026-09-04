"""Negamax search with alpha-beta pruning, iterative deepening, move ordering, and TT."""

import math
import time
from typing import cast

import chess

from . import tt as _tt
from .bitboard import encode, evaluate
from .movegen import generate_pseudo_legal
from .tt import Flag, TTEntry

MATE = 1e6
_MAX_DEPTH = 10
_MOVES_ESTIMATE = 60  # expected moves remaining — used to divide the time budget

_PIECE_VAL: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


class _Timeout(Exception):
    pass


def _get_legal_moves(board: chess.Board) -> list[chess.Move]:
    bbs, stm = encode(board)
    ep_sq = board.ep_square if board.ep_square is not None else -1
    candidates = generate_pseudo_legal(bbs, stm, ep_sq, board.castling_rights)
    return [m for m in candidates if board.is_legal(m)]


def _move_score(board: chess.Board, move: chess.Move, tt_move: chess.Move | None) -> int:
    if move == tt_move:
        return 20_000
    victim = board.piece_type_at(move.to_square)
    if victim is not None:
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return 10_000 + _PIECE_VAL[victim] - _PIECE_VAL.get(attacker, 0)
    if board.is_en_passant(move):
        return 10_000  # pawn x pawn
    return 0


def alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    deadline: float,
) -> float:
    if time.monotonic() >= deadline:
        raise _Timeout()

    key = cast(int, board._transposition_key())
    entry = _tt.probe(key)
    tt_move: chess.Move | None = entry.move if entry is not None else None

    if entry is not None and entry.depth >= depth:
        if entry.flag == Flag.EXACT:
            return entry.score
        if entry.flag == Flag.LOWER_BOUND:
            alpha = max(alpha, entry.score)
        elif entry.flag == Flag.UPPER_BOUND:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    moves = _get_legal_moves(board)
    if not moves:
        return -MATE if board.is_check() else 0.0

    if depth == 0:
        bbs, stm = encode(board)
        score = float(evaluate(bbs, stm, len(moves)))
        _tt.store(key, TTEntry(0, score, Flag.EXACT, None))
        return score

    moves.sort(key=lambda m: _move_score(board, m, tt_move), reverse=True)

    orig_alpha = alpha
    best_score = -math.inf
    best_move_at_node: chess.Move | None = None

    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, deadline)
        board.pop()
        if score >= beta:
            _tt.store(key, TTEntry(depth, score, Flag.LOWER_BOUND, move))
            return beta
        if score > best_score:
            best_score = score
            best_move_at_node = move
        if score > alpha:
            alpha = score

    flag = Flag.UPPER_BOUND if best_score <= orig_alpha else Flag.EXACT
    _tt.store(key, TTEntry(depth, best_score, flag, best_move_at_node))
    return alpha


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    budget_s = (time_left_ms / _MOVES_ESTIMATE) / 1000.0
    deadline = time.monotonic() + budget_s

    moves = _get_legal_moves(board)
    assert moves
    best_move = moves[0]

    for depth in range(1, _MAX_DEPTH + 1):
        # Re-sort root moves using TT move from the completed previous depth
        root_key = cast(int, board._transposition_key())
        root_entry = _tt.probe(root_key)
        root_tt_move = root_entry.move if root_entry is not None else None
        moves.sort(key=lambda m: _move_score(board, m, root_tt_move), reverse=True)

        alpha = -math.inf
        candidate: chess.Move | None = None
        try:
            for move in moves:
                board.push(move)
                score = -alphabeta(board, depth - 1, -MATE, -alpha, deadline)
                board.pop()
                if score > alpha:
                    alpha = score
                    candidate = move
        except _Timeout:
            break
        if candidate is not None:
            best_move = candidate
            _tt.store(root_key, TTEntry(depth, alpha, Flag.EXACT, best_move))
        if time.monotonic() >= deadline:
            break

    return best_move.uci()
