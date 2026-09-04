"""Negamax search with alpha-beta pruning and iterative deepening."""

import math
import time

import chess

from .bitboard import encode, evaluate
from .movegen import generate_pseudo_legal

MATE = 1e6
_MAX_DEPTH = 10
_MOVES_ESTIMATE = 60  # expected moves remaining — used to divide the time budget


class _Timeout(Exception):
    pass


def _get_legal_moves(board: chess.Board) -> list[chess.Move]:
    bbs, stm = encode(board)
    ep_sq = board.ep_square if board.ep_square is not None else -1
    candidates = generate_pseudo_legal(bbs, stm, ep_sq, board.castling_rights)
    return [m for m in candidates if board.is_legal(m)]


def alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    deadline: float,
) -> float:
    if time.monotonic() >= deadline:
        raise _Timeout()
    moves = _get_legal_moves(board)
    if not moves:
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        bbs, stm = encode(board)
        return float(evaluate(bbs, stm, len(moves)))
    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, deadline)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    budget_s = (time_left_ms / _MOVES_ESTIMATE) / 1000.0
    deadline = time.monotonic() + budget_s

    moves = _get_legal_moves(board)
    assert moves
    best_move = moves[0]  # guaranteed fallback after depth-1 pass

    for depth in range(1, _MAX_DEPTH + 1):
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
            break  # abandon this depth; use previous complete result
        if candidate is not None:
            best_move = candidate  # commit only fully-searched depths
        if time.monotonic() >= deadline:
            break

    return best_move.uci()
