"""Negamax search."""

import math
import random

import chess

from .bitboard import encode, evaluate

MATE = 1e6


def negamax(board: chess.Board, depth: int) -> float:
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        bbs, stm = encode(board)
        return float(evaluate(bbs, stm, len(moves)))
    best = -math.inf
    for move in moves:
        board.push(move)
        best = max(best, -negamax(board, depth - 1))
        board.pop()
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    best_score = -math.inf
    best: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, 1)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
    return random.choice(best).uci()
