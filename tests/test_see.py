"""Regression tests for SEE and quiescence promotion handling."""

import chess

from src.search import _PIECE_VAL, _get_captures, _see


def test_see_quiet_promotion() -> None:
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    move = chess.Move.from_uci("a7a8q")
    assert move in board.legal_moves
    expected = _PIECE_VAL[chess.QUEEN] - _PIECE_VAL[chess.PAWN]
    assert _see(board, move) == expected


def test_see_promotion_capture() -> None:
    # White pawn on b7 captures the rook on a8 and promotes to a queen.
    board = chess.Board("r6k/1P6/8/8/8/8/8/7K w - - 0 1")
    move = chess.Move.from_uci("b7a8q")
    assert move in board.legal_moves
    expected = _PIECE_VAL[chess.ROOK] + _PIECE_VAL[chess.QUEEN] - _PIECE_VAL[chess.PAWN]
    assert _see(board, move) == expected


def test_get_captures_includes_underpromotions() -> None:
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    captures = _get_captures(board)
    promos = {m.promotion for m in captures if m.from_square == chess.A7}
    assert promos == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
