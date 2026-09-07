"""Regression tests for checkmate-vs-draw precedence, repetition threshold, and futility."""

import chess

from src.search import MATE, alphabeta


def test_checkmate_beats_fifty_move_draw() -> None:
    # White is checkmated (Fool's mate); force halfmove_clock to the 50-move threshold.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
    board.halfmove_clock = 100
    result = alphabeta(board, 1, -float(MATE), float(MATE), deadline=float("inf"))
    assert result.score < -(MATE // 2)


def _shuffle_kings() -> chess.Board:
    return chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")


def test_two_occurrences_does_not_force_draw() -> None:
    board = _shuffle_kings()
    board.push_uci("e1d1")
    board.push_uci("e5d5")
    board.push_uci("d1e1")
    board.push_uci("d5e5")
    assert not board.is_repetition(3)
    result = alphabeta(board, 1, -float(MATE), float(MATE), deadline=float("inf"))
    assert not (result.score == 0.0 and result.tainted)


def test_three_occurrences_forces_tainted_draw() -> None:
    board = _shuffle_kings()
    for _ in range(2):
        board.push_uci("e1d1")
        board.push_uci("e5d5")
        board.push_uci("d1e1")
        board.push_uci("d5e5")
    assert board.is_repetition(3)
    result = alphabeta(board, 1, -float(MATE), float(MATE), deadline=float("inf"))
    assert result.score == 0.0
    assert result.tainted


def test_futility_does_not_prune_checking_move() -> None:
    # King+Queen vs King: Qh1-b7# is a quiet (non-capturing) mate-in-one. No captures exist
    # anywhere on the board, so every legal move is "quiet" — a broken futility check would
    # prune all of them, including the mate, and the node would fail to improve past alpha.
    board = chess.Board("k7/8/1K6/8/8/8/8/7Q w - - 0 1")
    alpha, beta = 100_000.0, 200_000.0
    result = alphabeta(board, 1, alpha, beta, deadline=float("inf"))
    assert result.score == beta
