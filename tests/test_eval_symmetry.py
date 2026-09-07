"""Eval must be mirror-symmetric: evaluate(pos) == evaluate(mirror(pos))."""

import chess

from src.bitboard import encode, evaluate

_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pp2ppbp/2np1np1/2p5/2P5/2N1PN2/PP1PBPPP/R1BQK2R w KQkq - 0 1",
    "r1bqk2r/pp2ppbp/2np1np1/2p5/2P5/2N1PN2/PP1PBPPP/R1BQK2R b KQkq - 0 1",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "8/8/8/4k3/8/4K3/4P3/8 b - - 0 1",
    "r3k2r/pppq1ppp/2n1bn2/1B2p3/1b2P3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",
    "2kr3r/ppp2ppp/2n1b3/2b1p3/4P3/2N1BN2/PPP2PPP/2KR3R w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1",
    "r1b1kbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqk1nr/pppp1ppp/8/2b1p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 4 3",
]


def test_eval_mirror_symmetry() -> None:
    for fen in _FENS:
        board = chess.Board(fen)
        mirrored = board.mirror()
        score = evaluate(*encode(board))
        mirrored_score = evaluate(*encode(mirrored))
        assert score == mirrored_score, f"{fen}: {score} != {mirrored_score} (mirrored)"


def test_eval_mirror_symmetry_regression_case() -> None:
    """Exact case that exposed the original stm-vs-color PST mirroring bug (was -6 vs 24)."""
    fen = "r1bqk2r/pp2ppbp/2np1np1/2p5/2P5/2N1PN2/PP1PBPPP/R1BQK2R w KQkq - 0 1"
    board = chess.Board(fen)
    score = evaluate(*encode(board))
    mirrored_score = evaluate(*encode(board.mirror()))
    assert score == mirrored_score
