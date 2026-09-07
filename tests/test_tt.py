"""Regression tests for the bounded transposition table."""

import chess

import src.tt as tt
from src.tt import Flag, TTEntry


def setup_function() -> None:
    tt.clear()


def teardown_function() -> None:
    tt.clear()


def test_tt_bounded_size() -> None:
    before = len(tt._keys)
    for i in range(before * 4):
        tt.store(i, TTEntry(depth=1, score=0.0, flag=Flag.EXACT, move=None))
    assert len(tt._keys) == before


def test_tt_depth_preferred_replacement() -> None:
    key = 12345
    deep_move = chess.Move.from_uci("e2e4")
    shallow_move = chess.Move.from_uci("d2d4")
    tt.store(key, TTEntry(depth=5, score=10.0, flag=Flag.EXACT, move=deep_move))
    tt.store(key, TTEntry(depth=2, score=20.0, flag=Flag.EXACT, move=shallow_move))
    entry = tt.probe(key)
    assert entry is not None
    assert entry.depth == 5
    assert entry.move == deep_move


def test_tt_probe_miss_returns_none() -> None:
    assert tt.probe(999_999_999) is None
