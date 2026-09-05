"""Dynamic time allocation per move."""

import time

import chess

_OVERHEAD_MS = 50


class TimeManager:
    __slots__ = ("hard_deadline", "soft_deadline")

    def __init__(self, time_left_ms: int, board: chess.Board) -> None:
        now = time.monotonic()
        safe_ms = float(max(100, time_left_ms - _OVERHEAD_MS))
        # estimate remaining moves this side has to play
        moves_to_go = max(8, 35 - board.ply() // 2)
        base_ms = safe_ms / moves_to_go
        # soft: stop ID after completing a depth if past this
        # hard: abort mid-search at this point
        soft_ms = base_ms * 0.7
        hard_ms = min(base_ms * 2.0, safe_ms * 0.2)
        self.soft_deadline: float = now + soft_ms / 1000.0
        self.hard_deadline: float = now + hard_ms / 1000.0

    def should_stop(self) -> bool:
        return time.monotonic() >= self.soft_deadline
