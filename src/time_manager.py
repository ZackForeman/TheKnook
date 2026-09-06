"""Dynamic time allocation per move."""

import time

import chess

_OVERHEAD_MS = 50
_INCREMENT_MS = 500  # platform gives 0.5 s/move (see CLAUDE.md)


class TimeManager:
    __slots__ = ("hard_deadline", "soft_deadline")

    def __init__(self, time_left_ms: int, board: chess.Board) -> None:
        now = time.monotonic()
        safe_ms = float(max(100, time_left_ms - _OVERHEAD_MS))
        moves_to_go = max(8, 40 - board.ply() // 2)
        # factor in future increments so early moves aren't under-budgeted
        effective_budget = safe_ms + _INCREMENT_MS * moves_to_go
        base_ms = effective_budget / moves_to_go
        # soft: stop ID after completing a depth if past this
        # hard: capped at 1.5x base and 15% of actual clock
        soft_ms = base_ms * 0.75
        hard_ms = min(base_ms * 2.0, safe_ms * 0.15)
        self.soft_deadline: float = now + soft_ms / 1000.0
        self.hard_deadline: float = now + hard_ms / 1000.0

    def should_stop(self) -> bool:
        return time.monotonic() >= self.soft_deadline

    def extend(self, factor: float) -> None:
        """Stretch the remaining soft budget by factor (called on score instability)."""
        now = time.monotonic()
        remaining = self.soft_deadline - now
        if remaining > 0:
            self.soft_deadline = now + remaining * factor
