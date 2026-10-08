"""Validated, path-scoped exact chip sizes for the finite betting abstraction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


MAX_HISTORY = 32
MAX_OVERRIDES = 32


def validate_history(history: Sequence[str]) -> tuple[str, ...]:
    if (isinstance(history, (str, bytes)) or not isinstance(history, Sequence)
            or len(history) > MAX_HISTORY
            or any(not isinstance(label, str) or not label or len(label) > 64 for label in history)):
        raise ValueError("history must contain at most 32 action labels of 1 to 64 characters")
    return tuple(history)


def validate_min_bet(min_bet: float) -> float:
    if isinstance(min_bet, bool) or not isinstance(min_bet, (int, float)) or not math.isfinite(min_bet) or not 0 <= min_bet <= 1e9:
        raise ValueError("min_bet must be a finite chip amount between 0 and 1e9")
    return float(min_bet)


class ActionOverrides:
    """At most one additional target per public history; no size quantization."""

    def __init__(self, overrides: Sequence[dict] | None = None):
        overrides = () if overrides is None else overrides
        if (isinstance(overrides, (str, bytes)) or not isinstance(overrides, Sequence)
                or len(overrides) > MAX_OVERRIDES):
            raise ValueError("action_overrides must contain at most 32 entries")
        self.targets: dict[tuple[str, ...], float] = {}
        self.used: set[tuple[str, ...]] = set()
        for override in overrides:
            if not isinstance(override, Mapping) or set(override) != {"history", "to"}:
                raise ValueError("Each action override must contain only history and to")
            history = validate_history(override["history"])
            if len(history) >= MAX_HISTORY:
                raise ValueError("An action override must leave room for its action within the 32-label history limit")
            if history in self.targets:
                raise ValueError("Duplicate or ambiguous action overrides at the same history")
            target = override["to"]
            if (isinstance(target, bool) or not isinstance(target, (int, float))
                    or not math.isfinite(target) or not 0 < target <= 1e9):
                raise ValueError("An action override target must be finite, positive, and at most 1e9 chips")
            self.targets[history] = float(target)

    def target(self, history: Sequence[str]) -> float | None:
        return self.targets.get(tuple(history))

    def mark_used(self, history: Sequence[str]) -> None:
        self.used.add(tuple(history))

    def assert_used(self) -> None:
        unused = set(self.targets) - self.used
        if unused:
            first = min(unused, key=lambda path: (len(path), path))
            raise ValueError(f"Unused or unreachable action override history: {list(first)}")


def validate_target(target: float, baseline: float, stack: float, minimum: float) -> None:
    if target <= baseline or target > stack:
        raise ValueError("Action override target must exceed the current bet and not exceed effective_stack")
    if target < stack and target < baseline + minimum:
        raise ValueError("Action override violates the minimum opening bet or minimum full raise")


def exact_label(facing_bet: bool, target: float) -> str:
    return f"{'raise' if facing_bet else 'bet'}_to_{target:.17g}"
