"""Chip ledgers and terminal side-pot payouts, independent of the backend."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _amounts(values: Sequence[float], count: int | None, name: str) -> tuple[float, ...]:
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            or (count is not None and len(values) != count)):
        raise ValueError(f"{name} must contain one amount per player")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1e9 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative chip amounts at most 1e9")
    return tuple(float(value) for value in values)


def _external_money(pot: float, recorded: float) -> float:
    external = pot - recorded
    if external < 0:
        # Bounded addition-order noise must scale with the actual chip units;
        # a fixed absolute epsilon would permit entire pots at tiny scales.
        tolerance = 64 * max(math.ulp(pot), math.ulp(recorded))
        if -external > tolerance:
            raise ValueError("pot must include all prior committed and dead contributions")
        return 0.0
    return external


def normalize_ledger(players: int, pot: float, stack: float,
                     stacks: Sequence[float] | None = None,
                     committed: Sequence[float] | None = None,
                     dead_contributions: Sequence[float] | None = None) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    stacks = _amounts([stack] * players if stacks is None else stacks, players, "stacks")
    prior = _amounts([0.0] * players if committed is None else committed, players, "committed")
    dead = _amounts([] if dead_contributions is None else dead_contributions, None, "dead_contributions")
    if len(dead) > 32:
        raise ValueError("dead_contributions must contain at most 32 amounts")
    recorded = math.fsum((*prior, *dead))
    _external_money(pot, recorded)
    return stacks, prior, dead


def settlement_payouts(pot: float, prior_committed: Sequence[float],
                       current_committed: Sequence[float], folded: Sequence[bool],
                       dead_contributions: Sequence[float], strengths: Sequence[float]) -> np.ndarray:
    """Award cumulative contribution layers and refund unmatched investments.

    ``pot`` contains prior commitments/dead contributions and optional external
    money. Larger strengths win. Returned payouts include previously invested
    chips; subtract only current-round contributions to obtain round utility.
    """
    players = len(prior_committed)
    prior = _amounts(prior_committed, players, "committed")
    current = _amounts(current_committed, players, "current_committed")
    dead = _amounts(dead_contributions, None, "dead_contributions")
    if len(folded) != players or len(strengths) != players:
        raise ValueError("folded and strengths must contain one value per player")
    active = [index for index in range(players) if not folded[index]]
    if not active:
        raise ValueError("At least one player must remain eligible for the pot")
    totals = tuple(prior[index] + current[index] for index in range(players))
    contributions = (*totals, *dead)
    recorded = math.fsum((*prior, *dead))
    external = _external_money(pot, recorded)
    payouts = np.zeros(players, dtype=np.float64)

    def award(amount: float, eligible: Sequence[int]) -> None:
        best = max(strengths[index] for index in eligible)
        winners = [index for index in eligible if strengths[index] == best]
        share = amount / len(winners)
        for index in winners:
            payouts[index] += share

    if external:
        award(external, active)
    previous = 0.0
    for level in sorted(set(value for value in contributions if value > 0)):
        contributors = [index for index, value in enumerate(contributions) if value >= level]
        amount = (level - previous) * len(contributors)
        previous = level
        if len(contributors) == 1 and contributors[0] < players:
            # Uncalled chips have no opponent and are returned independently of
            # hand rank. This also preserves the fold branch's sunk-chip cost.
            payouts[contributors[0]] += amount
            continue
        eligible = [index for index in active if totals[index] >= level]
        # A layer supplied solely by already folded players is dead money. In
        # reachable betting states at least one live matching player exists;
        # the fallback keeps explicit external ledgers chip-conserving too.
        award(amount, eligible or active)
    return payouts
