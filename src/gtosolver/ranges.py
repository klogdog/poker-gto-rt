"""Expand weighted Hold'em ranges into deterministic physical combinations.

When several tokens cover one combination, its weight is the maximum of the
token weights.  Thus ``AK,AKs`` is identical to ``AK`` rather than overweighting
the suited combinations.  Weights are positive finite relative frequencies.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations
import math
from numbers import Integral
import re

import numpy as np

from .cards import RANKS, canonical_combo, encode_card, normalize_board

ExpandedHand = tuple[str, tuple[int, int], float]
ExpandedRange = list[ExpandedHand]
_CLASS_RE = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)$", re.IGNORECASE)
_EXACT_RE = re.compile(r"^[2-9TJQKA][CDHS][2-9TJQKA][CDHS]$", re.IGNORECASE)


def _weight(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Range weights must be positive finite numbers")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Range weights must be positive finite numbers") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Range weights must be positive finite numbers")
    return result


def _parse_class(token: str) -> tuple[int, int, str]:
    match = _CLASS_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"Unknown range token {token!r}")
    rank1, rank2, suffix = match.groups()
    first, second = RANKS.index(rank1.upper()), RANKS.index(rank2.upper())
    if first < second:
        first, second = second, first
    suffix = suffix.lower()
    if first == second and suffix:
        raise ValueError(f"Pairs cannot have a suited/offsuit suffix: {token!r}")
    return first, second, suffix


def _class_combos(first: int, second: int, suffix: str) -> list[tuple[int, int]]:
    if first == second:
        return [(first * 4 + a, first * 4 + b) for a, b in combinations(range(4), 2)]
    return [
        (first * 4 + a, second * 4 + b)
        for a in range(4)
        for b in range(4)
        if not suffix or (suffix == "s" and a == b) or (suffix == "o" and a != b)
    ]


def _expand_token(token: str) -> list[tuple[int, int]]:
    if token.lower() in {"random", "*"}:
        return list(combinations(range(52), 2))
    if _EXACT_RE.fullmatch(token):
        cards = encode_card(token[:2]), encode_card(token[2:])
        if cards[0] == cards[1]:
            raise ValueError(f"Private hand contains duplicate cards: {token!r}")
        return [cards]
    if token.endswith("+"):
        first, second, suffix = _parse_class(token[:-1])
        classes = (
            [(rank, rank, "") for rank in range(first, 13)]
            if first == second
            else [(first, rank, suffix) for rank in range(second, first)]
        )
        return [combo for hand_class in classes for combo in _class_combos(*hand_class)]
    if "-" in token:
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Malformed range interval {token!r}")
        start, end = _parse_class(parts[0]), _parse_class(parts[1])
        if start[0] == start[1] and end[0] == end[1]:
            low, high = sorted((start[0], end[0]))
            classes = [(rank, rank, "") for rank in range(low, high + 1)]
        elif start[0] == end[0] and start[2] == end[2] and start[0] != start[1] and end[0] != end[1]:
            low, high = sorted((start[1], end[1]))
            classes = [(start[0], rank, start[2]) for rank in range(low, high + 1)]
        else:
            raise ValueError(f"Range interval must connect pairs or same-high-card classes: {token!r}")
        return [combo for hand_class in classes for combo in _class_combos(*hand_class)]
    return _class_combos(*_parse_class(token))


def expand_range(spec: str | Mapping[str, float], board: Sequence[str]) -> ExpandedRange:
    """Expand exact combos, hand classes, +/dash intervals, or random.

    Strings use comma/whitespace-separated tokens with optional ``:weight``.
    Dictionaries map range tokens to weights.  Board blockers are removed;
    an empty resulting range is an error.
    """
    board_ids = {encode_card(card) for card in normalize_board(board)}
    if isinstance(spec, str):
        tokens = [token for token in re.split(r"[\s,]+", spec.strip()) if token]
        weighted: list[tuple[str, float]] = []
        for token in tokens:
            parts = token.split(":")
            if len(parts) > 2 or not parts[0]:
                raise ValueError(f"Malformed weighted range token {token!r}")
            weighted.append((parts[0], _weight(parts[1]) if len(parts) == 2 else 1.0))
    elif isinstance(spec, Mapping):
        weighted = []
        for token, value in spec.items():
            if not isinstance(token, str) or not token or re.search(r"[\s,:]", token):
                raise ValueError("Range dictionary keys must each contain one unweighted range token")
            weighted.append((token, _weight(value)))
    else:
        raise ValueError("Range must be a string or a dictionary of token weights")
    result: dict[str, ExpandedHand] = {}
    for token, weight in weighted:
        for cards in _expand_token(token):
            if board_ids.intersection(cards):
                continue
            combo = canonical_combo(cards)
            ordered = encode_card(combo[:2]), encode_card(combo[2:])
            if combo not in result or weight > result[combo][2]:
                result[combo] = (combo, ordered, weight)
    if not result:
        raise ValueError("Range has no legal combinations after applying board blockers")
    return sorted(
        result.values(),
        key=lambda hand: (-(hand[1][0] // 4), hand[1][0] % 4, -(hand[1][1] // 4), hand[1][1] % 4),
    )


def _joint_feasible(ranges: Sequence[ExpandedRange], blocked: set[int], limit: int = 100_000) -> bool | None:
    """Bounded search only diagnoses rejection failures; never supplies samples."""
    ordered = sorted(ranges, key=len)
    visited = 0

    def visit(position: int, used: set[int]) -> bool | None:
        nonlocal visited
        if position == len(ordered):
            return True
        for _, cards, _ in ordered[position]:
            visited += 1
            if visited > limit:
                return None
            if not used.intersection(cards):
                result = visit(position + 1, used.union(cards))
                if result is not False:
                    return result
        return False

    return visit(0, blocked)


def sample_joint_hands(
    ranges: Sequence[ExpandedRange],
    board: Sequence[str | int],
    rng: np.random.Generator,
    *,
    max_attempts: int = 10_000,
) -> list[tuple[int, int]]:
    """Sample weighted independent ranges conditioned on joint card legality.

    Rejection sampling preserves the joint prior even for overlapping ranges.
    Sequentially sampling each seat from the remaining deck would not.
    """
    if not ranges or any(not hand_range for hand_range in ranges):
        raise ValueError("Every player must have a nonempty range")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    blocked: set[int] = set()
    for card in board:
        if not isinstance(card, str) and (isinstance(card, bool) or not isinstance(card, Integral)):
            raise ValueError("Board card ids must be integers from 0 to 51")
        card_id = encode_card(card) if isinstance(card, str) else int(card)
        if not 0 <= card_id < 52 or card_id in blocked:
            raise ValueError("Board must contain distinct legal cards")
        blocked.add(card_id)
    legal_ranges = [[hand for hand in hand_range if not blocked.intersection(hand[1])] for hand_range in ranges]
    if any(not hand_range for hand_range in legal_ranges):
        raise ValueError("A player has no legal hands after applying board blockers")
    union = {card for hand_range in legal_ranges for _, cards, _ in hand_range for card in cards}
    if len(union) < 2 * len(ranges):
        raise ValueError("Player ranges cannot form a legal joint deal: too few distinct available cards")
    probabilities = []
    for hand_range in legal_ranges:
        weights = np.asarray([_weight(hand[2]) for hand in hand_range], dtype=np.float64)
        # Scaling avoids overflow when finite but large relative weights are supplied.
        weights /= weights.max()
        probabilities.append(weights / weights.sum())
    for _ in range(max_attempts):
        sampled = [hand_range[int(rng.choice(len(hand_range), p=probability))][1] for hand_range, probability in zip(legal_ranges, probabilities)]
        flattened = [card for cards in sampled for card in cards]
        if len(set(flattened)) == len(flattened):
            return sampled
    feasible = _joint_feasible(legal_ranges, blocked)
    if feasible is False:
        raise ValueError("Player ranges cannot form any legal joint deal")
    raise ValueError(
        f"Could not sample a legal joint deal in {max_attempts} attempts; "
        "ranges overlap too heavily. Broaden the ranges or increase max_attempts."
    )
