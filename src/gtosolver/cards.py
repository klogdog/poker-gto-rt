"""Canonical Hold'em cards and seven-card hand evaluation.

Card ids are rank-major: 2c=0, 2d=1, ..., As=51.  The public
evaluator returns a strength where a larger value wins.
"""

from collections.abc import Sequence
from functools import lru_cache
from numbers import Integral
import re

from treys import Card, Evaluator

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_CARD_RE = re.compile(r"^[2-9TJQKA][CDHS]$", re.IGNORECASE)
_EVALUATOR = Evaluator()
_TREYS_CARDS = tuple(Card.new(rank + suit) for rank in RANKS for suit in SUITS)


def normalize_card(card: str) -> str:
    """Return e.g. ``As``; reject malformed or non-string cards."""
    if not isinstance(card, str) or not _CARD_RE.fullmatch(card.strip()):
        raise ValueError(f"Invalid card {card!r}; use rank 2-9/T/J/Q/K/A and suit c/d/h/s")
    return card.strip()[0].upper() + card.strip()[1].lower()


def encode_card(card: str) -> int:
    card = normalize_card(card)
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def decode_card(card: int) -> str:
    if isinstance(card, bool) or not isinstance(card, Integral) or not 0 <= card < 52:
        raise ValueError(f"Invalid card id {card!r}; expected an integer from 0 to 51")
    return RANKS[card // 4] + SUITS[card % 4]


def normalize_board(
    board: Sequence[str], min_cards: int = 0, max_cards: int = 5
) -> list[str]:
    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence):
        raise ValueError("Board must be a sequence of card strings")
    if not min_cards <= len(board) <= max_cards:
        raise ValueError(f"Board must contain between {min_cards} and {max_cards} cards")
    normalized = [normalize_card(card) for card in board]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Board contains duplicate cards")
    return normalized


def canonical_combo(cards: Sequence[int]) -> str:
    """Higher rank first; equal ranks use c/d/h/s suit order."""
    if len(cards) != 2 or cards[0] == cards[1]:
        raise ValueError("A private hand must contain two different cards")
    for card in cards:
        decode_card(card)
    ordered = sorted(cards, key=lambda card: (-(card // 4), card % 4))
    return "".join(decode_card(card) for card in ordered)


@lru_cache(maxsize=200_000)
def _evaluate_cached(cards: tuple[int, ...]) -> int:
    value = _EVALUATOR.evaluate([], [_TREYS_CARDS[card] for card in cards])
    return 7463 - value


def evaluate_seven(cards: Sequence[int]) -> int:
    """Evaluate seven distinct encoded cards; larger strength is better."""
    if len(cards) != 7:
        raise ValueError("Seven-card evaluation requires exactly seven cards")
    if any(isinstance(card, bool) or not isinstance(card, Integral) for card in cards):
        raise ValueError("Card ids must be integers from 0 to 51")
    normalized = tuple(int(card) for card in cards)
    if any(card < 0 or card >= 52 for card in normalized):
        raise ValueError("Card ids must be integers from 0 to 51")
    if len(set(normalized)) != 7:
        raise ValueError("Seven-card evaluation contains duplicate cards")
    return _evaluate_cached(tuple(sorted(normalized)))
