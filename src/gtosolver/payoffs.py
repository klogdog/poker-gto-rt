"""Showdown equity matrices conditional on both players' card blockers.

These utilities model showdown continuation, without future betting rounds.
River and turn are exact; flop uses seeded common random runouts.
"""

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .cards import encode_card, evaluate_seven, normalize_board
from .ranges import ExpandedRange


def _hands_array(hands: ExpandedRange, board: set[int]) -> np.ndarray:
    if not hands:
        raise ValueError("Showdown payoff calculation needs nonempty player ranges")
    raw_cards = [hand[1] for hand in hands]
    if any(len(cards) != 2 for cards in raw_cards):
        raise ValueError("Private hands must contain two encoded cards")
    if any(isinstance(card, bool) or not isinstance(card, Integral) or not 0 <= card < 52 for cards in raw_cards for card in cards):
        raise ValueError("Private hands must contain two different legal cards")
    result = np.asarray(raw_cards, dtype=np.int16)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("Private hands must contain two encoded cards")
    if np.any(result < 0) or np.any(result >= 52) or np.any(result[:, 0] == result[:, 1]):
        raise ValueError("Private hands must contain two different legal cards")
    if any(board.intersection(hand) for hand in result):
        raise ValueError("Private hands overlap the board; expand ranges using this board first")
    return result


def build_payoffs(
    board: Sequence[str],
    oop_hands: ExpandedRange,
    ip_hands: ExpandedRange,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return legal-pair mask and OOP expected win-minus-loss matrix.

    Illegal hand pairs have zero payoff and must receive zero reach weight.
    On flop, each pair conditions the common samples on its private cards.
    If any legal pair receives no valid runouts, increase ``samples``.
    """
    board_cards = [encode_card(card) for card in normalize_board(board, 3, 5)]
    blocked = set(board_cards)
    oop = _hands_array(oop_hands, blocked)
    ip = _hands_array(ip_hands, blocked)
    compatibility = np.all(oop[:, None, :, None] != ip[None, :, None, :], axis=(2, 3))
    if not np.any(compatibility):
        raise ValueError("The two ranges cannot form a legal joint deal")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("samples must be a positive integer")
    deck = np.asarray([card for card in range(52) if card not in blocked], dtype=np.int16)
    if len(board_cards) == 5:
        runouts = np.empty((1, 0), dtype=np.int16)
        mode = "exact_river"
    elif len(board_cards) == 4:
        runouts = deck[:, None]
        mode = "exact_turn"
    else:
        rng = np.random.default_rng(seed)
        runouts = np.asarray([rng.choice(deck, size=2, replace=False) for _ in range(samples)], dtype=np.int16)
        mode = "sampled_flop"
    totals = np.zeros(compatibility.shape, dtype=np.float64)
    counts = np.zeros(compatibility.shape, dtype=np.int32)
    for runout in runouts:
        oop_valid = ~np.isin(oop, runout).any(axis=1)
        ip_valid = ~np.isin(ip, runout).any(axis=1)
        complete_board = board_cards + runout.tolist()
        oop_strength = np.zeros(len(oop), dtype=np.int32)
        ip_strength = np.zeros(len(ip), dtype=np.int32)
        for index in np.flatnonzero(oop_valid):
            oop_strength[index] = evaluate_seven(complete_board + oop[index].tolist())
        for index in np.flatnonzero(ip_valid):
            ip_strength[index] = evaluate_seven(complete_board + ip[index].tolist())
        legal = compatibility & oop_valid[:, None] & ip_valid[None, :]
        totals += np.sign(oop_strength[:, None] - ip_strength[None, :]) * legal
        counts += legal
    if np.any(compatibility & (counts == 0)):
        raise ValueError(
            "Some legal hand pairs received no valid sampled runouts after blocker filtering; "
            "increase samples or choose a different seed"
        )
    showdown = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0).astype(np.float32)
    metadata = {
        "mode": mode,
        "exact": len(board_cards) >= 4,
        "runouts": len(runouts),
        "requested_samples": samples,
        "seed": seed,
        "min_valid_runouts": int(counts[compatibility].min()),
        "max_valid_runouts": int(counts[compatibility].max()),
        "legal_hand_pairs": int(compatibility.sum()),
        "blocker_conditioning": "joint_private_cards",
        "future_betting": False,
    }
    return compatibility, showdown, metadata
