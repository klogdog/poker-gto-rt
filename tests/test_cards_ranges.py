from itertools import combinations

import numpy as np
import pytest

from gtosolver.cards import canonical_combo, decode_card, encode_card, evaluate_seven, normalize_board
from gtosolver.payoffs import build_payoffs
from gtosolver.ranges import expand_range, sample_joint_hands


def test_card_roundtrip_normalization_and_board_uniqueness():
    assert encode_card("as") == 51
    assert encode_card(" 2C ") == 0
    assert [encode_card(decode_card(card)) for card in range(52)] == list(range(52))
    assert normalize_board(["AS", "2c", "Th"]) == ["As", "2c", "Th"]
    assert canonical_combo([encode_card("Kh"), encode_card("As")]) == "AsKh"
    assert canonical_combo([encode_card("As"), encode_card("Ah")]) == "AhAs"
    with pytest.raises(ValueError, match="duplicate"):
        normalize_board(["As", "aS"])
    for card in ("10s", "1h", "Ax", "AA", 7):
        with pytest.raises(ValueError):
            encode_card(card)


@pytest.mark.parametrize(
    ("spec", "count"),
    [("AA", 6), ("AKs", 4), ("AKo", 12), ("AK", 16), ("QQ+", 18),
     ("AJs+", 12), ("99-JJ", 18), ("A2s-A5s", 16), ("random", 1326), ("AsKh", 1)],
)
def test_range_expansion_counts(spec, count):
    assert len(expand_range(spec, [])) == count


def test_range_weights_overlap_and_blockers():
    hands = expand_range("aks:0.5,AK:0.2", ["As", "Kd", "2c"])
    assert len(hands) == 9
    assert all(not set(cards).intersection({encode_card("As"), encode_card("Kd"), encode_card("2c")}) for _, cards, _ in hands)
    assert {weight for _, _, weight in hands} == {0.2, 0.5}
    assert expand_range({"AKs": 0.5, "AK": 0.2}, ["As", "Kd", "2c"]) == hands
    assert expand_range("AK,AKs", []) == expand_range("AK", [])
    assert expand_range("AhAs", []) == expand_range("AsAh", [])
    assert len(expand_range("random", ["As", "Kh", "Qd"])) == 1176


@pytest.mark.parametrize("spec", ["", "wat", "AAo", "AA:0", "AA:-1", "AA:nan", "AA:inf", "AsAs", "AsKh", "AA-KQs", "AKs-AQo", "AKs:0.5:0.2"])
def test_ranges_reject_invalid_or_empty(spec):
    with pytest.raises(ValueError):
        expand_range(spec, ["As", "Kh", "2c"])


def test_evaluator_ordering_and_board_tie():
    royal = [encode_card(card) for card in ["As", "Ks", "Qs", "Js", "Ts", "2h", "3c"]]
    quads = [encode_card(card) for card in ["Ac", "Ad", "Ah", "As", "Kh", "2h", "3c"]]
    assert evaluate_seven(royal) > evaluate_seven(quads)
    board = ["As", "Ks", "Qs", "Js", "Ts"]
    assert evaluate_seven([encode_card(card) for card in board + ["2h", "3c"]]) == evaluate_seven([encode_card(card) for card in board + ["4h", "5c"]])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_seven(royal[:-1] + [royal[0]])
    with pytest.raises(ValueError, match="exactly seven"):
        evaluate_seven(royal[:-1])


def test_river_exact_payoffs_and_private_blockers():
    board = ["2c", "3d", "4h", "9s", "Jc"]
    oop = expand_range("AsAh", board)
    ip = expand_range("KsKh,AsKd", board)
    compatible, showdown, metadata = build_payoffs(board, oop, ip, 32, 7)
    assert compatible.tolist() == [[False, True]]
    assert showdown.tolist() == [[0.0, 1.0]]
    assert metadata["mode"] == "exact_river"
    assert metadata["min_valid_runouts"] == 1
    assert metadata["future_betting"] is False
    tie_board = ["As", "Ks", "Qs", "Js", "Ts"]
    _, tie, _ = build_payoffs(tie_board, expand_range("2h3c", tie_board), expand_range("4h5c", tie_board), 32, 7)
    assert tie.item() == 0.0


def test_turn_exact_matches_manual_conditioned_enumeration():
    board = ["2c", "3d", "4h", "9s"]
    oop = expand_range("AsAh", board)
    ip = expand_range("KsKh", board)
    compatible, showdown, metadata = build_payoffs(board, oop, ip, 32, 17)
    occupied = {encode_card(card) for card in board}.union(oop[0][1], ip[0][1])
    outcomes = [
        np.sign(evaluate_seven([encode_card(card) for card in board] + [river] + list(oop[0][1])) - evaluate_seven([encode_card(card) for card in board] + [river] + list(ip[0][1])))
        for river in range(52) if river not in occupied
    ]
    assert compatible.item()
    assert showdown.item() == pytest.approx(np.mean(outcomes))
    assert metadata["mode"] == "exact_turn"
    assert metadata["runouts"] == 48
    assert metadata["min_valid_runouts"] == 44


def test_flop_sampling_reproducible_and_matches_exact_reference():
    board = ["2c", "3d", "4h"]
    oop = expand_range("AsAh", board)
    ip = expand_range("KsKh", board)
    result = build_payoffs(board, oop, ip, 2048, 123)
    repeated = build_payoffs(board, oop, ip, 2048, 123)
    assert np.array_equal(result[1], repeated[1])
    assert result[2] == repeated[2]
    occupied = {encode_card(card) for card in board}.union(oop[0][1], ip[0][1])
    deck = [card for card in range(52) if card not in occupied]
    outcomes = [
        np.sign(evaluate_seven([encode_card(card) for card in board] + list(runout) + list(oop[0][1])) - evaluate_seven([encode_card(card) for card in board] + list(runout) + list(ip[0][1])))
        for runout in combinations(deck, 2)
    ]
    assert result[1].item() == pytest.approx(np.mean(outcomes), abs=0.06)
    assert result[2]["mode"] == "sampled_flop"
    assert 0 < result[2]["min_valid_runouts"] <= 2048


def test_joint_sampler_legal_deterministic_and_infeasible():
    board = ["2c", "3d", "4h"]
    ranges = [expand_range("AA", board), expand_range("KK", board), expand_range("QQ", board)]
    first = sample_joint_hands(ranges, board, np.random.default_rng(7))
    second = sample_joint_hands(ranges, board, np.random.default_rng(7))
    assert first == second
    assert len({card for hand in first for card in hand}) == 6
    with pytest.raises(ValueError, match="too few distinct"):
        sample_joint_hands([expand_range("AsAh", board)] * 2, board, np.random.default_rng(7))
    # Plenty of distinct cards in the union, but no legal choice for seats 0/1.
    impossible = [expand_range("AsAh", board), expand_range("AsKh,AhKd", board), expand_range("QcQd", board)]
    with pytest.raises(ValueError, match="cannot form any legal"):
        sample_joint_hands(impossible, board, np.random.default_rng(7), max_attempts=8)


def test_joint_sampler_preserves_conditioned_weighted_prior():
    # Only legal pairs are (AsAh,KcKd), (KsKh,AsAd), (KsKh,KcKd).
    # Prior masses 2*1, 1*3, 1*1 => probabilities 1/3, 1/2, 1/6.
    board = ["2c", "3d", "4h"]
    ranges = [expand_range({"AsAh": 2, "KsKh": 1}, board), expand_range({"AsAd": 3, "KcKd": 1}, board)]
    rng = np.random.default_rng(46)
    counts = {}
    for _ in range(4000):
        deal = sample_joint_hands(ranges, board, rng)
        key = tuple(canonical_combo(hand) for hand in deal)
        counts[key] = counts.get(key, 0) + 1
    assert counts[("AhAs", "KcKd")] / 4000 == pytest.approx(1 / 3, abs=0.035)
    assert counts[("KhKs", "AdAs")] / 4000 == pytest.approx(1 / 2, abs=0.035)
    assert counts[("KhKs", "KcKd")] / 4000 == pytest.approx(1 / 6, abs=0.035)
