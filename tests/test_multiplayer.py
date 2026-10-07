"""Multiplayer betting, regret-learning, conditioning, and device checks."""

import math
import platform

import numpy as np
import pytest

from gtosolver.backend import Backend
from gtosolver.cards import encode_card
from gtosolver.multiplayer import (
    BettingRound, MultiPlayerError, _Tables, _Trainer, _evaluate_policy, solve,
)
from gtosolver.ranges import expand_range


BOARD = ["2c", "3d", "7h", "9s", "Jc"]
PLAYERS = [{"name": "Kings", "range": "KsKh"},
           {"name": "Aces", "range": "AsAh"},
           {"name": "Queens", "range": "QsQh"}]


def choose(game, state, label):
    return game.advance(state, next(action for action in game.actions(state)
                                   if action.label == label))


@pytest.mark.parametrize("players", range(3, 7))
def test_every_player_must_check_before_showdown(players):
    game = BettingRound(players, 12, 100, [0.5], 1)
    state = game.initial
    for actor in range(players):
        assert state.actor == actor
        state = choose(game, state, "check")
        assert state.terminal == (actor == players - 1)
    values = game.payoff(state, [7] * players)
    assert values == pytest.approx(np.full(players, 12 / players))


def test_late_bet_reopens_checked_players_in_order_and_call_closes():
    game = BettingRound(4, 20, 100, [0.5], 1)
    state = game.query(["check", "check", "bet_50%"])
    assert state.pending == (3, 0, 1)
    assert state.contributions == (0, 0, 10, 0)
    state = choose(game, state, "call")
    state = choose(game, state, "fold")
    assert not state.terminal
    state = choose(game, state, "call")
    assert state.terminal
    assert state.contributions == (0, 10, 10, 10)
    assert game.payoff(state, [100, 1, 4, 4]) == pytest.approx([0, -10, 15, 15])


def test_raise_uses_pot_after_call_and_requires_previous_increment():
    game = BettingRound(3, 100, 1000, [0.1, 0.5, 1.0], 2)
    state = game.query(["bet_100%"])
    actions = {action.label: action for action in game.actions(state)}
    assert "raise_10%" not in actions  # Increment30 < initial bet100.
    assert actions["raise_50%"].to == 250  # 100 + .5*(pot200 + call100).
    state = choose(game, state, "raise_50%")
    assert state.pending == (2, 0)
    state = choose(game, state, "call")
    state = choose(game, state, "call")
    assert state.terminal
    assert state.contributions == (250, 250, 250)


def test_raise_cap_keeps_fold_and_call_legal():
    game = BettingRound(3, 10, 100, [0.5], 0)
    state = game.query(["bet_50%"])
    assert [action.label for action in game.actions(state)] == ["fold", "call"]


def test_equal_stack_all_in_has_no_side_pot_or_further_raise():
    game = BettingRound(4, 20, 12, [0.5], 2)
    state = game.query(["bet_50%", "all_in"])
    # The increase from10 to12 is a short raise. Shared ceiling means everyone
    # can only fold or match12, so reopening does not permit a further raise.
    assert state.pending == (2, 3, 0)
    assert [action.label for action in game.actions(state)] == ["fold", "call"]
    state = choose(game, state, "call")
    state = choose(game, state, "fold")
    state = choose(game, state, "call")
    assert state.terminal
    assert state.contributions == (12, 12, 12, 0)
    assert game.payoff(state, [3, 2, 1, 100]) == pytest.approx([44, -12, -12, 0])


def test_fold_to_one_player_awards_pot_without_requiring_checks():
    game = BettingRound(3, 10, 100, [0.5], 1)
    state = game.query(["bet_50%", "fold", "fold"])
    assert state.terminal
    assert game.payoff(state, [1, 10, 100]) == pytest.approx([10, 0, 0])


@pytest.mark.parametrize("pot,stack", [(1, 1e9), (1e-12, 1e-10), (1e-80, 1e-78)])
def test_small_opening_bets_survive_large_stacks_and_small_chip_units(pot, stack):
    game = BettingRound(3, pot, stack, [0.5], 1)
    actions = {action.label: action for action in game.actions(game.initial)}
    assert actions["bet_50%"].to == pot / 2
    assert actions["all_in"].to == stack
    state = game.advance(game.initial, actions["bet_50%"])
    assert [action.label for action in game.actions(state)][:2] == ["fold", "call"]


def test_nearby_sizes_have_distinct_history_labels():
    game = BettingRound(3, 10, 100, [0.50000001, 0.50000002], 1)
    actions = game.actions(game.initial)
    assert len({action.label for action in actions}) == len(actions)


def test_invalid_and_overlong_history_is_rejected():
    game = BettingRound(3, 20, 40, [0.5], 1)
    with pytest.raises(MultiPlayerError, match="legal actions"):
        game.query(["call"])
    with pytest.raises(MultiPlayerError, match="legal actions"):
        game.query(["check", "check", "check", "check"])


def test_mccfr_learns_to_avoid_a_dominated_bet_with_known_inferior_hand():
    result = solve(board=BOARD, players=PLAYERS, pot=20, effective_stack=40,
                   iterations=500, samples=32, backend="cpu", history=["bet_50%"])
    assert result["root"]["strategy"]["check"] > 0.98
    assert result["query"]["strategy"]["fold"] < 0.01  # The aces do not fold.
    assert result["query"]["player"] == 1
    assert sum(row["expected_value"] for row in result["ev"]["per_player"]) == pytest.approx(20)
    assert result["diagnostics"]["traversals"] == 1500
    assert "MCCFR" in result["algorithm"]
    assert result["query"]["history_conditioning"]["effective_sample_size"] == pytest.approx(32)


def test_root_policy_uses_compatible_joint_deal_marginal_and_excludes_impossible_hand():
    players = [{"name": "Hero", "range": {"AsAh": 10, "KsKh": 1}},
               {"name": "Other aces", "range": "AsAd"},
               {"name": "Other queens", "range": "QcQd"}]
    result = solve(board=BOARD, players=players, pot=20, effective_stack=40,
                   iterations=100, samples=32, backend="cpu", include_tree=True)
    feasible = next(row for row in result["root"]["hand_strategies"] if "K" in row["hand"])
    impossible = next(row for row in result["root"]["hand_strategies"] if "A" in row["hand"])
    assert result["root"]["strategy"] == pytest.approx(feasible["strategy"])
    assert feasible["posterior_probability"] == 1
    assert impossible["posterior_probability"] == 0
    assert not impossible["trained"]
    assert all("recommendation" not in node and "strategy" not in node for node in result["tree"])


def test_original_large_weights_do_not_overflow_reporting():
    players = [{"name": "Hero", "range": {"AsAh": 1e308, "KsKh": 1e308}},
               {"name": "Queens", "range": "QsQh"},
               {"name": "Tens", "range": "TsTh"}]
    result = solve(board=BOARD, players=players, pot=20, effective_stack=40,
                   iterations=20, samples=32, backend="cpu")
    assert sum(result["root"]["strategy"].values()) == pytest.approx(1)
    assert all(math.isfinite(value) for value in result["root"]["strategy"].values())


@pytest.mark.parametrize("players", range(3, 7))
def test_flop_sampling_and_player_counts_are_explicit(players):
    specs = [{"name": str(index), "range": hand} for index, hand in enumerate(
        ["AsAh", "KsKh", "QsQh", "JsJh", "TsTh", "9s9h"][:players])]
    result = solve(board=BOARD[:3], players=specs, pot=20, effective_stack=30,
                   iterations=10, samples=8, backend="cpu", max_raises=0)
    assert result["model"]["player_count"] == players
    assert result["model"]["future_betting"] is False
    assert "sampled final-board" in result["model"]["showdown"]
    assert sum(row["expected_value"] for row in result["ev"]["per_player"]) == pytest.approx(20)


def test_seed_reproduces_policy_and_ev():
    request = dict(board=BOARD, players=PLAYERS, pot=20, effective_stack=40,
                   iterations=30, samples=16, backend="cpu", seed=43)
    first, second = solve(**request), solve(**request)
    assert first["root"] == second["root"]
    assert first["query"] == second["query"]
    assert first["ev"] == second["ev"]


def _controlled_tables(root_hand, action_probabilities):
    game = BettingRound(3, 20, 40, [0.5], 0)
    tables = _Tables(Backend("cpu"))
    row = tables.row(game.initial, root_hand, game.actions(game.initial))
    tables.average(row, np.asarray(action_probabilities), 1)
    tables.commit()
    return game, tables, tables.average_policies()


def test_zero_probability_history_reports_no_sampled_reach_or_ev():
    ranges = [expand_range(player["range"], BOARD) for player in PLAYERS]
    game, tables, average = _controlled_tables(ranges[0][0][1], [1, 0, 0])
    ev, _, posterior, conditioning = _evaluate_policy(
        game, ranges, [encode_card(card) for card in BOARD], tables, average,
        np.random.default_rng(1), 8, PLAYERS, start=game.query(["bet_50%"]))
    assert not conditioning["sampled_reach"]
    assert conditioning["effective_sample_size"] == 0
    assert sum(posterior.values()) == 0
    assert all(row["expected_value"] is None and row["standard_error"] is None for row in ev["per_player"])


def test_query_history_likelihood_changes_private_hand_posterior():
    players = [{"name": "Hero", "range": {"AsAh": 1, "KsKh": 1}}, *PLAYERS[1:]]
    # Use disjoint exact opponents so each hero combo is feasible.
    players[1] = {"name": "Jacks", "range": "JsJh"}
    players[2] = {"name": "Queens", "range": "QsQh"}
    board = ["2c", "3d", "7h", "9s", "Tc"]
    ranges = [expand_range(player["range"], board) for player in players]
    game = BettingRound(3, 20, 40, [0.5], 0)
    tables = _Tables(Backend("cpu"))
    for _, hand, _ in ranges[0]:
        row = tables.row(game.initial, hand, game.actions(game.initial))
        # Only aces can generate the all_in history.
        policy = [0, 0, 1] if hand[0] // 4 == 12 else [1, 0, 0]
        tables.average(row, np.asarray(policy), 1)
    tables.commit()
    average = tables.average_policies()
    # At a query after actor0 all_in, the actingplayer is1. Test posterior via
    # likelihood estimate: roughly half deals, rather than treating everydeal
    # as compatible with the prior actor0 range.
    _, _, _, conditioning = _evaluate_policy(
        game, ranges, [encode_card(card) for card in board], tables, average,
        np.random.default_rng(4), 256, players, start=game.query(["all_in"]))
    assert 0.35 < conditioning["estimated_history_probability"] < 0.65
    assert 0.35 * 256 < conditioning["effective_sample_size"] < 0.65 * 256


def test_metal_rejects_chip_underflow_before_silent_uniform_updates():
    with pytest.raises(MultiPlayerError, match="float32 precision"):
        solve(board=BOARD, players=PLAYERS, pot=1e-50, effective_stack=1e-49,
              iterations=1, samples=2, backend="metal")


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="Actual Metal execution requires Apple silicon")
def test_metal_kernels_execute_and_match_cpu_on_deterministic_toy():
    pytest.importorskip("mlx.core")
    request = dict(board=BOARD, players=PLAYERS, pot=20, effective_stack=40,
                   iterations=60, samples=32, seed=22)
    cpu = solve(**request, backend="cpu")
    metal = solve(**request, backend="metal")
    assert metal["backend"]["metal_execution"] is True
    assert metal["backend"]["executed"] == "mlx_metal_gpu"
    assert metal["backend"]["table_update_batches"] == 60
    assert metal["backend"]["regret_matching_batches"] == 59
    assert metal["root"]["strategy"] == pytest.approx(cpu["root"]["strategy"], abs=1e-6)
    assert [row["expected_value"] for row in metal["ev"]["per_player"]] == pytest.approx(
        [row["expected_value"] for row in cpu["ev"]["per_player"]], abs=1e-6)
