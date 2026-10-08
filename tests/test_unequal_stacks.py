"""Real stack ceilings, reopening rights, and cumulative pot eligibility."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gtosolver.api import app, metal_status
from gtosolver.backend import Backend
from gtosolver.multiplayer import BettingRound
from gtosolver.schemas import SolveRequest
from gtosolver.settlement import normalize_ledger, settlement_payouts
from gtosolver.solver import CFRSolver
from gtosolver.tree import BettingTree


BOARD = ["2c", "3d", "7h", "9s", "Jc"]
HANDS = ["AsAh", "KsKh", "QsQh", "TsTh", "8s8h", "6s6h"]
client = TestClient(app)


def make_game(players, *, pot=20, stacks=None, committed=None,
              dead_contributions=(), min_bet=3, max_raises=2,
              action_overrides=()):
    options = dict(stacks=stacks, committed=committed,
                   dead_contributions=list(dead_contributions), min_bet=min_bet,
                   action_overrides=list(action_overrides))
    if players == 2:
        return BettingTree(pot, 100, [0.5], max_raises, **options)
    return BettingRound(players, pot, 100, [0.5], max_raises, **options)


def state_at(game, history=()):
    return (game.resolve(list(history)) if isinstance(game, BettingTree)
            else game.query(history))


def actions_at(game, history=()):
    state = state_at(game, history)
    return state.actions if isinstance(game, BettingTree) else game.actions(state)


def actions_by_id(game, history=()):
    return {(action.id if isinstance(game, BettingTree) else action.label): action
            for action in actions_at(game, history)}


def request_for(players, *, explicit=False):
    specs = [{"name": f"Player {index}", "range": hand}
             for index, hand in enumerate(HANDS[:players])]
    if explicit:
        for player in specs:
            player.update(stack=40, committed=0)
    return dict(board=BOARD, players=specs, pot=20, effective_stack=40,
                bet_sizes=[0.5], max_raises=0, iterations=2, samples=32,
                seed=19, backend="cpu")


@pytest.mark.parametrize("players", [2, 3, 6])
def test_short_call_uses_call_identity_and_the_actors_actual_ceiling(players):
    game = make_game(players, stacks=[100, 7] + [100] * (players - 2))
    actions = actions_by_id(game, ["bet_50%"])
    assert set(actions) == {"fold", "call"}
    assert actions["call"].to == 7
    assert actions["call"].amount == 7
    state = state_at(game, ["bet_50%", "call"])
    current = state.committed if players == 2 else state.contributions
    assert current[:2] == (10, 7)
    if players == 2:
        assert actions["call"].type == "call"
        assert actions["call"].is_all_in
        assert state.terminal == "showdown"
    else:
        assert actions["call"].kind == "call"
        assert 1 not in state.pending


@pytest.mark.parametrize("players", [2, 3, 6])
def test_opening_all_in_is_not_clipped_to_an_opponents_smaller_stack(players):
    game = make_game(players, stacks=[80, 7] + [40] * (players - 2))
    action = actions_by_id(game)["all_in"]
    assert action.to == action.amount == 80
    assert actions_by_id(game, ["all_in"])["call"].to == 7


@pytest.mark.parametrize("players", [2, 3, 6])
def test_default_stack_contract_matches_explicit_equal_stack_actions(players):
    implicit = make_game(players)
    explicit = make_game(players, stacks=[100] * players,
                         committed=[0] * players)
    for history in [[], ["check"], ["bet_50%"], ["bet_50%", "raise_50%"]]:
        before, after = actions_by_id(implicit, history), actions_by_id(explicit, history)
        assert list(before) == list(after)
        assert [(action.to, action.amount) for action in before.values()] == [
            (action.to, action.amount) for action in after.values()]


@pytest.mark.parametrize("players", [2, 3, 6])
def test_exact_override_is_bounded_by_own_stack_and_can_exceed_short_opponent(players):
    stacks = [100, 7] + [40] * (players - 2)
    game = make_game(players, stacks=stacks,
                     action_overrides=[{"history": [], "to": 60}])
    assert actions_by_id(game)["bet_to_60"].to == 60
    assert actions_by_id(game, ["bet_to_60"])["call"].to == 7
    with pytest.raises(ValueError):
        make_game(players, stacks=stacks,
                  action_overrides=[{"history": [], "to": 101}])


@pytest.mark.parametrize("players", [2, 3, 6])
def test_previously_all_in_player_keeps_pot_eligibility_but_is_not_an_actor(players):
    game = make_game(players, pot=10 * players, stacks=[0] + [100] * (players - 1),
                     committed=[10] * players)
    assert state_at(game).actor == (None if players == 2 else 1)
    history = ["check"] * (players - 1) if players > 2 else []
    state = state_at(game, history)
    assert state.terminal
    if players > 2:
        assert not state.folded[0]
        assert game.payoff(state, [10] + [1] * (players - 1)) == pytest.approx(
            [10 * players] + [0] * (players - 1))


@pytest.mark.parametrize("players", [2, 3, 6])
def test_sole_player_with_chips_cannot_bet_into_only_previously_all_in_players(players):
    game = make_game(players, pot=10 * players,
                     stacks=[0] * (players - 1) + [100], committed=[10] * players)
    assert state_at(game).terminal
    assert state_at(game).actor is None
    assert actions_at(game) == (() if players > 2 else [])


def test_short_all_in_does_not_reopen_a_player_who_already_called():
    game = make_game(3, stacks=[100, 100, 12])
    history = ["bet_50%", "call", "all_in"]
    state = state_at(game, history)
    assert state.actor == 0
    assert state.last_raise == 10
    assert set(actions_by_id(game, history)) == {"fold", "call"}
    assert set(actions_by_id(game, history + ["call"])) == {"fold", "call"}
    with pytest.raises(ValueError):
        make_game(3, stacks=[100, 100, 12], action_overrides=[
            {"history": history, "to": 40}])


def test_unacted_player_can_raise_after_a_short_all_in():
    game = make_game(4, stacks=[100, 12, 100, 100])
    history = ["bet_50%", "all_in"]
    assert state_at(game, history).actor == 2
    assert "raise_50%" in actions_by_id(game, history)
    assert actions_by_id(game, history)["raise_50%"].to == 39


def test_cumulative_short_all_ins_reopen_once_full_increment_has_been_faced():
    game = make_game(4, stacks=[100, 100, 16, 20])
    history = ["bet_50%", "call", "all_in", "all_in"]
    state = state_at(game, history)
    assert state.contributions == (10, 10, 16, 20)
    assert state.last_raise == 10
    assert state.actor == 0
    assert "raise_50%" in actions_by_id(game, history)
    # Calling preserves the same reopening right for the player who called10.
    assert "raise_50%" in actions_by_id(game, history + ["call"])


def test_one_ulp_short_all_in_does_not_reopen_betting_or_consume_full_raise_cap():
    target = math.nextafter(20, 0)
    game = make_game(3, stacks=[100, target, 100], min_bet=3)
    history = ["bet_50%", "all_in", "call"]
    state = state_at(game, history)
    assert state.contributions == (10, target, target)
    assert state.actor == 0
    assert state.last_raise == 10
    assert state.raises == 0
    assert set(actions_by_id(game, history)) == {"fold", "call"}
    with pytest.raises(ValueError):
        make_game(3, stacks=[100, target, 100], min_bet=3,
                  action_overrides=[{"history": history, "to": 40}])


@pytest.mark.parametrize("target", [20, math.nextafter(20, math.inf)])
def test_full_raise_boundary_and_one_ulp_above_reopen_prior_bettor(target):
    game = make_game(3, stacks=[100, target, 100], min_bet=3)
    history = ["bet_50%", "all_in", "call"]
    state = state_at(game, history)
    assert state.actor == 0
    assert state.last_raise == target - 10
    assert state.raises == 1
    assert "raise_50%" in actions_by_id(game, history)


@pytest.mark.parametrize("players", [2, 3, 6])
def test_non_all_in_exact_raise_one_ulp_below_minimum_is_rejected(players):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[
            {"history": ["bet_50%"], "to": math.nextafter(20, 0)}])


@pytest.mark.parametrize("players", [2, 3, 6])
@pytest.mark.parametrize("target", [20, math.nextafter(20, math.inf)])
def test_non_all_in_exact_raise_at_or_above_full_boundary_is_legal(players, target):
    game = make_game(players, action_overrides=[
        {"history": ["bet_50%"], "to": target}])
    assert actions_by_id(game, ["bet_50%"])[f"raise_to_{target:.17g}"].to == target


def long_integer_raise_chain():
    history, overrides = [], []
    for target in range(1, 28):
        overrides.append({"history": history.copy(), "to": target})
        history.append(f"{'bet' if target == 1 else 'raise'}_to_{target}")
    return history, overrides


def test_unequal_short_all_ins_cannot_extend_a_nominally_32_label_override_path():
    _, overrides = long_integer_raise_chain()
    with pytest.raises(ValueError, match="32-label"):
        BettingRound(6, 10, 100, [0.5], 0, min_bet=1,
                     stacks=[27.1, 27.2, 27.3, 27.4, 27.5, 27.6],
                     action_overrides=overrides)


def test_fractional_raise_path_is_rejected_when_later_short_all_ins_exceed_history_budget():
    history = (["check"] * 5 + ["bet_50%"] + ["call"] * 4
               + ["raise_50%"] + ["call"] * 4
               + ["raise_50%"] + ["call"] * 4)
    with pytest.raises(ValueError, match="32-label"):
        BettingRound(6, 30, 3500, [0.5], 2, min_bet=3,
                     stacks=[1785, 2550, 3187.5, 2677.5, 3442.5, 2932.5],
                     action_overrides=[{"history": history, "to": 1275}])


def test_far_equal_stack_override_path_that_finishes_at_label_32_remains_legal():
    history, overrides = long_integer_raise_chain()
    game = BettingRound(6, 10, 100, [0.5], 0, min_bet=1,
                        action_overrides=overrides)
    assert {action.label for action in game.actions(game.query(history))} == {"fold", "call"}
    completed = history + ["call"] * 5
    assert len(completed) == 32
    terminal = game.query(completed)
    assert terminal.terminal
    assert terminal.contributions == (27,) * 6


def test_full_raise_reopens_a_previously_checked_player_and_respects_own_ceiling():
    game = make_game(3, stacks=[60, 100, 25])
    history = ["check", "bet_50%", "call"]
    actions = actions_by_id(game, history)
    assert "raise_50%" in actions
    assert actions["all_in"].to == 60
    assert actions["all_in"].amount == 60


def test_check_does_not_use_up_raise_rights_after_an_incomplete_opening_all_in():
    history = ["check", "all_in", "call"]
    game = make_game(3, stacks=[100, 2, 100], min_bet=3,
                     action_overrides=[{"history": history, "to": 5}])
    state = state_at(game, history)
    assert state.actor == 0
    assert state.contributions == (0, 2, 2)
    actions = actions_by_id(game, history)
    assert actions["raise_to_5"].to == actions["raise_to_5"].amount == 5
    assert "raise_50%" in actions
    following = state_at(game, history + ["raise_to_5"])
    assert following.last_raise == 3
    assert following.actor == 2  # The opening short bettor is already all-in.


def test_only_non_all_in_player_must_answer_an_incomplete_bet_without_raising():
    game = make_game(2, stacks=[100, 2], min_bet=3)
    history = ["check", "all_in"]
    assert state_at(game, history).actor == 0
    assert set(actions_by_id(game, history)) == {"fold", "call"}
    assert actions_by_id(game, history)["call"].to == 2
    assert state_at(game, history + ["call"]).terminal == "showdown"


@pytest.mark.parametrize("pot,expected", [
    (130, [70, 120, 0]),
    (150, [90, 120, 0]),
])
def test_prior_commitments_define_main_and_side_pot_with_shared_remainder(pot, expected):
    payouts = settlement_payouts(pot, [20, 50, 50], [0, 30, 30],
                                 [False] * 3, [10], [3, 2, 1])
    assert payouts == pytest.approx(expected)
    assert sum(payouts) == pytest.approx(pot + 60)


def test_ties_split_only_the_pots_for_which_both_winners_are_eligible():
    payouts = settlement_payouts(130, [20, 50, 50], [0, 30, 30],
                                 [False] * 3, [10], [3, 3, 1])
    assert payouts == pytest.approx([35, 155, 0])


def test_folded_prior_and_dead_players_fund_pots_without_winning_them():
    payouts = settlement_payouts(95, [10, 20, 30], [0, 20, 10],
                                 [False, True, False], [5, 30], [3, 100, 1])
    assert payouts == pytest.approx([45, 0, 80])
    assert sum(payouts) == 125


def test_unmatched_overbet_is_returned_even_when_the_short_caller_wins():
    payouts = settlement_payouts(20, [10, 10], [100, 7],
                                 [False, False], [], [1, 2])
    assert payouts == pytest.approx([93, 34])


@pytest.mark.parametrize("unit", [1, 1e-12])
def test_one_ulp_ledger_addition_noise_is_allowed_without_negative_external_money(unit):
    prior = [amount * unit for amount in [0.1, 0.2, 0.3]]
    dead = [amount * unit for amount in [0.05, 0.15]]
    current = [0.05 * unit, 0.05 * unit, 0]
    recorded = math.fsum([*prior, *dead])
    noisy_pot = math.nextafter(recorded, 0)
    stacks, accepted_prior, accepted_dead = normalize_ledger(
        3, noisy_pot, unit, committed=prior, dead_contributions=dead)
    assert stacks == (unit,) * 3
    assert accepted_prior == tuple(prior)
    assert accepted_dead == tuple(dead)
    payouts = settlement_payouts(noisy_pot, prior, current,
                                 [False] * 3, dead, [3, 2, 1])
    canonical = settlement_payouts(recorded, prior, current,
                                   [False] * 3, dead, [3, 2, 1])
    assert np.array_equal(payouts, canonical)
    assert np.all(np.isfinite(payouts))
    total_pot = math.fsum([noisy_pot, *current])
    assert abs(math.fsum(payouts) - total_pot) <= 8 * math.ulp(total_pot)


def test_tiny_chip_units_do_not_hide_a_prior_ledger_larger_than_the_entire_pot():
    with pytest.raises(ValueError, match="pot must include"):
        normalize_ledger(2, 1e-12, 1e-9, committed=[1e-10, 0])
    with pytest.raises(ValueError, match="pot must include"):
        settlement_payouts(1e-12, [1e-10, 0], [0, 0],
                           [False, False], [], [1, 2])
    with pytest.raises(ValueError, match="pot must include"):
        make_game(2, pot=1e-12, committed=[1e-10, 0])


def test_multiplayer_net_utilities_subtract_only_new_street_contributions():
    game = make_game(3, pot=130, stacks=[0, 30, 30],
                     committed=[20, 50, 50], dead_contributions=[10])
    state = state_at(game, ["all_in", "call"])
    assert state.terminal
    assert state.contributions == (0, 30, 30)
    values = game.payoff(state, [3, 2, 1])
    assert values == pytest.approx([70, 90, -30])
    assert sum(values) == pytest.approx(game.pot)


def test_six_player_later_folds_keep_their_prior_chips_in_every_eligible_pot():
    # Two prior all-ins remain eligible. Two other players fold this street;
    # their old investments still fund the surviving players' contribution tiers.
    game = make_game(6, pot=195, stacks=[0, 10, 0, 20, 10, 50],
                     committed=[5, 10, 20, 30, 40, 50],
                     dead_contributions=[15, 25])
    state = state_at(game, ["check", "all_in", "call", "fold", "fold"])
    assert state.terminal
    assert state.contributions == (0, 0, 0, 20, 10, 0)
    assert state.folded == (False, True, False, False, False, True)
    strengths = [6, 100, 5, 4, 3, 200]
    payouts = settlement_payouts(game.pot, game.prior_committed,
                                 state.contributions, state.folded,
                                 game.dead_contributions, strengths)
    assert payouts == pytest.approx([40, 0, 90, 95, 0, 0])
    assert sum(payouts) == 225
    utility = game.payoff(state, strengths)
    assert utility == pytest.approx([40, 0, 90, 75, -10, 0])
    assert sum(utility) == 195


@pytest.mark.parametrize("sign,expected", [(1, 20), (-1, -40), (0, -10)])
def test_heads_up_terminal_utility_uses_cumulative_eligibility_and_zero_sum_centering(sign, expected):
    game = make_game(2, pot=60, stacks=[10, 40], committed=[20, 40])
    terminal = state_at(game, ["check", "all_in", "call"])
    assert terminal.terminal == "showdown"
    solver = CFRSolver(game, [[1]], [[sign]], ([1], [1]), Backend("cpu"))
    values = [np.asarray(solver._terminal_value(
        terminal, player, np.ones(1), {})).item() for player in (0, 1)]
    assert values == pytest.approx([expected, -expected])


@pytest.mark.parametrize("players", [2, 3, 6])
@pytest.mark.parametrize("field,value", [("stack", -1), ("stack", True),
                                         ("stack", 1e9 + 1),
                                         ("committed", -1), ("committed", True),
                                         ("committed", 1e9 + 1)])
def test_schema_rejects_invalid_player_stack_and_prior_commitment(players, field, value):
    request = request_for(players)
    request["players"][0][field] = value
    with pytest.raises(ValidationError):
        SolveRequest(**request)


@pytest.mark.parametrize("field", ["stack", "committed"])
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_schema_rejects_nonfinite_player_chips(field, value):
    request = request_for(2)
    request["players"][0][field] = value
    with pytest.raises(ValidationError):
        SolveRequest(**request)


@pytest.mark.parametrize("players", [2, 3, 6])
def test_api_omitted_fields_and_explicit_equal_defaults_have_identical_policies(players):
    implicit = client.post("/v1/solve", json=request_for(players))
    explicit = client.post("/v1/solve", json=request_for(players, explicit=True))
    assert implicit.status_code == explicit.status_code == 200
    assert implicit.json()["root"] == explicit.json()["root"]
    assert implicit.json()["query"] == explicit.json()["query"]


@pytest.mark.parametrize("players", [2, 3, 6])
def test_api_reports_short_call_without_erasing_the_overbettors_actual_wager(players):
    request = request_for(players)
    request["players"][0].update(stack=80, committed=10)
    request["players"][1].update(stack=7, committed=10)
    for player in request["players"][2:]:
        player.update(stack=40, committed=0)
    request["history"] = ["all_in"]
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 200, response.text
    query = response.json()["query"]
    actions = {action.get("id", action.get("label")): action for action in query["actions"]}
    assert set(actions) == {"fold", "call"}
    assert actions["call"]["to"] == 7
    assert actions["call"]["amount"] == 7
    assert actions["call"]["type"] == "call"
    current = query["committed"] if players == 2 else query["contributions"]
    assert current[0] == 80
    assert request["pot"] + sum(current) == 100


@pytest.mark.parametrize("players", [2, 3, 6])
def test_api_accepts_zero_stacks_and_prior_side_pot_ledgers(players):
    request = request_for(players)
    for index, player in enumerate(request["players"]):
        player.update(stack=0, committed=5 + index)
    request["dead_contributions"] = [3, 8]
    request["pot"] = sum(player["committed"] for player in request["players"]) + 11
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 200, response.text
    query = response.json()["query"]
    assert query["actions"] == []
    assert query["terminal"]
    assert query.get("player", query.get("actor")) is None


@pytest.mark.parametrize("players", [2, 3, 6])
def test_api_rejects_a_pot_smaller_than_the_supplied_prior_and_dead_ledger(players):
    request = request_for(players)
    for player in request["players"]:
        player.update(committed=10)
    request["dead_contributions"] = [5]
    request["pot"] = 10 * players + 4
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("players", [2, 3, 6])
def test_api_rejects_non_all_in_exact_raise_even_one_ulp_below_full_minimum(players):
    request = request_for(players)
    request["min_bet"] = 3
    request["action_overrides"] = [{"history": ["bet_50%"],
                                    "to": math.nextafter(20, 0)}]
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 422, response.text


def test_api_rejects_invalid_prior_ledger_even_when_chips_are_tiny():
    request = request_for(2)
    request.update(pot=1e-12, effective_stack=1e-9)
    request["players"][0]["committed"] = 1e-10
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 422, response.text


def test_api_rejects_unclosable_unequal_override_path_before_backend_allocation(monkeypatch):
    def unexpected_backend(*args, **kwargs):
        pytest.fail("An invalid betting history allocated a solver backend")

    monkeypatch.setattr("gtosolver.multiplayer.Backend", unexpected_backend)
    request = request_for(6)
    _, request["action_overrides"] = long_integer_raise_chain()
    request.update(pot=10, effective_stack=100, min_bet=1)
    for player, stack in zip(request["players"], [27.1, 27.2, 27.3, 27.4, 27.5, 27.6]):
        player["stack"] = stack
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 422, response.text
    assert "32-label" in response.json()["detail"]["message"]


@pytest.mark.parametrize("players", [2, 3, 6])
@pytest.mark.parametrize("options", [
    {"stacks": [10]},
    {"committed": [10]},
    {"stacks": [-1]},
    {"dead_contributions": [-1]},
])
def test_direct_constructors_reject_invalid_ledger_shapes_or_values(players, options):
    with pytest.raises(ValueError):
        make_game(players, **options)


@pytest.mark.metal
@pytest.mark.parametrize("players", [2, 3, 6])
def test_unequal_stack_requests_execute_actual_metal_kernels(players):
    if not metal_status()["available"]:
        pytest.skip("Metal unavailable")
    request = request_for(players)
    request.update(backend="metal", iterations=2)
    for index, player in enumerate(request["players"]):
        player.update(stack=15 + 5 * index, committed=2 + index)
    request["pot"] = sum(player["committed"] for player in request["players"]) + 20
    response = client.post("/v1/solve", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["backend"]["actual"] == "metal"
    assert result["backend"]["metal_execution"] is True
    assert result["backend"]["executed"] == "mlx_metal_gpu"
