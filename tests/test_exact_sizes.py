"""Exact public-history wagers remain legal, stable, and solver-visible."""

import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gtosolver.api import app
from gtosolver.multiplayer import BettingRound, MAX_ACTIONS
from gtosolver.schemas import SolveRequest
from gtosolver.tree import BettingTree


@pytest.fixture(params=[2, 3, 6], ids=["heads-up", "three-player", "six-player"])
def players(request):
    return request.param


def make_game(players, *, pot=20, stack=500, sizes=(0.5, 1.0),
              max_raises=1, min_bet=3, action_overrides=()):
    options = dict(min_bet=min_bet, action_overrides=list(action_overrides))
    if players == 2:
        return BettingTree(pot, stack, list(sizes), max_raises, **options)
    return BettingRound(players, pot, stack, sizes, max_raises, **options)


def state_at(game, history=()):
    return (game.resolve(list(history)) if isinstance(game, BettingTree)
            else game.query(history))


def actions_at(game, history=()):
    state = state_at(game, history)
    return state.actions if isinstance(game, BettingTree) else game.actions(state)


def action_label(action):
    return action.id if hasattr(action, "id") else action.label


def labeled(game, history=()):
    return {action_label(action): action for action in actions_at(game, history)}


def target_label(kind, target):
    return f"{kind}_to_{target:.17g}"


def test_exact_opening_preserves_existing_targets_and_history(players):
    baseline = make_game(players)
    exact = make_game(players, action_overrides=[{"history": [], "to": 7}])
    before, after = labeled(baseline), labeled(exact)
    assert set(after) == set(before) | {"bet_to_7"}
    for label, action in before.items():
        assert after[label].to == action.to
        assert after[label].amount == action.amount
    custom = after["bet_to_7"]
    assert custom.to == custom.amount == 7
    response = labeled(exact, ["bet_to_7"])
    assert response["call"].amount == 7
    assert response["call"].to == 7
    assert "bet_to_7" not in labeled(exact, ["check"])


@pytest.mark.parametrize("target,expected", [(10, "bet_50%"), (20, "bet_100%"), (500, "all_in")])
def test_duplicate_targets_keep_the_existing_canonical_label(players, target, expected):
    baseline = make_game(players)
    exact = make_game(players, action_overrides=[{"history": [], "to": target}])
    assert list(labeled(exact)) == list(labeled(baseline))
    matching = [action for action in actions_at(exact) if action.to == target]
    assert len(matching) == 1
    assert action_label(matching[0]) == expected


def test_duplicate_raise_keeps_the_previously_returned_fraction_label(players):
    history = ["bet_50%"]
    baseline = make_game(players)
    exact = make_game(players, action_overrides=[{"history": history, "to": 30}])
    assert list(labeled(exact, history)) == list(labeled(baseline, history))
    assert labeled(exact, history)["raise_50%"].to == 30
    assert "raise_to_30" not in labeled(exact, history)


def test_override_preflight_resolves_parent_overrides_before_child_overrides(players):
    overrides = [{"history": ["bet_to_7"], "to": 18},
                 {"history": [], "to": 7}]
    game = make_game(players, action_overrides=overrides)
    assert labeled(game, ["bet_to_7"])["raise_to_18"].to == 18
    assert labeled(game, ["bet_to_7", "raise_to_18"])["call"].to == 18


def test_duplicate_fractional_target_does_not_make_a_new_history_label(players):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": [], "to": 10},
                  {"history": ["bet_to_10"], "to": 30}])


def test_exact_raise_includes_the_actors_existing_commitment(players):
    history = ["bet_50%"]
    game = make_game(players, action_overrides=[{"history": history, "to": 40}])
    actions = labeled(game, history)
    assert actions["raise_to_40"].to == 40
    assert actions["raise_to_40"].amount == 40
    assert len(actions) == MAX_ACTIONS
    state = state_at(game, history + ["raise_to_40"])
    commitments = state.committed if isinstance(game, BettingTree) else state.contributions
    assert commitments[:2] == (10, 40)
    next_actions = labeled(game, history + ["raise_to_40"])
    assert next_actions["call"].to == 40
    expected_call = 30 if players == 2 else 40
    assert next_actions["call"].amount == expected_call


def test_override_past_the_raise_cap_is_scoped_to_its_exact_node(players):
    history = ["bet_50%", "raise_50%"]
    baseline = make_game(players)
    assert set(labeled(baseline, history)) == {"fold", "call"}
    game = make_game(players, action_overrides=[{"history": history, "to": 60}])
    actions = labeled(game, history)
    assert set(actions) == {"fold", "call", "raise_to_60", "all_in"}
    assert actions["raise_to_60"].to == 60
    assert "raise_50%" not in actions
    assert set(labeled(game, history + ["raise_to_60"])) == {"fold", "call"}
    # A sibling with the same contribution amounts has no exact override.
    sibling = ["check", "bet_50%", "raise_50%"]
    assert set(labeled(game, sibling)) == {"fold", "call"}


def test_exact_raise_can_reopen_a_zero_raise_cap_without_changing_future_cap(players):
    game = make_game(players, max_raises=0,
                     action_overrides=[{"history": ["bet_50%"], "to": 25}])
    assert set(labeled(game, ["bet_50%"])) == {"fold", "call", "raise_to_25", "all_in"}
    assert set(labeled(game, ["bet_50%", "raise_to_25"])) == {"fold", "call"}


@pytest.mark.parametrize("pot,stack,target", [(100, 200, 1), (1, 100, 20)])
def test_exact_chips_are_independent_of_the_pot_fraction_schema_limits(players, pot, stack, target):
    game = make_game(players, pot=pot, stack=stack, min_bet=0,
                     action_overrides=[{"history": [], "to": target}])
    action = labeled(game)[target_label("bet", target)]
    assert action.to == action.amount == target
    assert target / pot < 0.05 or target / pot > 10


@pytest.mark.parametrize("players", [3, 6])
def test_overrides_reject_more_than_six_actions_before_allocating_gpu_tables(players):
    with pytest.raises(ValueError):
        make_game(players, sizes=(0.25, 0.5, 1.0),
                  action_overrides=[{"history": ["bet_25%"], "to": 15}])


def test_exact_label_retains_full_float_precision(players):
    target = 7.123456789012345
    game = make_game(players, action_overrides=[{"history": [], "to": target}])
    label = target_label("bet", target)
    assert label in labeled(game)
    assert labeled(game)[label].to == target
    assert labeled(game, [label])["call"].to == target


def test_minimum_opening_bet_filters_fractional_bets(players):
    game = make_game(players, pot=10, stack=100, min_bet=6)
    actions = labeled(game)
    assert "bet_50%" not in actions
    assert actions["bet_100%"].to == 10
    assert actions["all_in"].to == 100


def test_short_opening_all_in_remains_legal_below_the_minimum(players):
    game = make_game(players, pot=10, stack=2, min_bet=3)
    assert set(labeled(game)) == {"check", "all_in"}
    assert labeled(game)["all_in"].to == 2
    assert set(labeled(game, ["all_in"])) == {"fold", "call"}


def test_short_all_in_raise_remains_legal_below_the_previous_full_raise(players):
    game = make_game(players, pot=20, stack=12, min_bet=3)
    actions = labeled(game, ["bet_50%"])
    assert actions["all_in"].to == 12
    assert actions["all_in"].amount == 12
    assert set(labeled(game, ["bet_50%", "all_in"])) == {"fold", "call"}


def test_minimum_full_raise_uses_the_actual_exact_opening_increment(players):
    overrides = [{"history": [], "to": 7},
                 {"history": ["bet_to_7"], "to": 14}]
    game = make_game(players, action_overrides=overrides)
    action = labeled(game, ["bet_to_7"])["raise_to_14"]
    assert action.to == 14
    assert action.amount == 14
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[overrides[0],
                  {"history": ["bet_to_7"], "to": 13.5}])


@pytest.mark.parametrize("minimum", [-1, math.inf, -math.inf, math.nan, True, "3"])
def test_invalid_minimum_is_rejected_before_building_a_tree(players, minimum):
    with pytest.raises(ValueError):
        make_game(players, min_bet=minimum)


@pytest.mark.parametrize("target", [0, -1, 2, 501, math.inf, -math.inf, math.nan, True, "7"])
def test_invalid_exact_opening_target_is_rejected(players, target):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": [], "to": target}])


@pytest.mark.parametrize("target", [5, 10, 19, 501])
def test_nonraising_or_under_minimum_raise_target_is_rejected(players, target):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": ["bet_50%"], "to": target}])


@pytest.mark.parametrize("second_target", [7, 8])
def test_repeated_or_ambiguous_override_history_is_rejected(players, second_target):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": [], "to": 7},
                  {"history": [], "to": second_target}])


@pytest.mark.parametrize("history", [["not_an_action"], ["fold"], ["all_in", "call", "check"]])
def test_unreachable_override_history_is_rejected(players, history):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": history, "to": 7}])


def test_terminal_override_history_is_rejected_instead_of_silently_unused(players):
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": ["check"] * players, "to": 7}])


def exact_raise_chain(count):
    history, overrides = [], []
    for index in range(count):
        target = (index + 1) / 1024
        overrides.append({"history": history.copy(), "to": target})
        history.append(target_label("bet" if index == 0 else "raise", target))
    return history, overrides


def test_exact_actions_preserve_a_long_accepted_history_that_closes_within_32(players):
    aggressive_count = 32 - (players - 1)
    history, overrides = exact_raise_chain(aggressive_count)
    game = make_game(players, pot=1, stack=100, max_raises=0,
                     min_bet=1 / 1024, action_overrides=overrides)
    assert set(labeled(game, history)) == {"fold", "call"}
    completed_history = history + ["call"] * (players - 1)
    assert len(completed_history) == 32
    state = state_at(game, completed_history)
    commitments = state.committed if isinstance(game, BettingTree) else state.contributions
    assert all(amount == aggressive_count / 1024 for amount in commitments)
    assert state.terminal


def test_override_chain_is_rejected_if_calls_cannot_close_within_history_budget(players):
    _, overrides = exact_raise_chain(33 - (players - 1))
    with pytest.raises(ValueError):
        make_game(players, pot=1, stack=100, max_raises=0,
                  min_bet=1 / 1024, action_overrides=overrides)


def test_override_count_and_child_history_limits_are_bounded(players):
    _, overrides = exact_raise_chain(33)
    with pytest.raises(ValueError):
        make_game(players, pot=1, stack=100, max_raises=0,
                  min_bet=1 / 1024, action_overrides=overrides)
    with pytest.raises(ValueError):
        make_game(players, action_overrides=[{"history": ["check"] * 33, "to": 7}])


def api_payload(players=2, **changes):
    result = dict(board=["2c", "3d", "7h", "9s", "Jc"],
                  players=[{"name": str(index), "range": hand} for index, hand in enumerate(
                      ["AsAh", "KsKh", "QsQh", "TsTh", "8s8h", "6s6h"][:players])],
                  pot=20, effective_stack=100, bet_sizes=[0.5, 1.0],
                  iterations=2, samples=32, max_raises=0, backend="cpu",
                  min_bet=3, action_overrides=[{"history": [], "to": 7}],
                  history=["bet_to_7"], seed=2)
    result.update(changes)
    return result


def test_request_schema_exposes_exact_wagers_and_a_minimum_unit():
    schema = TestClient(app).get("/openapi.json").json()
    properties = schema["components"]["schemas"]["SolveRequest"]["properties"]
    assert properties["min_bet"]["default"] == 0
    assert "action_overrides" in properties
    request = SolveRequest.model_validate(api_payload())
    assert request.min_bet == 3
    assert request.model_dump()["action_overrides"] == [{"history": [], "to": 7}]


@pytest.mark.parametrize("changes", [
    {"min_bet": -1}, {"min_bet": math.inf}, {"min_bet": True},
    {"action_overrides": [{"history": [], "to": math.nan}]},
    {"action_overrides": [{"history": [], "to": True}]},
    {"action_overrides": [{"history": [], "to": 0}]},
    {"action_overrides": [{"history": [], "to": 7, "extra": True}]},
    {"action_overrides": [{"history": ["check"] * 33, "to": 7}]},
    {"action_overrides": [{"history": [""], "to": 7}]},
    {"action_overrides": [{"history": ["x" * 65], "to": 7}]},
    {"action_overrides": [{"history": [], "to": 7}] * 33},
])
def test_schema_rejects_malformed_exact_wager_requests(changes):
    with pytest.raises(ValidationError):
        SolveRequest.model_validate(api_payload(**changes))


@pytest.mark.parametrize("players", [2, 3])
def test_cpu_http_solver_preserves_exact_history_and_amounts(players):
    payload = api_payload(players, action_overrides=[
        {"history": [], "to": 7},
        {"history": ["bet_to_7"], "to": 18},
    ])
    response = TestClient(app).post("/v1/solve", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["backend"]["actual"] == "cpu"
    assert result["request"]["action_overrides"] == payload["action_overrides"]
    query = result["query"]
    assert query["history"] == ["bet_to_7"]
    actions = {action.get("id", action.get("label")): action for action in query["actions"]}
    assert actions["call"]["amount"] == 7
    assert actions["raise_to_18"]["to"] == 18
    assert actions["raise_to_18"]["amount"] == 18
    assert "raise_50%" not in actions


@pytest.mark.parametrize("players", [2, 3])
def test_http_rejects_unused_or_out_of_stack_exact_wagers(players):
    client = TestClient(app)
    for overrides in ([{"history": ["not_an_action"], "to": 7}],
                      [{"history": [], "to": 101}],
                      [{"history": [], "to": 7}, {"history": [], "to": 8}]):
        response = client.post("/v1/solve", json=api_payload(players,
                               action_overrides=overrides, history=[]))
        assert response.status_code == 422, response.text
