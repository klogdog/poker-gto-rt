"""Independent estimator checks with exactly enumerated sample likelihoods."""

import numpy as np

from gtosolver.backend import Backend
from gtosolver.cards import encode_card
from gtosolver.multiplayer import (
    Action, BettingRound, MAX_ACTIONS, State, _Tables, _Trainer,
    _evaluate_policy, _state_report,
)
from gtosolver.ranges import expand_range


class _ThreePlayerGame:
    """Seat 2 acts, seat 0 acts, the traverser acts, then seat 2 acts again."""

    players = 3
    pot = stack = 1.0
    initial = State((0.0,) * 3, (False,) * 3, (2, 0, 1, 2))

    def actions(self, state):
        return () if state.terminal else (
            Action("a", "check", 0.0, 0.0),
            Action("b", "check", 0.0, 0.0),
        )

    def advance(self, state, action):
        return State(state.contributions, state.folded, state.pending[1:],
                     history=state.history + (action.label,))

    def payoff(self, state, strengths):
        return np.zeros(3)


class _ChooseA:
    """Enumerate the one sampled path that reaches the tested information set."""

    def choice(self, count, p):
        return 0


def _expected_target_average(iteration, third_party_probability, target_policy):
    game = _ThreePlayerGame()
    hands = ((0, 1), (2, 3), (4, 5))
    tables = _Tables(Backend("cpu"))
    own_prior_probability = 0.5
    policies = [
        [own_prior_probability, 1 - own_prior_probability],
        [third_party_probability, 1 - third_party_probability],
        [0.5, 0.5],
        target_policy,
    ]
    state = game.initial
    for policy in policies:
        tables.row(state, hands[state.actor], game.actions(state))
        state = game.advance(state, game.actions(state)[0])
    tables.policy = np.zeros((len(policies), MAX_ACTIONS))
    tables.policy[:, :2] = policies
    trainer = _Trainer(game, tables, _ChooseA())
    trainer.traverse(game.initial, hands, (1, 1, 1), 1, np.ones(3), iteration)
    target_row = tables.index[(("a", "a", "a"), hands[2])]
    # The traverser enumerates its own actions, so its reach is absent from the
    # sample likelihood. Seats 2 and 0 take the sampled actions preceding it.
    sample_likelihood = own_prior_probability * third_party_probability
    return tables.average_delta[target_row][:2] * sample_likelihood


def test_multiplayer_average_cancels_third_party_visitation_not_own_reach():
    first = _expected_target_average(1, 0.2, [0.8, 0.2])
    second = _expected_target_average(2, 0.8, [0.2, 0.8])
    # Expected accumulation retains the actor's own realization reach (0.5),
    # while removing the changing third-party visitation probabilities.
    np.testing.assert_allclose(first, [0.4, 0.1])
    np.testing.assert_allclose(second, [0.2, 0.8])
    average = (first + second) / (first + second).sum()
    np.testing.assert_allclose(average, [0.4, 0.6])


def _forced_history_case(allow_check):
    board = ["2c", "3d", "4h", "9s", "Jc"]
    players = [
        {"name": "A", "range": "AsAh,KsKh"},
        {"name": "B", "range": "QsQh"},
        {"name": "C", "range": "TcTd"},
    ]
    ranges = [expand_range(player["range"], board) for player in players]
    game = BettingRound(3, 10, 20, [0.5], 0)
    tables = _Tables(Backend("cpu"))
    for _, hand, _ in ranges[0]:
        row = tables.row(game.initial, hand, game.actions(game.initial))
        tables.infos[row].average_updates = 1
    average = np.zeros((len(tables.infos), MAX_ACTIONS))
    for row, info in enumerate(tables.infos):
        king_pair = all(card // 4 == 11 for card in info.hand)
        average[row, 0 if king_pair and allow_check else 1] = 1.0
    query = game.query(["check", "check", "bet_50%"])
    ev, root_prior, posterior, conditioning = _evaluate_policy(
        game, ranges, [encode_card(card) for card in board], tables,
        average, np.random.default_rng(19), 32, players, start=query,
    )
    report = _state_report(game, query, ranges, tables, average, players,
                           posterior=posterior)
    return ev, posterior, conditioning, report


def test_multiplayer_history_posterior_excludes_hand_with_zero_action_likelihood():
    ev, posterior, conditioning, report = _forced_history_case(True)
    king_pair = tuple(sorted((encode_card("Ks"), encode_card("Kh"))))
    assert posterior[king_pair] > 0
    assert all(value == 0 for hand, value in posterior.items() if hand != king_pair)
    assert conditioning["sampled_reach"]
    assert report["recommendation"] is not None
    np.testing.assert_allclose(
        sum(value["expected_value"] for value in ev["per_player"]), 10.0,
    )


def test_multiplayer_zero_history_likelihood_returns_no_aggregate_or_ev():
    ev, _, conditioning, report = _forced_history_case(False)
    assert not conditioning["sampled_reach"]
    assert conditioning["effective_sample_size"] == 0
    assert report["recommendation"] is None
    assert report["strategy"] is None
    assert all(value["expected_value"] is None for value in ev["per_player"])
