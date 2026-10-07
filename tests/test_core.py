"""Analytic Kuhn acceptance, legal tree behavior, and real GPU parity."""

import platform

import numpy as np
import pytest

from gtosolver.backend import Backend, BackendUnavailable
from gtosolver.solver import CFRSolver, solve
from gtosolver.tree import BettingTree


def kuhn(backend="cpu", iterations=800):
    ranks = np.arange(3)
    game = CFRSolver(BettingTree(2, 1, [.5], 0), 1 - np.eye(3),
                     np.sign(ranks[:, None] - ranks[None, :]), ([1, 1, 1], [1, 1, 1]), Backend(backend))
    game.run(iterations)
    return game


def test_kuhn_game_known_value_and_best_response_gap():
    game = kuhn()
    diagnostics = game.diagnostics(game.average_policy())
    assert diagnostics["root_ev"]["oop"] == pytest.approx(-1 / 18, abs=2e-5)
    assert diagnostics["root_ev"]["oop"] + diagnostics["root_ev"]["ip"] == pytest.approx(0, abs=1e-12)
    assert diagnostics["nash_conv"] < .0005
    for strategy in game.average_policy().values():
        assert np.all(np.asarray(strategy) >= 0)
        np.testing.assert_allclose(np.asarray(strategy).sum(axis=1), 1)


def test_tree_minimum_raise_short_all_in_and_raise_limit():
    tree = BettingTree(10, 8, [.1, .5], 2)
    facing = tree.resolve(["bet_50%"])
    assert [a.id for a in facing.actions] == ["fold", "call", "all_in"]
    all_in = tree.resolve(["bet_50%", "all_in"])
    assert [a.id for a in all_in.actions] == ["fold", "call"]
    called = tree.resolve(["bet_50%", "all_in", "call"])
    assert called.terminal == "showdown"
    assert called.committed == (8, 8)
    with pytest.raises(ValueError, match="unavailable"):
        tree.resolve(["bet_50%", "raise_10%"])


def test_tree_fraction_raise_uses_pot_after_call():
    tree = BettingTree(10, 100, [.5], 1)
    node = tree.resolve(["bet_50%"])
    action = next(a for a in node.actions if a.id == "raise_50%")
    assert action.to == 15  # 5 call + half of the 20 pot after calling.
    assert action.amount == 15
    assert [a.id for a in action.child.actions] == ["fold", "call"]


def test_tree_tiny_chip_units_preserve_actions():
    tree = BettingTree(1e-12, 1e-12, [.5], 0)
    assert [a.id for a in tree.root.actions] == ["check", "bet_50%", "all_in"]


def test_stable_range_weight_normalization():
    tree = BettingTree(2, 1, [.5], 0)
    game = CFRSolver(tree, np.ones((2, 2)), np.zeros((2, 2)),
                     ([1e308, 1e308], [1e308, 1e308]), Backend("cpu"))
    game.run(2)
    assert game.joint_mass == pytest.approx(1)
    assert game.diagnostics(game.average_policy())["root_ev"]["oop"] == pytest.approx(0, abs=.5)


def test_real_river_history_query_and_terminal_units():
    result = solve(["As", "Kd", "7h", "2c", "3s"], "AA,KK", "QQ,JJ", 10, 50,
                   iterations=200, backend="cpu", history=["bet_50%"], include_tree=True)
    assert result["payoff_model"]["mode"] == "exact_river"
    assert result["query"]["actor"] == "ip"
    assert result["root"]["actor"] == "oop"
    assert result["query"]["history"] == ["bet_50%"]
    assert result["query"]["pot"] == 15
    assert result["diagnostics"]["root_ev"]["oop"] == pytest.approx(5, abs=.01)
    assert result["tree"]["all_in_included"]
    assert result["limitations"]
    for hand in result["query"]["strategy"]:
        assert hand["action_ev"]["fold"] == pytest.approx(-5)
        assert sum(hand["probabilities"].values()) == pytest.approx(1)


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple silicon required")
def test_actual_metal_kuhn_matches_cpu():
    pytest.importorskip("mlx.core")
    cpu = kuhn("cpu", 300)
    metal = kuhn("metal", 300)
    assert metal.backend.info["metal_execution"] is True
    assert metal.backend.info["actual"] == "metal"
    assert metal.backend.info["executed"] == "mlx_metal_gpu"
    cpu_diag = cpu.diagnostics(cpu.average_policy())
    metal_diag = metal.diagnostics(metal.average_policy())
    assert metal_diag["root_ev"]["oop"] == pytest.approx(cpu_diag["root_ev"]["oop"], abs=3e-5)
    assert metal_diag["nash_conv"] == pytest.approx(cpu_diag["nash_conv"], abs=3e-5)
    for key in cpu.average_policy():
        np.testing.assert_allclose(np.asarray(metal.average_policy()[key]), np.asarray(cpu.average_policy()[key]), atol=3e-4)


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple silicon required")
def test_metal_rejects_unrepresentable_weight_distribution():
    pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="too small for Metal"):
        CFRSolver(BettingTree(10, 20, [.5], 0), np.array([[0, 1], [1, 1]]), np.zeros((2, 2)),
                  ([1, 1e-100], [1, 1e-100]), Backend("metal"))


def test_requested_metal_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(BackendUnavailable, match="Apple silicon"):
        Backend("metal")


@pytest.mark.parametrize("minor_weight", [1e-20, 1e-100])
def test_rare_compatible_ranges_remain_reachable_cpu(minor_weight):
    result = solve(["2c", "3d", "4h", "9s", "Jc"],
                   {"AsAh": 1, "KsKh": minor_weight},
                   {"AsAd": 1, "QsQh": minor_weight},
                   10, 20, iterations=1, backend="cpu")
    root = result["root"]
    assert root["history_probability"] == pytest.approx(1)
    assert root["reachable_under_average_strategy"]
    assert sum(root["aggregate_strategy"].values()) == pytest.approx(1)
    assert all(hand["opponent_compatible"] for hand in root["strategy"])
    assert all(np.isfinite(hand["ev"]) for hand in root["strategy"])
    assert sum(hand["posterior_probability"] for hand in root["strategy"]) == pytest.approx(1)


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple silicon required")
def test_rare_representable_compatible_ranges_remain_reachable_metal():
    pytest.importorskip("mlx.core")
    result = solve(["2c", "3d", "4h", "9s", "Jc"],
                   {"AsAh": 1, "KsKh": 1e-20},
                   {"AsAd": 1, "QsQh": 1e-20},
                   10, 20, iterations=1, backend="metal")
    root = result["root"]
    assert root["history_probability"] == pytest.approx(1, abs=2e-6)
    assert root["reachable_under_average_strategy"]
    assert sum(root["aggregate_strategy"].values()) == pytest.approx(1, abs=2e-6)
    assert all(hand["opponent_compatible"] for hand in root["strategy"])
    assert all(np.isfinite(hand["ev"]) for hand in root["strategy"])
