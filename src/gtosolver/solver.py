"""Heads-up vector CFR+ with explicit blocker conditioning and best responses.

The solved game is one betting round. Earlier streets use showdown equity for
the remaining runout, without modelling subsequent betting rounds.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .backend import Backend
from .tree import BettingTree, Node


class CFRSolver:
    """Alternating CFR+, linear realization-weighted average strategies.

    Each public decision node and private hand is an information set. Private
    chance probabilities are independent range weights, conditioned jointly on
    non-overlap. Counterfactual values omit a player's own realization reach.
    """

    def __init__(self, tree: BettingTree, compatibility, showdown, weights, backend: Backend):
        self.tree = tree
        self.backend = backend
        self.xp = backend.xp
        self.counts = (len(weights[0]), len(weights[1]))
        matrix = np.asarray(compatibility, dtype=np.float64)
        signs = np.asarray(showdown, dtype=np.float64)
        if matrix.shape != self.counts or signs.shape != self.counts:
            raise ValueError("payoff matrices do not match the private hand counts")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(signs)):
            raise ValueError("payoff matrices must contain finite numbers")
        if np.any((matrix < 0) | (matrix > 1)) or np.any(np.abs(signs) > 1.000001):
            raise ValueError("compatibility and showdown entries are outside their bounds")
        normalized = []
        for ws in weights:
            ws = np.asarray(ws, dtype=np.float64)
            if not len(ws) or np.any(~np.isfinite(ws)) or np.any(ws < 0) or ws.max() <= 0:
                raise ValueError("each player needs a finite, positive total range weight")
            scaled = ws / ws.max()
            probabilities = scaled / scaled.sum()
            if np.any((ws > 0) & (probabilities == 0)):
                raise ValueError("range weight ratios exceed the representable precision of this backend")
            normalized.append(probabilities)
        self.joint_mass = float(normalized[0] @ matrix @ normalized[1])
        if self.joint_mass <= 0:
            raise ValueError("the ranges contain no compatible heads-up hand pairs")
        if backend.name == "metal":
            # Metal float32 arithmetic can flush subnormal inputs to zero. A
            # lost chance distribution must be rejected rather than reported
            # as a solved game with zero exploitability.
            smallest = np.finfo(np.float32).tiny
            if self.joint_mass < smallest or any(np.any((w > 0) & (w < smallest)) for w in normalized):
                raise ValueError("range probabilities/blocker-conditioned mass are too small for Metal float32; use backend='cpu' or less extreme weights")
            if min(tree.pot, tree.stack) < smallest:
                raise ValueError("pot and stack are too small for Metal float32; use backend='cpu'")
        with backend.context():
            self.compatibility = backend.array(matrix)
            self.showdown = backend.array(matrix * signs)
            self.weights = tuple(backend.array(w) for w in normalized)
            self.regrets = {
                n.id: backend.zeros((self.counts[n.actor], len(n.actions)))
                for n in tree.nodes if n.actor is not None
            }
            self.strategy_sums = {key: backend.zeros(value.shape) for key, value in self.regrets.items()}
            backend.evaluate([self.compatibility, self.showdown, *self.weights])
        self.iterations = 0

    def _normalize(self, values):
        totals = self.xp.sum(values, axis=1, keepdims=True)
        # Safe denominators prevent invalid divisions even for unvisited hands.
        denominator = self.xp.where(totals > 0, totals, 1.0)
        return self.xp.where(totals > 0, values / denominator, 1.0 / values.shape[1])

    def current_policy(self) -> dict:
        return {key: self._normalize(value) for key, value in self.regrets.items()}

    def average_policy(self) -> dict:
        return {key: self._normalize(value) for key, value in self.strategy_sums.items()}

    def _terminal_value(self, node, player, opponent_reach, cache):
        # Own decisions preserve the same opponent reach. Reuse matrix products
        # across those branches rather than multiplying at every terminal.
        key = (player, id(opponent_reach))
        if key not in cache:
            weighted = self.weights[1 - player] * opponent_reach
            compatibility = self.compatibility if player == 0 else self.compatibility.T
            showdown = self.showdown if player == 0 else -self.showdown.T
            cache[key] = (opponent_reach, compatibility @ weighted, showdown @ weighted)
        _, compatible_mass, showdown_value = cache[key]
        if node.terminal == "showdown":
            return showdown_value * (self.tree.pot / 2 + node.committed[0])
        if node.terminal == "fold":
            sign = -1.0 if node.folded_player == player else 1.0
            return compatible_mass * sign * (self.tree.pot / 2 + node.committed[node.folded_player])
        raise RuntimeError("unknown terminal node")

    def _traverse(self, node, player, own_reach, opponent_reach, policy, weight, cache):
        if node.terminal:
            return self._terminal_value(node, player, opponent_reach, cache)
        strategy = policy[node.id]
        if node.actor == player:
            action_values = self.xp.stack([
                self._traverse(a.child, player, own_reach * strategy[:, i], opponent_reach, policy, weight, cache)
                for i, a in enumerate(node.actions)
            ], axis=1)
            value = self.xp.sum(action_values * strategy, axis=1)
            self.regrets[node.id] = self.xp.maximum(0, self.regrets[node.id] + action_values - value[:, None])
            self.strategy_sums[node.id] = self.strategy_sums[node.id] + weight * own_reach[:, None] * strategy
            return value
        action_values = [
            self._traverse(a.child, player, own_reach, opponent_reach * strategy[:, i], policy, weight, cache)
            for i, a in enumerate(node.actions)
        ]
        return self.xp.sum(self.xp.stack(action_values, axis=1), axis=1)

    def run(self, iterations: int) -> None:
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        with self.backend.context():
            for _ in range(iterations):
                self.iterations += 1
                for player in (0, 1):
                    policy = self.current_policy()
                    self._traverse(self.tree.root, player, self.backend.ones(self.counts[player]),
                                   self.backend.ones(self.counts[1 - player]), policy, self.iterations, {})
                # Evaluate every iteration so lazy GPU graphs remain bounded.
                self.backend.evaluate([*self.regrets.values(), *self.strategy_sums.values()])

    def _policy_value(self, node, player, opponent_reach, policy, cache):
        if node.terminal:
            return self._terminal_value(node, player, opponent_reach, cache)
        strategy = policy[node.id]
        if node.actor == player:
            values = self.xp.stack([
                self._policy_value(a.child, player, opponent_reach, policy, cache)
                for a in node.actions
            ], axis=1)
            return self.xp.sum(values * strategy, axis=1)
        values = [
            self._policy_value(a.child, player, opponent_reach * strategy[:, i], policy, cache)
            for i, a in enumerate(node.actions)
        ]
        return self.xp.sum(self.xp.stack(values, axis=1), axis=1)

    def _best_response(self, node, player, opponent_reach, policy, cache):
        if node.terminal:
            return self._terminal_value(node, player, opponent_reach, cache)
        strategy = policy[node.id]
        if node.actor == player:
            return self.xp.max(self.xp.stack([
                self._best_response(a.child, player, opponent_reach, policy, cache)
                for a in node.actions
            ], axis=1), axis=1)
        values = [
            self._best_response(a.child, player, opponent_reach * strategy[:, i], policy, cache)
            for i, a in enumerate(node.actions)
        ]
        return self.xp.sum(self.xp.stack(values, axis=1), axis=1)

    def diagnostics(self, policy: dict) -> dict:
        with self.backend.context():
            values = []
            best_responses = []
            for player in (0, 1):
                opponent_reach = self.backend.ones(self.counts[1 - player])
                value = self._policy_value(self.tree.root, player, opponent_reach, policy, {})
                best = self._best_response(self.tree.root, player, opponent_reach, policy, {})
                values.append(self.xp.sum(self.weights[player] * value) / self.joint_mass)
                best_responses.append(self.xp.sum(self.weights[player] * best) / self.joint_mass)
            self.backend.evaluate([*values, *best_responses])
            game_values = [float(x.item()) for x in values]
            br_values = [float(x.item()) for x in best_responses]
        if not all(math.isfinite(v) for v in game_values + br_values):
            raise RuntimeError("nonfinite solver diagnostics; the backend cannot represent this game's numerical scale")
        nashconv = max(0.0, sum(br_values) - sum(game_values))
        return {
            "root_ev": {"oop": game_values[0], "ip": game_values[1]},
            "best_response_ev": {"oop": br_values[0], "ip": br_values[1]},
            "nash_conv": nashconv,
            "exploitability": nashconv / 2,
            "units": "chips",
            "scope": "exact best responses to the returned average strategy in this finite action/payoff abstraction",
        }

    def query(self, node: Node, history: list[str], policy: dict, hands: tuple[list, list]) -> dict:
        with self.backend.context():
            reaches = [self.backend.ones(self.counts[0]), self.backend.ones(self.counts[1])]
            cursor = self.tree.root
            for label in history:
                i = next(i for i, a in enumerate(cursor.actions) if a.id == label)
                reaches[cursor.actor] = reaches[cursor.actor] * policy[cursor.id][:, i]
                cursor = cursor.actions[i].child
            mass = self.xp.sum((self.weights[0] * reaches[0]) * (self.compatibility @ (self.weights[1] * reaches[1])))
            joint_mass = float(mass.item())
            result = {
                "node_id": node.id,
                "history": history,
                "actor": None if node.actor is None else ("oop" if node.actor == 0 else "ip"),
                "committed": list(node.committed),
                "pot": self.tree.pot + sum(node.committed),
                "terminal": node.terminal,
                "history_probability": joint_mass / self.joint_mass,
                "reachable_under_average_strategy": joint_mass > 0,
                "actions": [a.describe() for a in node.actions],
            }
            if node.actor is None:
                result["folded_player"] = None if node.folded_player is None else ("oop" if node.folded_player == 0 else "ip")
                result["strategy"] = []
                result["aggregate_strategy"] = {}
                result["recommended_action"] = None
                return result
            player = node.actor
            opponent_mass = (self.compatibility if player == 0 else self.compatibility.T) @ (self.weights[1 - player] * reaches[1 - player])
            action_values = self.xp.stack([
                self._policy_value(a.child, player, reaches[1 - player], policy, {}) for a in node.actions
            ], axis=1)
            values = self.xp.sum(action_values * policy[node.id], axis=1)
            posterior = self.weights[player] * reaches[player] * opponent_mass
            self.backend.evaluate([opponent_mass, action_values, values, posterior, policy[node.id]])
            masses_np = self.backend.numpy(opponent_mass)
            action_np = self.backend.numpy(action_values)
            ev_np = self.backend.numpy(values)
            strategy_np = self.backend.numpy(policy[node.id])
            posterior_np = self.backend.numpy(posterior)
        action_ids = [a.id for a in node.actions]
        result["strategy"] = [
            {
                "hand": hand[0],
                "range_weight": float(hand[2]),
                "posterior_probability": float(posterior_np[i] / joint_mass) if joint_mass > 0 else None,
                "opponent_compatible": bool(masses_np[i] > 0),
                "probabilities": {label: float(strategy_np[i, j]) for j, label in enumerate(action_ids)},
                "ev": float(ev_np[i] / masses_np[i]) if masses_np[i] > 0 else None,
                "action_ev": {label: float(action_np[i, j] / masses_np[i]) if masses_np[i] > 0 else None
                              for j, label in enumerate(action_ids)},
                "recommended_action": action_ids[int(np.argmax(strategy_np[i]))] if masses_np[i] > 0 else None,
            }
            for i, hand in enumerate(hands[player])
        ]
        if joint_mass > 0:
            aggregate = np.sum(strategy_np * posterior_np[:, None], axis=0) / joint_mass
            result["aggregate_strategy"] = {label: float(aggregate[j]) for j, label in enumerate(action_ids)}
            result["ev"] = float(np.dot(posterior_np, np.divide(ev_np, masses_np, out=np.zeros_like(ev_np), where=masses_np > 0)) / joint_mass)
            result["recommended_action"] = action_ids[int(np.argmax(aggregate))]
        else:
            result["aggregate_strategy"] = None
            result["ev"] = None
            result["recommended_action"] = None
        result["recommendation_basis"] = "highest probability in the returned mixed strategy; EVs are continuation values against that policy"
        return result


def solve(
    board: list[str],
    oop_range: str | dict[str, float],
    ip_range: str | dict[str, float],
    pot: float,
    effective_stack: float,
    bet_sizes: list[float] | None = None,
    max_raises: int = 1,
    iterations: int = 1000,
    backend: str = "metal",
    samples: int = 256,
    seed: int = 0,
    history: list[str] | None = None,
    include_tree: bool = False,
) -> dict:
    """Solve a postflop, heads-up, single-street game from explicit inputs."""
    from .cards import normalize_board
    from .ranges import expand_range
    from .payoffs import build_payoffs

    start = time.perf_counter()
    board = normalize_board(board, min_cards=3, max_cards=5)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 10000:
        raise ValueError("iterations must be an integer between 1 and 10000")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 4096:
        raise ValueError("samples must be an integer between 1 and 4096")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if isinstance(max_raises, bool) or not isinstance(max_raises, int):
        raise ValueError("max_raises must be an integer between 0 and 2")
    sizes = [0.5] if bet_sizes is None else bet_sizes
    tree = BettingTree(pot, effective_stack, sizes, max_raises)
    history = [] if history is None else history
    if not isinstance(history, list) or any(not isinstance(x, str) for x in history):
        raise ValueError("history must be a list of action IDs")
    query_node = tree.resolve(history)
    numerical = Backend(backend)
    hands = (expand_range(oop_range, board), expand_range(ip_range, board))
    compatibility, showdown, payoff_metadata = build_payoffs(board, hands[0], hands[1], samples=samples, seed=seed)
    payoffs_elapsed = time.perf_counter() - start
    cfr = CFRSolver(tree, compatibility, showdown, tuple([h[2] for h in hs] for hs in hands), numerical)
    cfr.run(iterations)
    with numerical.context():
        average = cfr.average_policy()
        numerical.evaluate(list(average.values()))
    root_decision = cfr.query(tree.root, [], average, hands)
    query_decision = root_decision if query_node is tree.root else cfr.query(query_node, history, average, hands)
    result = {
        "solver": "heads_up_single_street_cfr_plus",
        "board": board,
        "backend": numerical.info,
        "iterations": iterations,
        "algorithm": "alternating CFR+; nonnegative cumulative regrets; linear iteration/realization weighted average",
        "combo_counts": {"oop": len(hands[0]), "ip": len(hands[1])},
        "compatible_pair_count": int(np.count_nonzero(compatibility)),
        "independent_range_compatible_mass": cfr.joint_mass,
        "node_count": len(tree.nodes),
        "payoff_model": payoff_metadata,
        "model": {
            "players": 2,
            "betting_rounds": 1,
            "future_betting": False,
            "action_abstraction": {"bet_sizes": sizes, "max_raises": max_raises, "all_in_included": True,
                                   "raise_fraction_basis": "pot_after_call", "minimum_full_raise_enforced": True},
            "ranges": "independent private range weights jointly conditioned on board and hole-card non-overlap",
            "utility": "zero-sum chips centered on half the initial pot; folds lose half the initial pot plus own committed chips",
            "equilibrium_claim": "finite-iteration approximate strategy; diagnostics apply to this action/payoff abstraction",
        },
        "diagnostics": cfr.diagnostics(average),
        "root": root_decision,
        "query": query_decision,
        "decision": query_decision,
        "limitations": [
            "Heads-up postflop only; one betting round is modeled.",
            "Configured pot-fraction actions plus all-in form a finite betting abstraction.",
            "Flop and turn continuations resolve at showdown without subsequent betting.",
            "Finite CFR+ iterations produce an approximate strategy, not an equilibrium guarantee.",
        ] + (["Flop showdown equity is sampled; reported exploitability measures the sampled payoff game."] if len(board) == 3 else []),
        "elapsed_seconds": 0.0,
        "payoff_preparation_seconds": payoffs_elapsed,
    }
    if include_tree:
        result["tree"] = tree.describe()
    result["elapsed_seconds"] = time.perf_counter() - start
    return result
