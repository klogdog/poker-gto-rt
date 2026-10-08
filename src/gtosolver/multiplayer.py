"""External-sampling MCCFR for a single multiplayer Hold'em betting round.

Private deals and unfinished boards are sampled; betting continues only on the
supplied street. This is an imperfect-information regret solver, not an equity
recommendation heuristic. Multiplayer regret minimization is not a certificate
of a Nash equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from .backend import Backend
from .cards import decode_card, encode_card, evaluate_seven, normalize_board
from .ranges import expand_range, sample_joint_hands
from .action_overrides import ActionOverrides, exact_label, validate_history, validate_min_bet, validate_target
from .settlement import normalize_ledger, settlement_payouts


MAX_INFOSETS = 50_000
MAX_VISITED_NODES = 10_000_000
MAX_ACTIONS = 6
MAX_PUBLIC_NODES = 50_000


class MultiPlayerError(ValueError):
    """An invalid or excessively large multiplayer solve request."""


@dataclass(frozen=True)
class Action:
    label: str
    kind: str
    amount: float
    to: float
    pot_fraction: float | None = None

    def as_dict(self) -> dict:
        return {"label": self.label, "type": self.kind, "amount": self.amount,
                "to": self.to, "pot_fraction": self.pot_fraction}


@dataclass(frozen=True)
class State:
    contributions: tuple[float, ...]
    folded: tuple[bool, ...]
    pending: tuple[int, ...]
    last_raise: float = 0.0
    raises: int = 0
    history: tuple[str, ...] = ()
    acted_at: tuple[float | None, ...] = ()

    @property
    def terminal(self) -> bool:
        return sum(not value for value in self.folded) <= 1 or not self.pending

    @property
    def actor(self) -> int | None:
        return None if self.terminal else self.pending[0]


class BettingRound:
    """One betting round with individual stack ceilings and cumulative pots.

    Index zero acts first. Players already all-in retain their private cards
    and pot eligibility, but do not receive another betting decision.
    """

    def __init__(self, players: int, pot: float, stack: float,
                 bet_sizes: Sequence[float], max_raises: int,
                 min_bet: float = 0.0, action_overrides: Sequence[dict] | None = None,
                 stacks: Sequence[float] | None = None,
                 committed: Sequence[float] | None = None,
                 dead_contributions: Sequence[float] | None = None):
        self.players, self.pot, self.stack = players, float(pot), float(stack)
        self.stacks, self.prior_committed, self.dead_contributions = normalize_ledger(
            players, self.pot, self.stack, stacks, committed, dead_contributions)
        self.utility_scale = max(self.pot, *self.stacks)
        self.bet_sizes = tuple(sorted(set(float(value) for value in bet_sizes)))
        self.max_raises = max_raises
        self.min_bet = validate_min_bet(min_bet)
        self.overrides = ActionOverrides(action_overrides)
        self.public_histories: set[tuple[str, ...]] = set()
        pending = tuple(index for index, amount in enumerate(self.stacks) if amount > 0)
        self.initial = State((0.0,) * players, (False,) * players,
                             pending if len(pending) > 1 else (),
                             acted_at=(None,) * players)
        # Validate each target against its reachable state before allocating any
        # GPU tables. Shorter paths establish stable IDs for descendant paths.
        for history in sorted(self.overrides.targets, key=lambda path: (len(path), path)):
            state = self.query(history)
            actions = self.actions(state)
            target = self.overrides.targets[history]
            action = next((candidate for candidate in actions if candidate.to == target
                           and candidate.kind not in {"fold", "call", "check"}), None)
            if action is None:
                raise MultiPlayerError(f"Unused or unreachable action override history: {list(history)}")
            following = self.advance(state, action)
            # After a bet/raise, every remaining actor can call. Each remaining
            # standard full raise can start at most one further active cycle.
            # This validates closure before training, rather than discovering
            # an unreportable 33rd history label in a GPU-backed traversal.
            capable_seats = [index for index, folded in enumerate(following.folded)
                             if not folded and following.contributions[index] < self.stacks[index]]
            capable = len(capable_seats)
            cycles = max(0, self.max_raises - following.raises) if capable > 1 else 0
            active_others = max(0, capable - 1)
            longest_default = len(following.history) + len(following.pending) + cycles * active_others
            # Short all-ins can create further return-call cycles without
            # consuming the full-raise cap. At an exhausted cap, only ceilings
            # reachable through successive short increases need reserving.
            # Before the cap is exhausted, later full raises can bring any
            # remaining ceiling into short-raise range.
            short_actors = capable
            if not cycles:
                ceiling = max(following.contributions)
                minimum = max(self.min_bet, following.last_raise)
                reachable: set[float] = set()
                for target in sorted(set(self.stacks[index] for index in capable_seats)):
                    if target <= ceiling:
                        continue
                    if target >= ceiling + minimum:
                        break
                    reachable.add(target)
                    ceiling = target
                short_actors = sum(self.stacks[index] in reachable for index in capable_seats)
            longest_default += sum(max(0, capable - offset - 1) for offset in range(short_actors))
            if longest_default > 32:
                # The bound may include mutually exclusive full/short paths.
                # Check actual continuations only near the limit, retaining
                # valid 32-label paths while failing before GPU allocation.
                self._assert_history_can_finish(following)
        self.overrides.assert_used()

    def _assert_history_can_finish(self, state: State) -> None:
        if state.terminal:
            return
        if len(state.history) >= 32:
            raise MultiPlayerError("Action override cannot finish within the 32-label history limit")
        for action in self.actions(state):
            self._assert_history_can_finish(self.advance(state, action))

    def actions(self, state: State) -> tuple[Action, ...]:
        if state.terminal:
            return ()
        self.public_histories.add(state.history)
        if len(self.public_histories) > MAX_PUBLIC_NODES:
            raise MultiPlayerError(f"Action abstraction exceeds {MAX_PUBLIC_NODES} public nodes")
        if len(state.history) >= 32:
            raise MultiPlayerError("Betting history exceeds the 32-label limit before it can finish")
        actor = state.pending[0]
        contribution = state.contributions[actor]
        stack = self.stacks[actor]
        current_bet = max(state.contributions)
        call_to = min(current_bet, stack)
        call = max(0.0, call_to - contribution)
        current_pot = self.pot + sum(state.contributions)
        facing_bet = current_bet > contribution
        actions = ([Action("fold", "fold", 0.0, contribution),
                    Action("call", "call", call, call_to)] if facing_bet
                   else [Action("check", "check", 0.0, contribution)])
        opening = current_bet == 0
        override = self.overrides.target(state.history)
        minimum = self.min_bet if opening else max(self.min_bet, state.last_raise)
        acted_at = state.acted_at[actor] if state.acted_at else None
        reopened = acted_at is None or current_bet >= acted_at + minimum
        capable = sum(not folded and state.contributions[index] < self.stacks[index]
                      for index, folded in enumerate(state.folded))
        raise_eligible = reopened and capable > 1 and current_bet < stack
        may_raise = raise_eligible and (opening or state.raises < self.max_raises)
        short_all_in = raise_eligible and stack < current_bet + minimum
        if override is not None:
            if not raise_eligible:
                raise MultiPlayerError("Action override cannot raise when betting has not reopened or no opponent can wager")
            validate_target(override, current_bet, stack, minimum)
            self.overrides.mark_used(state.history)
        if may_raise or override is not None or short_all_in:
            used: set[float] = set()
            for fraction in self.bet_sizes if may_raise else ():
                increment = fraction * (current_pot + call)
                target = increment if opening else current_bet + increment
                if target < (minimum if opening else current_bet + minimum):
                    continue
                if target >= stack:
                    continue  # This actor's ceiling has one unambiguous all_in action.
                if target in used or target <= current_bet:
                    continue
                used.add(target)
                kind = "bet" if opening else "raise"
                label = f"{kind}_{fraction * 100:.17g}%"
                actions.append(Action(label, kind, target - contribution,
                                      target, fraction))
            if override is not None and override not in used and override < stack:
                kind = "bet" if opening else "raise"
                actions.append(Action(exact_label(not opening, override), kind,
                                      override - contribution, override,
                                      (override - current_bet) / (current_pot + call)))
            # A short all-in is legal even below the minimum full increment;
            # advance() preserves the earlier full raise and reopening rights.
            actions.append(Action("all_in", "all_in", stack - contribution, stack))
        if len(actions) > MAX_ACTIONS:
            raise MultiPlayerError("Action override exceeds the six-action budget; use at most two standard bet sizes")
        return tuple(actions)

    def advance(self, state: State, action: Action) -> State:
        if state.terminal or action not in self.actions(state):
            raise MultiPlayerError("Illegal betting action")
        if len(state.history) >= 32:
            raise MultiPlayerError("Betting history exceeds the 32-label limit")
        actor = state.pending[0]
        contributions = list(state.contributions)
        folded = list(state.folded)
        acted_at = list(state.acted_at or (None,) * self.players)
        last_raise, raises = state.last_raise, state.raises
        old_bet = max(contributions)
        if action.kind == "fold":
            folded[actor] = True
        else:
            contributions[actor] = action.to
        aggressive = action.to > old_bet
        if aggressive:
            increment = action.to - old_bet
            minimum = self.min_bet if old_bet == 0 else max(self.min_bet, last_raise)
            if action.to >= old_bet + minimum:
                last_raise = increment
                raises += int(old_bet > 0)
            order = tuple((actor + offset) % self.players
                          for offset in range(1, self.players))
            pending = tuple(index for index in order if not folded[index]
                            and contributions[index] < self.stacks[index])
        else:
            pending = tuple(index for index in state.pending[1:] if not folded[index]
                            and contributions[index] < self.stacks[index])
        # Checking retains the right to open/raise when action returns after an
        # incomplete opening all-in; it is not a prior response to that bet.
        acted_at[actor] = None if action.kind == "check" else max(contributions)
        # With one non-all-in seat remaining, only an outstanding call needs a
        # decision. No player can wager into a pot nobody else can contest.
        capable = [index for index, is_folded in enumerate(folded)
                   if not is_folded and contributions[index] < self.stacks[index]]
        if len(capable) <= 1:
            pending = tuple(index for index in pending
                            if contributions[index] < max(contributions))
        return State(tuple(contributions), tuple(folded), pending,
                     last_raise, raises, state.history + (action.label,), tuple(acted_at))

    def query(self, history: Sequence[str]) -> State:
        validate_history(history)
        state = self.initial
        for label in history:
            legal = self.actions(state)
            match = next((action for action in legal if action.label == label), None)
            if match is None:
                raise MultiPlayerError(f"Invalid history action {label!r} after "
                                       f"{list(state.history)}; legal actions are "
                                       f"{[action.label for action in legal]}")
            state = self.advance(state, match)
        return state

    def payoff(self, state: State, strengths: Sequence[int]) -> np.ndarray:
        if not state.terminal:
            raise MultiPlayerError("Payoff requires a terminal betting state")
        payouts = settlement_payouts(self.pot, self.prior_committed,
                                     state.contributions, state.folded,
                                     self.dead_contributions, strengths)
        return payouts - np.asarray(state.contributions, dtype=np.float64)


@dataclass
class _Info:
    state: State
    hand: tuple[int, int]
    actions: tuple[Action, ...]
    regret_updates: int = 0
    average_updates: int = 0


class _Tables:
    """GPU kernels perform regret matching and batched table updates.

    The host reads one policy snapshot per iteration and traverses the sampled
    tree. New infosets receive a uniform policy until the next snapshot. This
    makes an iteration use a fixed strategy and avoids thousands of tiny GPU
    synchronization calls.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.infos: list[_Info] = []
        self.index: dict[tuple[tuple[str, ...], tuple[int, int]], int] = {}
        with backend.context():
            self.regrets = backend.zeros((0, MAX_ACTIONS))
            self.averages = backend.zeros((0, MAX_ACTIONS))
        self.policy = np.empty((0, MAX_ACTIONS))
        self.regret_delta: dict[int, np.ndarray] = {}
        self.average_delta: dict[int, np.ndarray] = {}
        self.regret_matching_batches = 0
        self.update_batches = 0

    def row(self, state: State, hand: tuple[int, int], actions: tuple[Action, ...]) -> int:
        hand = tuple(sorted(hand))
        key = (state.history, hand)
        if key not in self.index:
            if len(self.infos) >= MAX_INFOSETS:
                raise MultiPlayerError(f"Solve exceeded {MAX_INFOSETS} information sets; "
                                       "reduce iterations, range size, or betting sizes")
            self.index[key] = len(self.infos)
            self.infos.append(_Info(state, hand, actions))
        return self.index[key]

    def snapshot(self) -> None:
        if not self.infos:
            return
        backend, xp = self.backend, self.backend.xp
        with backend.context():
            counts = backend.array([[len(info.actions)] for info in self.infos])
            mask = xp.arange(MAX_ACTIONS)[None, :] < counts
            positive = xp.maximum(self.regrets, 0) * mask
            sums = xp.sum(positive, axis=1, keepdims=True)
            denominator = xp.where(sums > 0, sums, 1)
            policy = xp.where(sums > 0, positive / denominator, mask / counts)
            backend.evaluate([policy])
            self.policy = backend.numpy(policy).astype(np.float64)
        if not np.all(np.isfinite(self.policy)):
            raise MultiPlayerError("Nonfinite regret table; reduce the betting or iteration budget")
        self.regret_matching_batches += 1

    def strategy(self, row: int) -> np.ndarray:
        count = len(self.infos[row].actions)
        if row >= len(self.policy):
            return np.full(count, 1.0 / count)
        result = self.policy[row, :count]
        # Float32 GPU policies may sum a few ulps away from one.
        return result / result.sum()

    @staticmethod
    def _add(destination: dict[int, np.ndarray], row: int, delta: np.ndarray) -> None:
        if row not in destination:
            destination[row] = np.zeros(MAX_ACTIONS)
        destination[row][:len(delta)] += delta

    def regret(self, row: int, delta: np.ndarray) -> None:
        self._add(self.regret_delta, row, delta)
        self.infos[row].regret_updates += 1

    def average(self, row: int, strategy: np.ndarray, weight: float) -> None:
        self._add(self.average_delta, row, weight * strategy)
        self.infos[row].average_updates += 1

    def commit(self) -> None:
        backend, xp = self.backend, self.backend.xp
        with backend.context():
            missing = len(self.infos) - self.regrets.shape[0]
            if missing:
                self.regrets = xp.concatenate([self.regrets, backend.zeros((missing, MAX_ACTIONS))])
                self.averages = xp.concatenate([self.averages, backend.zeros((missing, MAX_ACTIONS))])
            for attribute, updates in (("regrets", self.regret_delta),
                                       ("averages", self.average_delta)):
                if not updates:
                    continue
                indices = np.asarray(list(updates), dtype=np.int32)
                deltas = np.asarray(list(updates.values()))
                if not np.all(np.isfinite(deltas)):
                    raise MultiPlayerError("Importance-sampling weights overflowed; reduce the solve size")
                table = getattr(self, attribute)
                if backend.name == "metal":
                    table = table.at[xp.array(indices)].add(backend.array(deltas))
                else:
                    np.add.at(table, indices, deltas)
                setattr(self, attribute, table)
            backend.evaluate([self.regrets, self.averages])
        self.regret_delta.clear()
        self.average_delta.clear()
        self.update_batches += 1

    def average_policies(self) -> np.ndarray:
        if not self.infos:
            return np.empty((0, MAX_ACTIONS))
        backend, xp = self.backend, self.backend.xp
        with backend.context():
            counts = backend.array([[len(info.actions)] for info in self.infos])
            mask = xp.arange(MAX_ACTIONS)[None, :] < counts
            sums = xp.sum(self.averages, axis=1, keepdims=True)
            average = xp.where(sums > 0, self.averages / xp.where(sums > 0, sums, 1),
                               mask / counts)
            backend.evaluate([average])
            result = backend.numpy(average).astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise MultiPlayerError("Nonfinite average-strategy table; reduce the solve budget")
        return result


class _Trainer:
    def __init__(self, game: BettingRound, tables: _Tables, rng: np.random.Generator):
        self.game, self.tables, self.rng = game, tables, rng
        self.visited_nodes = 0
        self.public_states: dict[tuple[str, ...], State] = {}
        self.utility_scale = getattr(game, "utility_scale", max(game.pot, game.stack))

    def traverse(self, state: State, hands: Sequence[tuple[int, int]],
                 strengths: Sequence[int], traverser: int, reaches: np.ndarray,
                 iteration: int) -> float:
        self.visited_nodes += 1
        if self.visited_nodes > MAX_VISITED_NODES:
            raise MultiPlayerError(f"Solve exceeded {MAX_VISITED_NODES} visited nodes; "
                                   "reduce iterations or betting sizes")
        if state.terminal:
            # Positive scaling preserves strategies and keeps Metal float32
            # regrets usable for both microscopic and large chip denominations.
            return float(self.game.payoff(state, strengths)[traverser] / self.utility_scale)
        self.public_states.setdefault(state.history, state)
        actor = state.pending[0]
        actions = self.game.actions(state)
        row = self.tables.row(state, hands[actor], actions)
        strategy = self.tables.strategy(row)
        if actor == traverser:
            values = np.empty(len(actions))
            for offset, action in enumerate(actions):
                child_reaches = reaches.copy()
                child_reaches[actor] *= strategy[offset]
                values[offset] = self.traverse(self.game.advance(state, action), hands,
                                               strengths, traverser, child_reaches, iteration)
            node_value = float(np.dot(strategy, values))
            # Opponent/chance reach cancels against the external sampling
            # probability. Own reach does not enter counterfactual regret.
            self.tables.regret(row, values - node_value)
            return node_value
        # Traverser actions are enumerated. Other players' prior actions are
        # sampled, including this actor's own reach. Divide only by sampled
        # third-party reach, so E[update] = iteration * own_reach * strategy.
        # This is needed for >2 players; simply adding strategy is biased.
        third_party_reach = math.prod(float(reaches[index])
                                     for index in range(self.game.players)
                                     if index not in (actor, traverser))
        self.tables.average(row, strategy, iteration / third_party_reach)
        offset = int(self.rng.choice(len(actions), p=strategy))
        child_reaches = reaches.copy()
        child_reaches[actor] *= strategy[offset]
        return self.traverse(self.game.advance(state, actions[offset]), hands,
                             strengths, traverser, child_reaches, iteration)


def _hand_label(hand: Sequence[int]) -> str:
    return "".join(decode_card(card) for card in sorted(hand))


def _completed_deal(ranges, board: Sequence[int], rng: np.random.Generator):
    hands = sample_joint_hands(ranges, board, rng)
    used = set(board)
    for hand in hands:
        used.update(hand)
    final_board = list(board)
    missing = 5 - len(board)
    if missing:
        remaining = np.asarray([card for card in range(52) if card not in used])
        final_board.extend(int(card) for card in rng.choice(remaining, size=missing, replace=False))
    strengths = [evaluate_seven([*hand, *final_board]) for hand in hands]
    return hands, strengths


def _state_report(game: BettingRound, state: State, ranges, tables: _Tables,
                  average: np.ndarray, players: Sequence[dict], *, hand_rows: bool = True,
                  posterior: dict[tuple[int, int], float] | None = None) -> dict:
    result = {"history": list(state.history), "terminal": state.terminal,
              "player": state.actor, "player_name": players[state.actor]["name"] if not state.terminal else None,
              "contributions": list(state.contributions), "folded": list(state.folded),
              "last_full_raise": state.last_raise, "full_raises": state.raises,
              "acted_at": list(state.acted_at)}
    actions = game.actions(state)
    result["actions"] = [action.as_dict() for action in actions]
    if state.terminal:
        result.update(strategy={}, hand_strategies=[], recommendation=None)
        return result
    actor = state.pending[0]
    aggregate = np.zeros(len(actions))
    weight_scale = max(weight for _, _, weight in ranges[actor])
    total_weight = sum(weight / weight_scale for _, _, weight in ranges[actor])
    posterior_total = sum(posterior.values()) if posterior is not None else None
    covered_weight = 0.0
    rows = []
    for label, hand, weight in ranges[actor]:
        row = tables.index.get((state.history, tuple(sorted(hand))))
        trained = row is not None and tables.infos[row].regret_updates > 0
        averaged = row is not None and tables.infos[row].average_updates > 0
        policy = (average[row, :len(actions)] if averaged
                  else np.full(len(actions), 1.0 / len(actions)))
        policy = policy / policy.sum()
        normalized_input_weight = (weight / weight_scale) / total_weight
        aggregate_weight = (posterior.get(tuple(sorted(hand)), 0.0) / posterior_total
                            if posterior is not None and posterior_total > 0 else
                            normalized_input_weight if posterior is None else 0.0)
        aggregate += aggregate_weight * policy
        if trained and averaged:
            covered_weight += aggregate_weight
        if hand_rows:
            rows.append({"hand": label, "weight": weight,
                         "posterior_probability": aggregate_weight if posterior is not None else None,
                         "strategy": {action.label: float(value) for action, value in zip(actions, policy)},
                         "trained": trained, "averaged": averaged,
                         "regret_updates": tables.infos[row].regret_updates if row is not None else 0,
                         "average_updates": tables.infos[row].average_updates if row is not None else 0})
    reached = posterior is None or posterior_total > 0
    result["strategy"] = ({action.label: float(value) for action, value in zip(actions, aggregate)}
                          if reached else None)
    result["recommendation"] = actions[int(np.argmax(aggregate))].label if reached else None
    result["sampled_reach"] = reached
    result["trained_and_averaged_range_weight"] = covered_weight
    result["aggregate_weighting"] = ("compatible joint-deal frequencies weighted by average-policy history likelihood"
                                     if posterior is not None else "input range weights after public-card removal")
    if hand_rows:
        result["hand_strategies"] = rows
    return result


def _policy_for(state: State, hand: tuple[int, int], game: BettingRound,
                tables: _Tables, average: np.ndarray) -> np.ndarray:
    actions = game.actions(state)
    row = tables.index.get((state.history, tuple(sorted(hand))))
    if row is None or tables.infos[row].average_updates == 0:
        return np.full(len(actions), 1.0 / len(actions))
    policy = average[row, :len(actions)]
    return policy / policy.sum()


def _evaluate_policy(game: BettingRound, ranges, board, tables: _Tables,
                     average: np.ndarray, rng: np.random.Generator, samples: int,
                     players: Sequence[dict], start: State | None = None) -> tuple[dict, dict, dict, dict]:
    values = np.empty((samples, game.players))
    likelihoods = np.ones(samples)
    root_marginal: dict[tuple[int, int], float] = {}
    query_marginal: dict[tuple[int, int], float] = {}
    query = game.initial if start is None else start
    for sample in range(samples):
        hands, strengths = _completed_deal(ranges, board, rng)
        if game.initial.actor is not None:
            root_hand = tuple(sorted(hands[game.initial.actor]))
            root_marginal[root_hand] = root_marginal.get(root_hand, 0.0) + 1.0
        history_state = game.initial
        for label in query.history:
            actions = game.actions(history_state)
            policy = _policy_for(history_state, hands[history_state.pending[0]], game, tables, average)
            offset = next(index for index, action in enumerate(actions) if action.label == label)
            likelihoods[sample] *= policy[offset]
            history_state = game.advance(history_state, actions[offset])
        if not query.terminal:
            query_hand = tuple(sorted(hands[query.pending[0]]))
            query_marginal[query_hand] = query_marginal.get(query_hand, 0.0) + likelihoods[sample]
        state = query
        while not state.terminal:
            actions = game.actions(state)
            policy = _policy_for(state, hands[state.pending[0]], game, tables, average)
            offset = int(rng.choice(len(actions), p=policy))
            state = game.advance(state, actions[offset])
        values[sample] = game.payoff(state, strengths)
    total = float(likelihoods.sum())
    reached = total > 0
    normalized = likelihoods / total if reached else np.zeros(samples)
    utility_scale = game.utility_scale
    normalized_means = np.sum(normalized[:, None] * (values / utility_scale), axis=0)
    means = normalized_means * utility_scale
    # Delta-method standard errors for a self-normalized importance estimator;
    # scaling values first avoids overflow when squaring chip denominations.
    centered = values / utility_scale - normalized_means
    errors = np.sqrt(np.sum(normalized[:, None] ** 2 * centered ** 2, axis=0)
                     * samples / (samples - 1)) * utility_scale
    conditioning = {"method": "compatible joint deals weighted by average-policy history likelihood",
                    "estimated_history_probability": float(likelihoods.mean()),
                    "history_probability_standard_error": float(likelihoods.std(ddof=1) / math.sqrt(samples)),
                    "effective_sample_size": 1.0 / float(np.dot(normalized, normalized)) if reached else 0.0,
                    "sampled_reach": reached,
                    "scope": "posterior private-card distribution under the trained average policies; finite-sample estimate"}
    ev = {"per_player": [{"player": index, "name": player["name"],
                            "expected_value": float(means[index]) if reached else None,
                            "standard_error": float(errors[index]) if reached else None}
                           for index, player in enumerate(players)],
            "samples": samples, "method": "history-likelihood-weighted Monte Carlo average-policy rollout",
            "units": "same chip units as pot and effective_stack",
            "utility": "terminal pot receipt and unmatched return minus chips committed in this betting round; prior investments are sunk",
            "standard_error_scope": "approximate ratio-estimator sampling error; excludes strategy-training error",
            "history_conditioning": conditioning}
    return ev, root_marginal, query_marginal, conditioning


def solve(*, board: Sequence[str], players: Sequence[dict], pot: float,
          effective_stack: float, bet_sizes: Sequence[float] = (0.5,),
          max_raises: int = 1, iterations: int = 1000, backend: str = "metal",
          samples: int = 128, seed: int = 0, history: Sequence[str] | None = None,
          include_tree: bool = False, min_bet: float = 0.0,
          action_overrides: Sequence[dict] | None = None,
          dead_contributions: Sequence[float] | None = None) -> dict:
    """Solve 3–6 players on the supplied street and return average policies.

    The root is the beginning of this round, with zero new contributions and
    all players live. Optional player ``stack`` values are remaining chips;
    optional ``committed`` values are prior cumulative hand investments.
    ``history`` selects a descendant policy to report; it does not reconstruct
    earlier streets or change the supplied ranges.
    """
    started = time.perf_counter()
    if not 3 <= len(players) <= 6:
        raise MultiPlayerError("Multiplayer solving requires 3 to 6 players")
    if not math.isfinite(pot) or not math.isfinite(effective_stack) or pot <= 0 or effective_stack <= 0:
        raise MultiPlayerError("pot and effective_stack must be positive finite numbers")
    if not math.isfinite(pot + len(players) * effective_stack):
        raise MultiPlayerError("pot plus total stacks must fit a finite floating-point chip amount")
    if not 1 <= len(bet_sizes) <= 3 or any(not math.isfinite(value) or value <= 0 for value in bet_sizes):
        raise MultiPlayerError("Supply 1 to 3 positive finite pot-relative bet sizes")
    if not isinstance(max_raises, int) or not 0 <= max_raises <= 2:
        raise MultiPlayerError("max_raises must be an integer between 0 and 2")
    if not isinstance(iterations, int) or not 1 <= iterations <= 10_000:
        raise MultiPlayerError("iterations must be an integer between 1 and 10000")
    if not isinstance(samples, int) or not 2 <= samples <= 2048:
        raise MultiPlayerError("samples must be an integer between 2 and 2048")
    normalized_board = normalize_board(board, min_cards=3, max_cards=5)
    board_ids = [encode_card(card) for card in normalized_board]
    player_specs = [{"name": player.get("name", f"Player {index + 1}"), "range": player["range"],
                     "stack": effective_stack if player.get("stack") is None else player["stack"],
                     "committed": player.get("committed", 0.0)}
                    for index, player in enumerate(players)]
    ranges = [expand_range(player["range"], normalized_board) for player in player_specs]
    game = BettingRound(len(players), pot, effective_stack, bet_sizes, max_raises,
                        min_bet=min_bet, action_overrides=action_overrides,
                        stacks=[player["stack"] for player in player_specs],
                        committed=[player["committed"] for player in player_specs],
                        dead_contributions=dead_contributions)
    positive_amounts = [game.pot, *(amount for amount in game.stacks if amount > 0)]
    if backend == "metal" and (min(positive_amounts) < np.finfo(np.float32).tiny
                               or min(positive_amounts) / game.utility_scale < np.finfo(np.float32).tiny):
        raise MultiPlayerError("Chip amounts or their relative scale are below Metal float32 precision; "
                               "rescale the chip units or explicitly use backend='cpu'")
    query = game.query(history or ())
    # This preflight rejects mutually incompatible ranges before allocating the
    # numerical backend. Every subsequent sampled deal is also bounded.
    sample_joint_hands(ranges, board_ids, np.random.default_rng(seed), max_attempts=10_000)
    numerical_backend = Backend(backend)
    tables = _Tables(numerical_backend)
    rng = np.random.default_rng(seed)
    trainer = _Trainer(game, tables, rng)
    for iteration in range(1, iterations + 1):
        hands, strengths = _completed_deal(ranges, board_ids, rng)
        tables.snapshot()
        for traverser in range(len(players)):
            trainer.traverse(game.initial, hands, strengths, traverser,
                             np.ones(len(players)), iteration)
        tables.commit()
    average = tables.average_policies()
    eval_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C]))
    ev, root_marginal, query_marginal, conditioning = _evaluate_policy(
        game, ranges, board_ids, tables, average, eval_rng, samples, player_specs, start=query)
    root_report = _state_report(game, game.initial, ranges, tables, average, player_specs,
                                posterior=root_marginal)
    query_report = _state_report(game, query, ranges, tables, average, player_specs,
                                 posterior=query_marginal)
    query_report["history_conditioning"] = conditioning
    query_report["sampled_reach"] = conditioning["sampled_reach"]
    backend_info = dict(numerical_backend.info)
    backend_info["operations"] = ["regret matching", "batched regret updates",
                                  "importance-weighted linear strategy accumulation", "average-policy normalization"]
    backend_info["host_operations"] = ["card sampling", "exact seven-card ranking", "betting-tree traversal"]
    backend_info["regret_matching_batches"] = tables.regret_matching_batches
    backend_info["table_update_batches"] = tables.update_batches
    backend_info["regret_utility_scale"] = game.utility_scale
    result = {"algorithm": "external-sampling MCCFR with linear average strategies",
              "model": {"game": "multiplayer no-limit Hold'em", "player_count": len(players),
                        "street": {3: "flop", 4: "turn", 5: "river"}[len(board_ids)],
                        "betting_rounds": 1, "future_betting": False,
                        "showdown": "exact ranking on sampled deals" if len(board_ids) == 5
                        else "uniform sampled final-board continuation with no later betting",
                        "equal_initial_stacks": len(set(game.stacks)) == 1, "side_pots": True,
                        "starting_stacks": list(game.stacks),
                        "prior_committed": list(game.prior_committed),
                        "dead_contributions": list(game.dead_contributions),
                        "side_pot_accounting": "cumulative investments and current wagers, folded dead money, and unmatched returns",
                        "acting_order": [player["name"] for player in player_specs],
                        "minimum_opening_bet": (game.min_bet if game.min_bet else "smallest available supplied positive sizing or all-in; no blind unit supplied"),
                        "minimum_raise": "max(min_bet, previous full bet or raise increment); short all-in permitted",
                        "action_abstraction": {"bet_sizes": list(game.bet_sizes), "max_raises": max_raises,
                                               "min_bet": game.min_bet, "all_in_included": True,
                                               "exact_override_count": len(game.overrides.targets),
                                               "exact_overrides": "one additional exact target at each specified reachable history; a legal override node can extend the raise cap with that target and its all-in comparison"},
                        "range_sampling": "independent weighted ranges conditioned on disjoint cards"},
              "backend": backend_info, "players": [{"player": index, "name": player["name"],
                                                        "combos": len(ranges[index]),
                                                        "stack": game.stacks[index],
                                                        "committed": game.prior_committed[index]}
                                                       for index, player in enumerate(player_specs)],
              "root": root_report, "query": query_report, "ev": ev,
              "diagnostics": {"iterations": iterations, "training_deals": iterations,
                              "evaluation_deals": samples, "deals": iterations + samples,
                              "traversals": iterations * len(players),
                              "infosets": len(tables.infos), "public_nodes": len(trainer.public_states),
                              "visited_nodes": trainer.visited_nodes, "seed": seed,
                              "min_bet": game.min_bet, "exact_override_count": len(game.overrides.targets),
                              "starting_stacks": list(game.stacks),
                              "prior_committed": list(game.prior_committed),
                              "dead_contributions": list(game.dead_contributions),
                              "public_node_limit": MAX_PUBLIC_NODES,
                              "solve_wall_ms": (time.perf_counter() - started) * 1000},
              "limitations": ["Multiplayer regret minimization does not guarantee a Nash equilibrium; no exploitability certificate is computed.",
                              "Only the current betting round is modeled; flop and turn continuation samples final boards without future betting.",
                              "Training and reported EV use Monte Carlo deals even on the river; river hand ranking and terminal payouts are exact.",
                              "Unvisited or unaveraged hand information sets use an explicitly flagged uniform policy.",
                              "Reported EV standard errors cover rollout sampling only, not solver convergence or model error.",
                              "All supplied players start live with no new-round commitments; all-in players retain showdown eligibility but cannot act. Side pots require complete prior committed and folded dead-contribution ledgers; any unexplained root pot is shared dead money.",
                              "Aggregate policies and query EV use finite-sample compatible deals; history posteriors are weighted by trained average-policy action likelihoods.",
                              "Rare history queries can have low effective sample size; zero sampled reach returns no aggregate policy, recommendation, or EV.",
                              "Metal executes batched numerical table kernels; no GPU speedup or CPU numerical parity guarantee is claimed."]}
    if include_tree:
        states = list(trainer.public_states.values())
        result["tree"] = [{"history": list(state.history), "player": state.actor,
                           "player_name": player_specs[state.actor]["name"],
                           "contributions": list(state.contributions), "folded": list(state.folded),
                           "last_full_raise": state.last_raise, "full_raises": state.raises,
                           "acted_at": list(state.acted_at),
                           "actions": [action.as_dict() for action in game.actions(state)]}
                          for state in states[:500]]
        result["tree_truncated"] = len(states) > 500
        result["tree_scope"] = "visited nonterminal public states and legal actions, capped at 500; policies are reported at root and query"
    return result
