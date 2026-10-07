"""A bounded heads-up no-limit single-street action abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Action:
    id: str
    type: str
    amount: float
    to: float
    pot_fraction: float | None
    is_all_in: bool
    child: "Node"

    def describe(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "amount": self.amount,
            "to": self.to,
            "pot_fraction": self.pot_fraction,
            "is_all_in": self.is_all_in,
            "child_node_id": self.child.id,
        }


@dataclass
class Node:
    id: int
    actor: int | None
    committed: tuple[float, float]
    terminal: str | None = None
    folded_player: int | None = None
    actions: list[Action] = field(default_factory=list)


class BettingTree:
    """Opening bets are fractions of the pot; raises use the pot after calling.

    All-in is always included. A fraction that duplicates an all-in is merged.
    max_raises counts raises after the initial bet. A short all-in is legal,
    while an ordinary raise must be at least the previous full raise size.
    """

    def __init__(self, pot: float, stack: float, sizes: list[float], max_raises: int = 1, max_nodes: int = 512):
        if not math.isfinite(pot) or pot <= 0 or not math.isfinite(stack) or stack <= 0:
            raise ValueError("pot and effective_stack must be finite and positive")
        if pot > 1e9 or stack > 1e9:
            raise ValueError("pot and effective_stack must be at most 1e9 chips")
        if not 0 <= max_raises <= 2:
            raise ValueError("max_raises must be between 0 and 2")
        if not 1 <= len(sizes) <= 4 or any(not math.isfinite(s) or s <= 0 for s in sizes):
            raise ValueError("bet_sizes needs one to four positive finite pot fractions")
        self.pot = float(pot)
        self.stack = float(stack)
        self.sizes = sorted(set(float(s) for s in sizes))
        self.max_raises = max_raises
        self.max_nodes = max_nodes
        self.nodes: list[Node] = []
        self.root = self._build((0.0, 0.0), 0, 0, 0, 0.0)

    def _node(self, committed, actor=None, terminal=None, folded_player=None):
        if len(self.nodes) >= self.max_nodes:
            raise ValueError(f"action abstraction exceeds {self.max_nodes} nodes; reduce bet_sizes or max_raises")
        node = Node(len(self.nodes), actor, committed, terminal, folded_player)
        self.nodes.append(node)
        return node

    def _edge(self, node, action_id, action_type, target, fraction, child):
        actor = node.actor
        node.actions.append(Action(action_id, action_type, target - node.committed[actor], target, fraction,
                                   action_type in {"bet", "raise", "call"} and target >= self.stack, child))

    def _targets(self, committed, actor, facing_bet, last_raise):
        other = 1 - actor
        baseline = committed[other] if facing_bet else committed[actor]
        current_pot = self.pot + sum(committed)
        if facing_bet:
            current_pot += committed[other] - committed[actor]
        seen: set[float] = set()
        for fraction in self.sizes:
            target = min(self.stack, baseline + fraction * current_pot)
            if target <= baseline:
                continue
            all_in = target >= self.stack
            increment = target - baseline
            if facing_bet and increment < last_raise and not math.isclose(increment, last_raise, rel_tol=1e-10) and not all_in:
                continue
            identity = target
            if identity in seen:
                continue
            seen.add(identity)
            label = "all_in" if all_in else f"{'raise' if facing_bet else 'bet'}_{fraction * 100:.17g}%"
            yield label, target, None if all_in else fraction
        if self.stack > baseline and self.stack not in seen:
            yield "all_in", self.stack, None

    def _build(self, committed, actor, checks, raises, last_raise):
        other = 1 - actor
        node = self._node(committed, actor)
        facing = committed[other] > committed[actor]
        if facing:
            child = self._node(committed, terminal="fold", folded_player=actor)
            self._edge(node, "fold", "fold", committed[actor], None, child)
            called = list(committed)
            called[actor] = committed[other]
            child = self._node(tuple(called), terminal="showdown")
            self._edge(node, "call", "call", called[actor], None, child)
            if raises < self.max_raises and committed[other] < self.stack:
                for label, target, fraction in self._targets(committed, actor, True, last_raise):
                    raised = list(committed)
                    raised[actor] = target
                    child = self._build(tuple(raised), other, 0, raises + 1, target - committed[other])
                    self._edge(node, label, "raise", target, fraction, child)
        else:
            child = self._node(committed, terminal="showdown") if checks else self._build(committed, other, 1, raises, last_raise)
            self._edge(node, "check", "check", committed[actor], None, child)
            for label, target, fraction in self._targets(committed, actor, False, last_raise):
                bet = list(committed)
                bet[actor] = target
                child = self._build(tuple(bet), other, 0, raises, target - committed[actor])
                self._edge(node, label, "bet", target, fraction, child)
        return node

    def resolve(self, history: list[str]) -> Node:
        node = self.root
        for label in history:
            action = next((action for action in node.actions if action.id == label), None)
            if action is None:
                legal = [action.id for action in node.actions]
                raise ValueError(f"history action {label!r} is unavailable at node {node.id}; legal actions: {legal}")
            node = action.child
        return node

    def describe(self) -> dict:
        return {
            "root_node_id": self.root.id,
            "opening_actor": "oop",
            "all_in_included": True,
            "raise_fraction_basis": "pot_after_call",
            "max_raises": self.max_raises,
            "nodes": [
                {"id": n.id, "actor": None if n.actor is None else ("oop" if n.actor == 0 else "ip"),
                 "committed": list(n.committed), "terminal": n.terminal,
                 "folded_player": None if n.folded_player is None else ("oop" if n.folded_player == 0 else "ip"),
                 "actions": [a.describe() for a in n.actions]}
                for n in self.nodes
            ],
        }
