"""A bounded heads-up adapter for the shared no-limit betting round."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .action_overrides import validate_history
from .multiplayer import BettingRound, State


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
    folded: tuple[bool, bool] = (False, False)


class BettingTree:
    """Materialize the shared betting rules for vector heads-up CFR.

    Each player retains their actual starting stack. Short calls/all-ins,
    full-raise reopening, and node-scoped exact targets use the same rules as
    multiplayer; the numerical solver still operates on a bounded finite tree.
    """

    def __init__(self, pot: float, stack: float, sizes: list[float], max_raises: int = 1, max_nodes: int = 512,
                 min_bet: float = 0.0, action_overrides: list[dict] | None = None,
                 stacks: Sequence[float] | None = None, committed: Sequence[float] | None = None,
                 dead_contributions: Sequence[float] | None = None):
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
        self.round = BettingRound(2, pot, stack, sizes, max_raises,
                                  min_bet=min_bet, action_overrides=action_overrides,
                                  stacks=stacks, committed=committed,
                                  dead_contributions=dead_contributions)
        self.min_bet = self.round.min_bet
        self.overrides = self.round.overrides
        self.stacks = self.round.stacks
        self.prior_committed = self.round.prior_committed
        self.dead_contributions = self.round.dead_contributions
        self.nodes: list[Node] = []
        self.root = self._build(self.round.initial)

    def _build(self, state: State) -> Node:
        if len(self.nodes) >= self.max_nodes:
            raise ValueError(f"action abstraction exceeds {self.max_nodes} nodes; reduce bet_sizes or max_raises")
        folded_player = next((index for index, folded in enumerate(state.folded) if folded), None)
        terminal = ("fold" if folded_player is not None else "showdown") if state.terminal else None
        node = Node(len(self.nodes), state.actor, tuple(state.contributions), terminal,
                    folded_player if terminal == "fold" else None, folded=tuple(state.folded))
        self.nodes.append(node)
        if terminal:
            return node
        for action in self.round.actions(state):
            child = self._build(self.round.advance(state, action))
            kind = action.kind
            if kind == "all_in":
                kind = "raise" if max(state.contributions) > 0 else "bet"
            node.actions.append(Action(action.label, kind, action.amount, action.to,
                                       action.pot_fraction,
                                       kind in {"bet", "raise", "call"} and action.to >= self.stacks[state.actor], child))
        return node

    def resolve(self, history: list[str]) -> Node:
        validate_history(history)
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
            "opening_actor": None if self.root.actor is None else ("oop" if self.root.actor == 0 else "ip"),
            "all_in_included": True,
            "raise_fraction_basis": "pot_after_call",
            "max_raises": self.max_raises,
            "min_bet": self.min_bet,
            "stacks": list(self.stacks),
            "prior_committed": list(self.prior_committed),
            "dead_contributions": list(self.dead_contributions),
            "action_overrides": [{"history": list(history), "to": target} for history, target in self.overrides.targets.items()],
            "nodes": [
                {"id": n.id, "actor": None if n.actor is None else ("oop" if n.actor == 0 else "ip"),
                 "committed": list(n.committed), "terminal": n.terminal,
                 "folded_player": None if n.folded_player is None else ("oop" if n.folded_player == 0 else "ip"),
                 "actions": [a.describe() for a in n.actions]}
                for n in self.nodes
            ],
        }
