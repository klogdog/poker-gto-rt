from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cards import decode_card, encode_card


RangeSpec = str | dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]]


class Player(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=64)
    range: RangeSpec = Field(description="Exact combos or weighted Hold'em range notation")
    stack: float | None = Field(default=None, ge=0, le=1e9, allow_inf_nan=False,
                               description="Actual chips available at this round's start; zero remains eligible but cannot act. Defaults to effective_stack.")
    committed: float = Field(default=0.0, ge=0, le=1e9, allow_inf_nan=False,
                             description="Cumulative chips invested before this round; included in pot and used for side-pot eligibility.")

    @field_validator("range")
    @classmethod
    def bounded_range(cls, value: RangeSpec) -> RangeSpec:
        if isinstance(value, str) and (not value.strip() or len(value) > 32768):
            raise ValueError("Range must be nonempty and at most 32768 characters")
        if isinstance(value, dict) and (not value or len(value) > 1326):
            raise ValueError("Range must contain 1 to 1326 weighted entries")
        return value


class ActionOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    history: list[str] = Field(max_length=32, description="Exact public path from this round's root")
    to: float = Field(gt=0, le=1e9, description="Exact total commitment on this street, in chips")

    @field_validator("history")
    @classmethod
    def bounded_history(cls, value: list[str]) -> list[str]:
        if any(not action or len(action) > 64 for action in value):
            raise ValueError("Action labels must contain 1 to 64 characters")
        return value


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    board: list[str] = Field(min_length=3, max_length=5, description="Flop, turn, or river cards")
    players: list[Player] = Field(min_length=2, max_length=6, description="Active players in betting order; index 0 acts first")
    pot: float = Field(gt=0, le=1e9, description="Pot before the modeled betting round, in chips")
    effective_stack: float = Field(gt=0, le=1e9, description="Fallback starting stack for players without stack; per-player stack preserves unequal bankrolls")
    dead_contributions: list[Annotated[float, Field(ge=0, le=1e9, allow_inf_nan=False)]] = Field(
        default_factory=list, max_length=32,
        description="Each previously folded player's cumulative invested chips, included in pot; retain separate amounts to preserve side-pot layers.")
    bet_sizes: list[Annotated[float, Field(ge=0.05, le=10, allow_inf_nan=False)]] = Field(
        default_factory=lambda: [0.5], min_length=1, max_length=3,
        description="Pot fractions; all-in is also included. Raises use pot after calling.",
    )
    max_raises: int = Field(default=1, ge=0, le=2, description="Raises allowed beyond the opening bet")
    min_bet: float = Field(default=0.0, ge=0, le=1e9, description="Minimum opening bet and full raise increment, in chips; zero preserves unspecified-blind behavior")
    action_overrides: list[ActionOverride] = Field(default_factory=list, max_length=32,
        description="At most one exact additional target per reachable history. May extend the raise cap only at that node; future default branches remain bounded.")
    iterations: int = Field(default=1000, ge=1, le=10000)
    backend: Literal["metal", "cpu"] = "metal"
    samples: int = Field(default=128, ge=32, le=2048, description="Flop showdown samples or multiplayer EV evaluation deals")
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    history: list[str] = Field(default_factory=list, max_length=32, description="Action labels from a previous response, from the round's root")
    include_tree: bool = False

    @field_validator("board")
    @classmethod
    def legal_board(cls, value: list[str]) -> list[str]:
        encoded = [encode_card(card) for card in value]
        if len(set(encoded)) != len(encoded):
            raise ValueError("Board cards must be unique")
        return [decode_card(card) for card in encoded]

    @field_validator("bet_sizes")
    @classmethod
    def unique_sizes(cls, value: list[float]) -> list[float]:
        return sorted(set(value))

    @field_validator("history")
    @classmethod
    def bounded_history(cls, value: list[str]) -> list[str]:
        if any(not action or len(action) > 64 for action in value):
            raise ValueError("Action labels must contain 1 to 64 characters")
        return value

    @model_validator(mode="after")
    def distinct_names(self) -> "SolveRequest":
        if len({player.name for player in self.players}) != len(self.players):
            raise ValueError("Player names must be unique")
        return self
