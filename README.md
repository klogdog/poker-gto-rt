# GTOSolver API

An explicit-input Hold'em strategy solver for **2–6 active players**, with **Apple Metal as the default backend**. Send JSON; receive mixed action strategies, legal sizes, EVs, execution metadata, and model limits. No screenshots, vision, OCR, neural models, or table scraping are included.

The requested [poker-gto-rt repository](https://github.com/fabienpierret/poker-gto-rt) has only a README and license at commit `ba81e9552b25a2a8399ed943d8eb8a8b8aa05536`. It publishes no solver implementation. This project implements the solver/API independently; unchanged upstream reference files and its MIT license are in [`reference/`](reference/). No upstream latency or completeness claim is inherited.

## Run

Clone the public fork and install with Python 3.12:

```sh
git clone https://github.com/klogdog/poker-gto-rt.git
cd poker-gto-rt
make install
make serve
```

Re-run `make install` after modifying the solver source; startup runs the installed package. Tests read the workspace source. A standard wheel install avoids macOS hidden `.pth` files being skipped by Python's editable-install loader.

For an existing checkout, install dependencies and start the API with:

```sh
make install
make serve
```

- API: `http://127.0.0.1:8000`
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Machine-readable schema: `/openapi.json`
- `GET /health`: backend availability and an executed Metal GPU probe.
- `POST /v1/solve`: synchronous solve. One solve runs at a time; concurrent requests receive HTTP 429.
- Override address with `GTOSOLVER_HOST` / `GTOSOLVER_PORT`.

```sh
curl -sS http://127.0.0.1:8000/v1/solve \
  -H 'Content-Type: application/json' \
  --data-binary @examples/multiplayer-river.json
```

Apple silicon installation includes MLX. A Metal request fails with HTTP 503 when Metal cannot execute; it never silently switches to CPU. Use `"backend": "cpu"` explicitly for CPU execution. The tested MLX wheel is 0.32.3; wheel/OS compatibility matters on another machine. [`requirements-lock.txt`](requirements-lock.txt) records the tested environment's complete dependency versions; `pyproject.toml` pins direct dependencies.

## Inputs

```json
{
  "board": ["Ks", "Qh", "Jd", "7c", "2s"],
  "players": [
    {"name": "OOP", "range": "ATs,KQs,JJ"},
    {"name": "Middle", "range": "AQs,TT,99"},
    {"name": "IP", "range": "AA,AKs,QJs"}
  ],
  "pot": 100,
  "effective_stack": 200,
  "dead_contributions": [],
  "bet_sizes": [0.5],
  "max_raises": 1,
  "min_bet": 3,
  "action_overrides": [],
  "iterations": 1000,
  "backend": "metal",
  "samples": 128,
  "seed": 7,
  "history": [],
  "include_tree": false
}
```

| Field | Meaning |
|---|---|
| `board` | 3, 4, or 5 distinct cards; `T` represents ten |
| `players` | 2–6 active players in betting order; index 0 acts first |
| `range` | A string or `{ "AsAh": 1, "AKs": 0.5 }` map of positive relative weights |
| `players[].stack` | Actual chips available at this round's start. Optional; defaults to `effective_stack`. Zero means already all-in, with cards/pot eligibility retained. |
| `players[].committed` | Cumulative chips invested before this round, included in `pot`. Optional; defaults to zero. |
| `pot` | Pot before this modeled betting round, including supplied prior contributions and external dead money, in chips |
| `effective_stack` | Positive fallback stack for players without an explicit `stack`; does not clip explicit unequal stacks |
| `dead_contributions` | Prior cumulative investment of each already folded player, included in `pot`; optional `[]`. Keep each player's amount separate to preserve pot layers. |
| `bet_sizes` | 1–3 pot fractions; all-in is also included; raises use the pot after calling |
| `max_raises` | 0–2 raises beyond the opening bet |
| `min_bet` | Minimum opening bet and full raise increment in chips; default `0` preserves unspecified-blind behavior. A short all-in remains legal. |
| `action_overrides` | Up to 32 exact chip targets as `{ "history": [], "to": 17.25 }`; one target per reachable public history. A target is the total street commitment, not the additional chips paid. |
| `iterations` | 1–10000; finite iterations do not establish convergence |
| `samples` | 32–2048; heads-up flop payoff samples, or multiplayer policy EV evaluation deals |
| `seed` | Deterministic random seed; CPU/GPU numerical rounding can change sampled trajectories |
| `history` | Public action labels from this round's root, e.g. `["check", "bet_50%"]` |
| `include_tree` | Heads-up full finite tree; multiplayer visited public nodes capped at 500 |

Ranges support `AsKh`, `AA`, `AKs`, `AKo`, `AK`, `QQ+`, `AJs+`, `99-JJ`, `A2s-A5s`, and `random` / `*`. Separate tokens with commas or spaces and optionally add `:weight`. Overlapping tokens use the **maximum** weight. Board-blocked combinations are removed; incompatible joint ranges, bad histories, and excessive work budgets are rejected with HTTP 422. Extremely skewed weights outside the Metal solver's numerical precision are rejected rather than returned as invalid strategies.

For an exact known private hand, supply a single combo as that player's range. Other players can have uncertain weighted ranges. Individual hand policies are more informative than a range-level recommendation.

To evaluate a wager outside the configured pot fractions, append an `action_overrides` entry for the decision's exact public history and re-solve the same round root. Use `bet_sizes: [0.5, 1]` with an extra target so a facing-bet node fits the six-action limit. For example, an opening `17.25` target has label `bet_to_17.25`; subsequent requests retain that override and use the returned label in `history`. Exact labels use 17 significant digits, and a target already present in the abstraction keeps its existing label (`bet_50%`, `raise_100%`, or `all_in`). Never manufacture a label by rounding a chip size into a fraction.

An override can add a legal full raise at its specified node beyond `max_raises`; default branches at later nodes retain the configured raise cap. The changed abstraction is trained again, so policies and EVs can change. Keep prior overrides and the original board, pot, starting stacks, ranges, sizes, and seed when replaying an accepted history. Overrides are validated against reachable histories before training: duplicates, terminal or unreachable paths, under-minimum non-all-in targets, out-of-stack targets, and excessive action/depth/node budgets are rejected with HTTP 422. Paths must be able to finish within 32 action labels. Heads-up trees are capped at 512 nodes; multiplayer has bounded public nodes, information sets, and traversal work. This adapts a finite action abstraction to exact observed bets; it does not enumerate every possible legal wager simultaneously.

For a continuing hand, freeze each contestant's own starting wallet and prior cumulative investment when the street begins. Send that player's `stack` and `committed` on every request for the street, plus `dead_contributions` for players who folded before it began. For example, prior investments `[10, 30, 30]` and `pot: 70` create a main pot of `30` eligible to all three and a side pot of `40` eligible only to the latter two. The first player may have `stack: 0`; they remain in range conditioning and showdown but are skipped in betting order. A previously folded contribution `[30]` is divided across those same investment levels rather than added entirely to the main pot. Any `pot` beyond the supplied prior ledger is shared external dead money. Missing prior ledger fields preserve the older shared-pot model and cannot describe existing side-pot eligibility.

Calls pay at most the actor's own remaining stack and retain the `call` label/type, including a short all-in call. All-in bets/raises use the actor's own ceiling. A short raise preserves the last full increment and does not reopen a previously acted player's raising rights until the cumulative faced increase reaches that full increment. A player who only checked before a short opening bet keeps the right to raise. When only one player still has chips, betting ends unless that player owes a call. Terminal payouts use cumulative contribution layers, folded dead money, split pots, and uncalled returns; wallets are never trimmed to the shortest stack.

## Solver scope and results

This release solves **one postflop betting round with a finite action abstraction**. Every supplied player starts live with zero new-round commitments; earlier all-ins retain their pot eligibility. Prior betting is represented by the supplied pot, ranges, and optional cumulative contribution ledger. A history chooses a descendant of that round. There is no preflop, later-street betting, rake, or tournament utility model. Adding those changes the game being solved.

- **Two players:** Alternating vector CFR+, linear realization-weighted average strategy, blocker-conditioned ranges, and exact best-response exploitability within the configured finite game. River payoffs are exact; turn enumerates all legal rivers; flop uses seeded sampled runouts. Both flop and turn end at showdown without future betting. The gap on a sampled flop applies to the sampled payoff model.
- **Three to six players:** External-sampling MCCFR with signed cumulative regrets, regret matching, and importance-weighted linear average strategies. Joint private cards and remaining board cards are sampled. River hand ranking/payouts are exact for each sampled deal, but range EV/training are still sampled. **Multiplayer CFR does not certify a Nash equilibrium**, and this API makes no full-game GTO or exploitability claim for multiplayer output.
- **Metal:** MLX GPU executes heads-up vector CFR operations and multiplayer batched regret/strategy table operations. Card parsing, sampling, hand evaluation, and multiplayer tree traversal run on the host. Backend/device/kernel-batch evidence accompanies each result. No speedup claim is made.

Heads-up responses expose `root`, `query` / `decision`, per-hand `probabilities` and continuation `action_ev`, plus `diagnostics.exploitability` in chips. Heads-up EV is centered by subtracting half the initial pot, so root values sum to zero. Multiplayer responses expose `root`, `query`, per-hand `hand_strategies`, training coverage, and sampled `ev.per_player`. Multiplayer EV is terminal pot receipt minus this round's commitments, so player EVs sum to the initial pot; it has a different additive normalization from heads-up EV. Standard errors describe sampling uncertainty, not strategy-training or model error.

Recommendations select the highest probability action from the returned mixed policy. A mixed strategy should retain its action probabilities; a single recommendation does not replace them. Unvisited multiplayer information sets use a flagged uniform policy. Query sampling quality and unsupported-model details are reported in the response.

## Verification

```sh
make test
.venv/bin/python scripts/smoke_api.py
```

Tests include known Kuhn poker equilibrium/convergence, seven-card rankings, blockers, weighted sampling, legal betting/raises, chip conservation, multiplayer regret/average updates, explicit backend errors, HTTP validation, and actual Metal execution/parity cases. The smoke script exercises live HTTP with two, three, and six players and saves responses plus source hashes to ignored `artifacts/api-smoke.json`.

Exact-size tests also cover two, three, and six-player paths; minimum opening/full raises and short all-ins; targets outside the fraction schema's limits; preserved action IDs and exact amounts; node-scoped raises beyond the default cap; rejected ambiguous, unreachable, and oversized requests; and CPU HTTP integration. Metal uses the same betting abstraction with unchanged numerical kernels.

Unequal-stack tests cover individual wager ceilings, short calls, no-bet lone actors, previous all-ins, layered prior and new contributions, folded dead money, uncalled refunds, split pots, centered heads-up and multiplayer utility conservation, exact one-ULP raise boundaries, and actual Metal execution with two, three, and six contestants.

See [`docs/verification.md`](docs/verification.md) for the completed local acceptance evidence and its limits. Foundational algorithm references: [CFR](https://www.johanson.ca/publications/poker/2007-nips-cfr/2007-nips-cfr.html), [external-sampling MCCFR](https://www.cs.cmu.edu/~kwaugh/publications/nips09b.pdf), and [MLX](https://github.com/ml-explore/mlx).
