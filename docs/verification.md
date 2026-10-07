# Local acceptance — 2026-10-07

This records the local acceptance before publication. The tested API ran at `http://127.0.0.1:8000`, with its interactive schema at `/docs`. Source is published in [klogdog/poker-gto-rt](https://github.com/klogdog/poker-gto-rt); this acceptance does not establish a hosted API deployment.

## Tested identity

- Apple M3 Pro, 18 GPU cores; host reports macOS 27.0 and Metal 4 support.
- Python 3.12 isolated `.venv`; MLX 0.32.3 / mlx-metal 0.32.3; NumPy 2.5.3.
- Version 0.1.0 installed as a standard wheel. Every installed package `.py` file was compared byte-for-byte with `src/gtosolver/`; no differences.
- Actual MLX GPU arithmetic probe, heads-up solver operations, and multiplayer batched table updates succeeded. Requests report `actual=metal`, `executed=mlx_metal_gpu`, `device=Apple M3 Pro`, and `metal_execution=true`.
- Source hashes and literal live HTTP responses are saved in ignored `artifacts/api-smoke.json`. They bind these measurements to the tested solver files; later edits require new installation/acceptance.

## Checks

`make install` completed, followed by **`make test`: 90 passed, 0 skipped** in the recorded local run. One Starlette TestClient deprecation warning concerns its development-only httpx adapter; no failed checks resulted.

The suite covers:

- Known Kuhn poker value `−1/18`, decreasing heads-up best-response gap, and actual CPU/Metal agreement on selected fixtures.
- Card encodings/rankings; range notation and stable weight normalization; public/private blockers; exact river/turn and seeded flop continuation.
- Three through six players; correct betting order/closure, min raises, raise caps, equal-stack all-ins, ties, and chip conservation.
- External-sampling multiplayer regret/average estimators, including an independently derived three-player average regression.
- Joint-card-conditioned root policy, action-conditioned history posterior, rare compatible ranges, untrained-hand flags, and zero sampled reach producing no aggregate recommendation or EV.
- Small chip units, large finite range weights, Metal precision rejection, JSON/schema errors, solver resource gating, and explicit unavailable-backend failure without fallback.

Python compilation and Git whitespace checks passed for implementation/docs. Upstream reference files remain byte-for-byte unchanged and were excluded from the whitespace check.

## Live installed-package HTTP smoke

All three example requests returned HTTP 200 and confirmed Metal execution. Each used seed 7, 1,000 solver iterations, and 128 payoff/EV samples; the six-player case capped raises at zero. Timings include local HTTP overhead and represent one run, not a benchmark distribution or speedup comparison.

| Example | Players | HTTP wall time | Reported result |
|---|---:|---:|---|
| `examples/heads-up-river.json` | 2 | 2.517 s | Abstract-game exploitability 0.000370 chips; centered root EVs sum to zero |
| `examples/multiplayer-river.json` | 3 | 1.664 s | 668 information sets, 33,346 visited nodes; sampled policy/EV with uncertainty |
| `examples/six-player-river.json` | 6 | 3.513 s | 328 information sets, 113,029 visited nodes; sampled policy/EV with uncertainty |

Reproduce against the running API:

```sh
.venv/bin/python scripts/smoke_api.py
```

The smoke script saves complete outputs and source hashes rather than just reporting backend availability.

## What this acceptance establishes

The explicit-input API, configured betting model, mathematical regressions, installed-package launch, and real Metal execution work on this Mac. CPU/Metal tests establish agreement for the selected fixtures; they do not establish universal numerical/performance parity.

The game remains a finite, one-round postflop abstraction with equal starting stacks. Heads-up best responses quantify the gap in that configured game; sampled flop payoffs add approximation. Multiplayer MCCFR supplies a learned average policy with sampling/coverage diagnostics, not a Nash certificate. Flop/turn continuation has no future betting, and full-hand/preflop/unequal-stack solving is outside this release.

## Upstream provenance

The supplied upstream main commit `ba81e9552b25a2a8399ed943d8eb8a8b8aa05536` has exactly two tracked files: README.md and LICENSE. Its README's claims of a completed solver and latency are not supported by published implementation files. The API here was independently implemented; the unchanged reference files and MIT license are retained under `reference/`.
