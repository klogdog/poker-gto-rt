from functools import lru_cache
from importlib.metadata import version
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException

from . import __version__
from .ranges import expand_range
from .schemas import SolveRequest


app = FastAPI(
    title="GTOSolver API",
    version=__version__,
    description=(
        "Explicit-input, 2–6 player Hold'em solver. Per-player stacks and contribution-ledger side pots, one betting round, "
        "pot-size action abstraction with optional exact wagers at specified public histories. "
        "Flop/turn continue to showdown without further betting. "
        "Heads-up exposes an abstract-game equilibrium gap; multiplayer strategies have no Nash guarantee."
    ),
)
_solve_lock = Lock()


@lru_cache(maxsize=1)
def metal_status() -> dict:
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return {"available": False, "reason": "MLX reports Metal unavailable"}
        with mx.stream(mx.gpu):
            probe = mx.array([[1.0, 2.0]], dtype=mx.float32) @ mx.array([[3.0], [4.0]], dtype=mx.float32)
            mx.eval(probe)
            if probe.item() != 11.0:
                raise RuntimeError("GPU execution probe returned an incorrect result")
        return {"available": True, "mlx_version": version("mlx"), "device": mx.device_info(), "gpu_probe_executed": True}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "cpu": {"available": True}, "metal": metal_status()}


@app.post("/v1/solve")
def solve(request: SolveRequest) -> dict:
    if request.backend == "metal" and not metal_status()["available"]:
        raise HTTPException(status_code=503, detail={"code": "metal_unavailable", "message": metal_status().get("reason")})
    if not _solve_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail={"code": "solver_busy", "message": "One solve is already running; retry after it finishes"})
    started = perf_counter()
    try:
        expanded = [expand_range(player.range, request.board) for player in request.players]
        if len(expanded) == 2:
            cells = len(expanded[0]) * len(expanded[1])
            if cells > 1_500_000 or (len(request.board) == 3 and cells * request.samples > 100_000_000):
                raise ValueError("Heads-up payoff work budget exceeded; narrow the ranges or reduce samples")
            from .solver import solve as solve_heads_up

            options = request.model_dump(exclude={"players"})
            result = solve_heads_up(oop_range=request.players[0].range, ip_range=request.players[1].range,
                                    stacks=[request.effective_stack if player.stack is None else player.stack for player in request.players],
                                    committed=[player.committed for player in request.players], **options)
            result["players"] = [{"index": i, "name": player.name, "combos": len(expanded[i]),
                                  "stack": request.effective_stack if player.stack is None else player.stack,
                                  "committed": player.committed} for i, player in enumerate(request.players)]
        else:
            from .multiplayer import solve as solve_multiplayer

            result = solve_multiplayer(**request.model_dump())
        result["request"] = request.model_dump()
        result.setdefault("diagnostics", {})["api_elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_game", "message": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "solver_unavailable", "message": str(exc)}) from exc
    finally:
        _solve_lock.release()
