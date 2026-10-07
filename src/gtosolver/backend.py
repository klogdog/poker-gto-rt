"""Numerical backends; Metal requests never silently fall back to CPU."""

from __future__ import annotations

from contextlib import nullcontext
from importlib.metadata import version
import platform

import numpy as np


class BackendUnavailable(RuntimeError):
    """The explicitly requested execution backend cannot run."""


class Backend:
    def __init__(self, name: str = "metal") -> None:
        if name not in {"cpu", "metal"}:
            raise ValueError("backend must be 'cpu' or 'metal'")
        self.name = name
        self.xp = np
        self.dtype = np.float64
        self.stream = None
        self.info: dict = {
            "requested": name,
            "actual": "cpu",
            "executed": "numpy_cpu",
            "device": platform.processor() or platform.machine(),
            "dtype": "float64",
            "metal_execution": False,
            "numpy_version": np.__version__,
            "operations": "counterfactual values, regret updates, strategy averaging and best responses run on NumPy CPU",
        }
        if name == "metal":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise BackendUnavailable("Metal requires an Apple silicon Mac; explicitly select backend='cpu' elsewhere.")
            try:
                import mlx.core as mx
            except ImportError as exc:
                raise BackendUnavailable("MLX is unavailable. Install the project's metal extra on an Apple silicon Mac.") from exc
            if not mx.metal.is_available():
                raise BackendUnavailable("MLX reports that the Metal backend is unavailable.")
            self.xp = mx
            self.dtype = mx.float32
            # Streams are created on the calling thread, including API worker threads.
            self.stream = mx.default_stream(mx.gpu)
            device = mx.device_info()
            self.info.update(
                executed="mlx_metal_gpu",
                actual="metal",
                device=device.get("device_name", device.get("name", "Apple Metal GPU")),
                dtype="float32",
                metal_execution=True,
                mlx_version=version("mlx"),
                device_info=device,
                operations="counterfactual matrix-vector values, regrets, average strategies and best responses run on MLX Metal GPU; card evaluation and payoff preparation run on CPU",
            )
            # This actual GPU operation verifies availability rather than reporting
            # success merely because a library was imported.
            with mx.stream(self.stream):
                probe = mx.sum(mx.array([1.0, 2.0], dtype=mx.float32))
                mx.eval(probe)
                if float(probe.item()) != 3.0:
                    raise BackendUnavailable("Metal execution probe failed.")

    def context(self):
        return self.xp.stream(self.stream) if self.name == "metal" else nullcontext()

    def array(self, value):
        return self.xp.array(value, dtype=self.dtype)

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.dtype)

    def ones(self, shape):
        return self.xp.ones(shape, dtype=self.dtype)

    def evaluate(self, arrays) -> None:
        if self.name == "metal":
            self.xp.eval(*arrays)

    def numpy(self, value) -> np.ndarray:
        if self.name == "metal":
            self.xp.eval(value)
        return np.asarray(value)


def backend_status(name: str = "metal") -> dict:
    """Run an execution probe and return device/version evidence."""
    return Backend(name).info
