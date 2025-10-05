"""backend.analysis package

Expose analysis submodules so code can import backend.analysis and access
the modules and commonly used functions/classes directly.

Example:
    from backend import analysis
    df = analysis.dataio.load_csv(...)
    model = analysis.model.TrainedModel(...)

This file purposely imports submodules lazily when possible to avoid heavy
startup costs; however for simplicity and backwards compatibility we import
the modules so `from backend.analysis import model` works.
"""
from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = [
    "dataio",
    "metrics",
    "model",
    "progress",
    "visualization",
]


def _import_submodule(name: str) -> ModuleType:
    """Import a submodule of backend.analysis by name and return it."""
    return import_module(f"backend.analysis.{name}")


# Import submodules and expose them as attributes on the package
dataio = _import_submodule("dataio")
metrics = _import_submodule("metrics")
model = _import_submodule("model")
progress = _import_submodule("progress")
visualization = _import_submodule("visualization")


# Re-export commonly used symbols (safe to change if you want to expose more)
try:
    load_csv = dataio.load_csv  # type: ignore[attr-defined]
except Exception:
    # Keep module-level attributes best-effort; some functions may not exist.
    load_csv = None  # type: ignore[assignment]

try:
    evaluate = metrics.evaluate  # type: ignore[attr-defined]
except Exception:
    evaluate = None  # type: ignore[assignment]
