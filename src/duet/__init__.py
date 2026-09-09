"""Duet's stable public surface, resolved lazily for lightweight tooling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FusionSpec": ("duet.fusion", "FusionSpec"),
    "MatrixSemigroupFusion": ("duet.fusion", "MatrixSemigroupFusion"),
    "TamariFusion": ("duet.fusion", "TamariFusion"),
}

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
