"""The model registry — the kernel of the Forecasting OS.

Models register themselves with the :func:`register` decorator and become
discoverable by name::

    @register("naive", family="baseline")
    class Naive(PerSeriesForecaster): ...

    model = get_model("naive")
    list_models()          # DataFrame of every installed model

Third-party packages plug in the same way; nothing in the engine is special-
cased to built-in models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseForecaster

FAMILIES = ("baseline", "statistical", "ml", "ensemble", "financial", "adapter")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cls: type
    family: str
    description: str


_REGISTRY: dict[str, ModelSpec] = {}


def register(name: str, family: str = "statistical", description: str | None = None):
    """Class decorator registering a forecaster under ``name``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, BaseForecaster)):
            raise TypeError(f"{cls!r} must be a BaseForecaster subclass")
        existing = _REGISTRY.get(name)
        if existing is not None and existing.cls.__qualname__ != cls.__qualname__:
            raise ValueError(
                f"model name {name!r} already registered by {existing.cls.__qualname__}"
            )
        desc = description or (cls.__doc__ or "").strip().splitlines()[0].strip() or name
        _REGISTRY[name] = ModelSpec(name=name, cls=cls, family=family, description=desc)
        cls.registry_name = name
        return cls

    return deco


def get_model(name: str, **params) -> BaseForecaster:
    """Instantiate a registered model by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"unknown model {name!r}; available: {available}")
    return _REGISTRY[name].cls(**params)


def list_models(family: str | None = None) -> pd.DataFrame:
    """All registered models as a DataFrame (name, family, class, description)."""
    specs = sorted(_REGISTRY.values(), key=lambda s: (s.family, s.name))
    if family is not None:
        specs = [s for s in specs if s.family == family]
    return pd.DataFrame(
        {
            "name": [s.name for s in specs],
            "family": [s.family for s in specs],
            "class": [s.cls.__name__ for s in specs],
            "description": [s.description for s in specs],
        }
    )
