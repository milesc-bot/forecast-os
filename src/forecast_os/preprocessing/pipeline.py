"""Chain preprocessing transforms with reversible inverse."""

from __future__ import annotations

import pandas as pd


class Pipeline:
    """Sequential container of (name, transform) steps.

    ``fit``/``transform``/``fit_transform`` apply steps in order;
    ``inverse_transform`` applies each step's inverse in reverse order.
    Step names must be unique non-empty strings.
    """

    def __init__(self, steps: list[tuple[str, object]]):
        steps = list(steps)
        names = []
        for step in steps:
            if not (isinstance(step, tuple) and len(step) == 2):
                raise ValueError(f"each step must be a (name, transform) tuple, got {step!r}")
            name, transform = step
            if not isinstance(name, str) or not name:
                raise ValueError(f"step names must be non-empty strings, got {name!r}")
            for method in ("fit", "transform", "inverse_transform"):
                if not callable(getattr(transform, method, None)):
                    raise ValueError(
                        f"step {name!r} transform must implement {method}()"
                    )
            names.append(name)
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate step name(s): {sorted(dupes)}")
        self.steps = steps

    @property
    def named_steps(self) -> dict[str, object]:
        return dict(self.steps)

    def fit(self, df: pd.DataFrame) -> Pipeline:
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for _, transform in self.steps:
            df = transform.fit(df).transform(df)
        self._is_fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for _, transform in self.steps:
            df = transform.transform(df)
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for _, transform in reversed(self.steps):
            df = transform.inverse_transform(df)
        return df
