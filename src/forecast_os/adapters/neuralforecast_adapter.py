"""Optional Nixtla ``neuralforecast`` backend adapter (import-guarded).

Constructible only when ``neuralforecast`` (torch-based) is installed
(``pip install "forecast-os[neural]"``). Because neuralforecast binds the
horizon at model construction, training is deferred to :meth:`predict`.
Nothing is registered in the global registry at import time — call
:func:`register_adapters`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..core.base import BaseForecaster
from ..core.registry import register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

try:  # pragma: no cover - depends on optional backend
    import neuralforecast  # noqa: F401

    _HAS_NEURALFORECAST = True
except ImportError:
    _HAS_NEURALFORECAST = False

__all__ = ["NeuralForecastAdapter", "register_adapters"]

_INSTALL_HINT = (
    'neuralforecast is not installed; run pip install "forecast-os[neural]" '
    "to enable the NeuralForecastAdapter"
)


def _infer_freq(df: pd.DataFrame):
    """Frequency argument for the backend: pandas freq string or integer step."""
    ds = df.loc[df[ID_COL] == df[ID_COL].iloc[0], TIME_COL]
    if pd.api.types.is_datetime64_any_dtype(ds):
        return pd.infer_freq(pd.DatetimeIndex(ds)) or "D"
    return 1


class NeuralForecastAdapter(BaseForecaster):
    """Run a named ``neuralforecast`` model through the forecast-os panel contract."""

    def __init__(self, model: str = "NHITS", max_steps: int = 100):
        if not _HAS_NEURALFORECAST:
            raise ImportError(_INSTALL_HINT)
        self.model = model
        self.max_steps = max_steps

    def fit(self, df: pd.DataFrame) -> NeuralForecastAdapter:
        # neuralforecast models bind ``h`` at construction, so we validate and
        # store the panel here and train inside predict().
        self._train_df = validate_panel(df)
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        from neuralforecast import NeuralForecast
        from neuralforecast import models as nf_models

        cls = getattr(nf_models, self.model, None)
        if cls is None:
            raise ValueError(f"unknown neuralforecast model {self.model!r}")
        nf = NeuralForecast(
            models=[cls(h=h, max_steps=self.max_steps)],
            freq=_infer_freq(self._train_df),
        )
        nf.fit(df=self._train_df)
        fc = nf.predict()
        if ID_COL not in fc.columns:  # older versions index by unique_id
            fc = fc.reset_index()
        point_cols = [
            c for c in fc.columns
            if c not in (ID_COL, TIME_COL) and "-lo-" not in c and "-hi-" not in c
        ]
        fc = fc.rename(columns={point_cols[0]: "yhat"})
        keep = [ID_COL, TIME_COL, "yhat"]
        keep += sorted(c for c in fc.columns if c.startswith(("lo-", "hi-")))
        out = fc[keep].reset_index(drop=True)
        if level:
            out = self._add_missing_intervals(out, h, level)
        return out

    def _add_missing_intervals(
        self, out: pd.DataFrame, h: int, level: list[int]
    ) -> pd.DataFrame:
        """Gaussian random-walk fallback intervals when the backend gives none.

        The default point losses of neuralforecast produce no quantiles; we
        approximate sigma per series from first differences of the training y.
        """
        missing = [lvl for lvl in level if f"lo-{int(lvl)}" not in out.columns]
        if not missing:
            return out
        sigmas = {
            uid: max(float(np.std(np.diff(g[TARGET_COL].to_numpy(dtype=float)))), 1e-12)
            for uid, g in self._train_df.groupby(ID_COL)
        }
        sigma = out[ID_COL].map(sigmas).to_numpy(dtype=float)
        step = out.groupby(ID_COL).cumcount().to_numpy() + 1
        spread = sigma * np.sqrt(step)
        for lvl in sorted(int(lvl) for lvl in missing):
            z = float(stats.norm.ppf(0.5 + lvl / 200))
            out[f"lo-{lvl}"] = out["yhat"] - z * spread
            out[f"hi-{lvl}"] = out["yhat"] + z * spread
        return out


def register_adapters() -> list[str]:
    """Register the neuralforecast backend when installed; returns registered names."""
    if not _HAS_NEURALFORECAST:
        return []
    register(
        "neuralforecast",
        family="adapter",
        description="Nixtla neuralforecast backend (optional dependency)",
    )(NeuralForecastAdapter)
    return ["neuralforecast"]
