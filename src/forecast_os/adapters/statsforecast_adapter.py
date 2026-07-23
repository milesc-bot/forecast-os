"""Optional Nixtla ``statsforecast`` backend adapter (import-guarded).

The adapter is constructible only when ``statsforecast`` is installed
(``pip install "forecast-os[nixtla]"``). It translates the shared panel
contract to a named ``statsforecast.models`` class; nothing is registered in
the global registry at import time — call :func:`register_adapters`.
"""

from __future__ import annotations

import pandas as pd

from ..core.base import BaseForecaster
from ..core.registry import register
from ..core.types import ID_COL, TIME_COL, validate_panel

try:  # pragma: no cover - depends on optional backend
    import statsforecast  # noqa: F401

    _HAS_STATSFORECAST = True
except ImportError:
    _HAS_STATSFORECAST = False

__all__ = ["StatsForecastAdapter", "register_adapters"]

_INSTALL_HINT = (
    'statsforecast is not installed; run pip install "forecast-os[nixtla]" '
    "to enable the StatsForecastAdapter"
)


def _infer_freq(df: pd.DataFrame):
    """Frequency argument for the backend: pandas freq string or integer step."""
    ds = df.loc[df[ID_COL] == df[ID_COL].iloc[0], TIME_COL]
    if pd.api.types.is_datetime64_any_dtype(ds):
        return pd.infer_freq(pd.DatetimeIndex(ds)) or "D"
    return 1


class StatsForecastAdapter(BaseForecaster):
    """Run a named ``statsforecast`` model through the forecast-os panel contract."""

    def __init__(self, model: str = "AutoARIMA", season_length: int = 1):
        if not _HAS_STATSFORECAST:
            raise ImportError(_INSTALL_HINT)
        self.model = model
        self.season_length = season_length

    def fit(self, df: pd.DataFrame) -> StatsForecastAdapter:
        df = validate_panel(df)
        from statsforecast import StatsForecast
        from statsforecast import models as sf_models

        cls = getattr(sf_models, self.model, None)
        if cls is None:
            raise ValueError(f"unknown statsforecast model {self.model!r}")
        try:
            instance = cls(season_length=self.season_length)
        except TypeError:  # model does not take season_length
            instance = cls()
        self._sf = StatsForecast(models=[instance], freq=_infer_freq(df))
        self._train_df = df
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        kwargs = {"level": sorted(int(lvl) for lvl in level)} if level else {}
        fc = self._sf.forecast(df=self._train_df, h=h, **kwargs)
        if ID_COL not in fc.columns:  # older versions index by unique_id
            fc = fc.reset_index()
        point_cols = [
            c for c in fc.columns
            if c not in (ID_COL, TIME_COL) and "-lo-" not in c and "-hi-" not in c
        ]
        rename = {point_cols[0]: "yhat"}
        for col in fc.columns:
            if "-lo-" in col:
                rename[col] = "lo-" + col.rsplit("-lo-", 1)[1]
            elif "-hi-" in col:
                rename[col] = "hi-" + col.rsplit("-hi-", 1)[1]
        fc = fc.rename(columns=rename)
        keep = [ID_COL, TIME_COL, "yhat"]
        keep += sorted(c for c in fc.columns if c.startswith(("lo-", "hi-")))
        return fc[keep].reset_index(drop=True)


def register_adapters() -> list[str]:
    """Register the statsforecast backend when installed; returns registered names."""
    if not _HAS_STATSFORECAST:
        return []
    register(
        "statsforecast",
        family="adapter",
        description="Nixtla statsforecast backend (optional dependency)",
    )(StatsForecastAdapter)
    return ["statsforecast"]
