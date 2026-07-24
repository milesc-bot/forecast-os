"""Optional Nixtla TimeGPT foundation-model adapter (import-guarded).

TimeGPT is a zero-shot forecasting foundation model served through Nixtla's
hosted API. The adapter is constructible only when ``nixtla`` is installed
(``pip install "forecast-os[timegpt]"``) — unless a client object is injected,
which is the seam used to exercise the adapter without the SDK or the network.

Because TimeGPT is zero-shot, :meth:`fit` performs no training: it validates
and retains the history and resolves the API client; the forecast itself is a
single hosted call made in :meth:`predict`. Nothing is registered in the global
registry at import time — call :func:`register_adapters`.
"""

from __future__ import annotations

import os

import pandas as pd

from ..core.base import BaseForecaster
from ..core.exceptions import ForecastOSError
from ..core.registry import register
from ..core.types import ID_COL, TIME_COL, validate_panel

try:  # pragma: no cover - depends on optional backend
    import nixtla  # noqa: F401
    from nixtla import NixtlaClient  # noqa: F401

    _HAS_NIXTLA = True
except ImportError:
    _HAS_NIXTLA = False

__all__ = ["TimeGPTAdapter", "register_adapters"]

_INSTALL_HINT = (
    'nixtla is not installed; run pip install "forecast-os[timegpt]" '
    "to enable the TimeGPTAdapter"
)

_NO_CREDENTIALS = (
    "TimeGPTAdapter needs Nixtla API credentials or an injected client: pass "
    "api_key=..., set the NIXTLA_API_KEY (or TIMEGPT_API_KEY) environment "
    "variable, or construct the adapter with client=..."
)


def _infer_freq(df: pd.DataFrame):
    """Frequency argument for TimeGPT: a pandas freq string or an integer step.

    Falls back to ``"D"`` for datetime series too short (< 3 points) for
    ``pd.infer_freq`` — pass ``freq=`` explicitly to override.
    """
    ds = df.loc[df[ID_COL] == df[ID_COL].iloc[0], TIME_COL]
    if pd.api.types.is_datetime64_any_dtype(ds):
        try:
            return pd.infer_freq(pd.DatetimeIndex(ds)) or "D"
        except ValueError:
            return "D"
    return 1


class TimeGPTAdapter(BaseForecaster):
    """Zero-shot forecasting via Nixtla's hosted TimeGPT foundation model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "timegpt-1",
        freq: str | None = None,
        client=None,
    ):
        # A caller-supplied client is the test/offline seam and needs no SDK;
        # otherwise the nixtla package must be importable to build one.
        if client is None and not _HAS_NIXTLA:
            raise ImportError(_INSTALL_HINT)
        self.api_key = api_key
        self.model = model
        self.freq = freq
        self.client = client

    def _resolve_client(self):
        """Return the injected client, or build a ``NixtlaClient`` from a key.

        The key is taken from ``api_key`` and falls back to the
        ``NIXTLA_API_KEY`` then ``TIMEGPT_API_KEY`` environment variables.
        Raises :class:`ForecastOSError` when neither a client nor any key is
        available.
        """
        if self.client is not None:
            return self.client
        api_key = (
            self.api_key
            or os.environ.get("NIXTLA_API_KEY")
            or os.environ.get("TIMEGPT_API_KEY")
        )
        if not api_key:
            raise ForecastOSError(_NO_CREDENTIALS)
        from nixtla import NixtlaClient

        return NixtlaClient(api_key=api_key)

    def fit(self, df: pd.DataFrame) -> TimeGPTAdapter:
        # Zero-shot: no training happens here. We validate + retain the panel
        # and resolve the client so a missing key fails fast, before predict().
        self._train_df = validate_panel(df)
        self._client = self._resolve_client()
        self._is_fitted = True
        return self

    def predict(
        self,
        h: int,
        level: list[int] | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast ``h`` steps for every fitted series via the hosted model.

        ``level`` requests prediction intervals (``lo-{l}/hi-{l}`` columns);
        ``X_df`` passes known future exogenous values through to the client.
        """
        self._check_is_fitted()
        freq = self.freq if self.freq is not None else _infer_freq(self._train_df)
        kwargs: dict = {"freq": freq, "model": self.model}
        if level:
            kwargs["level"] = sorted(int(lvl) for lvl in level)
        if X_df is not None:
            kwargs["X_df"] = X_df
        try:
            fc = self._client.forecast(df=self._train_df, h=h, **kwargs)
        except Exception as exc:  # wrap client/API/transport failures uniformly
            raise ForecastOSError(f"TimeGPT forecast call failed: {exc}") from exc
        return self._to_contract(fc)

    @staticmethod
    def _to_contract(fc: pd.DataFrame) -> pd.DataFrame:
        """Map a Nixtla forecast frame onto the (unique_id, ds, yhat, lo/hi) contract.

        Nixtla names the point column after the model (``TimeGPT``) and its
        intervals ``TimeGPT-lo-{level}`` / ``TimeGPT-hi-{level}``.
        """
        if ID_COL not in fc.columns:  # some SDK versions index by unique_id
            fc = fc.reset_index()
        point_cols = [
            c
            for c in fc.columns
            if c not in (ID_COL, TIME_COL) and "-lo-" not in c and "-hi-" not in c
        ]
        if not point_cols:
            raise ForecastOSError(
                f"TimeGPT response has no point-forecast column; got {list(fc.columns)}"
            )
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
    """Register the TimeGPT backend when nixtla is installed; returns the names."""
    if not _HAS_NIXTLA:
        return []
    register(
        "timegpt",
        family="adapter",
        description="Nixtla TimeGPT zero-shot foundation model (optional dependency)",
    )(TimeGPTAdapter)
    return ["timegpt"]
