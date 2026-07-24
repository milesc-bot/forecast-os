"""Tests for the import-guarded Nixtla ``statsforecast`` backend adapter.

Everything here runs offline with no backend installed. A fake ``statsforecast``
module (a ``FakeStatsForecast`` class plus a ``models`` namespace) is injected
into ``sys.modules`` and the module-level ``_HAS_STATSFORECAST`` flag is
monkeypatched True, mirroring the ``FakeNixtlaClient`` harness in
``test_timegpt_adapter.py``. The fake ``.forecast()`` returns a Nixtla-shaped
frame (``unique_id, ds, AutoARIMA, AutoARIMA-lo-XX, AutoARIMA-hi-XX``) so the
adapter's rename onto the ``(unique_id, ds, yhat, lo-{l}, hi-{l})`` contract is
exercised without touching the real SDK or the network.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecast_os.adapters import statsforecast_adapter as sf
from forecast_os.adapters.statsforecast_adapter import (
    StatsForecastAdapter,
    register_adapters,
)
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.registry import _REGISTRY


def _panel(freq: str = "D", n: int = 30) -> pd.DataFrame:
    ds = pd.date_range("2021-01-01", periods=n, freq=freq)
    frames = [
        pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.arange(n, dtype=float)})
        for uid in ("a", "b")
    ]
    return pd.concat(frames, ignore_index=True)


def _install_fake_statsforecast(monkeypatch, point_name: str = "AutoARIMA"):
    """Inject a fake ``statsforecast`` package and flip the import flag True.

    Returns the fake module. After ``fit()``, the adapter stores the created
    ``FakeStatsForecast`` on ``adapter._sf`` so tests can read its ``.calls``.
    """
    module = types.ModuleType("statsforecast")
    models_mod = types.ModuleType("statsforecast.models")

    class _Model:
        def __init__(self, season_length=None):
            self.season_length = season_length

    class _NoSeasonModel:
        # Mimics a statsforecast model that does not accept season_length,
        # so the adapter's TypeError fallback (cls()) is exercised.
        def __init__(self):
            pass

    models_mod.AutoARIMA = _Model
    models_mod.AutoETS = _Model
    models_mod.HistoricAverage = _NoSeasonModel

    class FakeStatsForecast:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq
            self.calls: list[dict] = []

        def forecast(self, df, h, level=None):
            self.calls.append({"df": df, "h": h, "level": level})
            ids = list(dict.fromkeys(df["unique_id"].tolist()))
            last = pd.Timestamp(pd.to_datetime(df["ds"]).max())
            future = pd.date_range(last, periods=h + 1, freq="D")[1:]
            rows = []
            for uid in ids:
                for i, ts in enumerate(future, start=1):
                    row = {"unique_id": uid, "ds": ts, point_name: 100.0 + i}
                    if level:
                        for lvl in level:
                            row[f"{point_name}-lo-{lvl}"] = 100.0 + i - lvl
                            row[f"{point_name}-hi-{lvl}"] = 100.0 + i + lvl
                    rows.append(row)
            return pd.DataFrame(rows)

    module.StatsForecast = FakeStatsForecast
    module.models = models_mod
    monkeypatch.setitem(sys.modules, "statsforecast", module)
    monkeypatch.setitem(sys.modules, "statsforecast.models", models_mod)
    monkeypatch.setattr(sf, "_HAS_STATSFORECAST", True)
    return module


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot/restore the global registry so registration tests can't leak."""
    before = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(before)


# -- fit / predict happy path --------------------------------------------------


def test_fit_returns_self(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter()
    assert adapter.fit(_panel()) is adapter


def test_fit_predict_maps_to_contract_columns(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter(model="AutoARIMA")
    out = adapter.fit(_panel()).predict(6, level=[80, 90])

    assert list(out.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert {"lo-80", "hi-80", "lo-90", "hi-90"} <= set(out.columns)
    assert (out.groupby("unique_id").size() == 6).all()
    assert set(out["unique_id"].unique()) == {"a", "b"}
    assert np.isfinite(out["yhat"]).all()

    # intervals bracket the point forecast and widen with the level.
    assert (out["lo-80"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["hi-80"]).all()
    assert (out["lo-90"] <= out["lo-80"]).all()
    assert (out["hi-90"] >= out["hi-80"]).all()

    # the validated (unique_id, ds, y) panel reached the backend forecast call.
    call = adapter._sf.calls[-1]
    assert list(call["df"].columns[:3]) == ["unique_id", "ds", "y"]
    assert call["h"] == 6
    assert call["level"] == [80, 90]


def test_yhat_carries_the_point_column_values(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    out = StatsForecastAdapter().fit(_panel()).predict(3)
    # the fake sets the point column to 100 + step for steps 1..h.
    for _, g in out.groupby("unique_id"):
        assert list(g["yhat"]) == [101.0, 102.0, 103.0]


def test_predict_without_level_returns_point_only(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter()
    out = adapter.fit(_panel()).predict(4)
    assert list(out.columns) == ["unique_id", "ds", "yhat"]
    # no level -> the backend is called without a level kwarg.
    assert adapter._sf.calls[-1]["level"] is None


def test_level_is_sorted_before_the_backend_call(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter()
    adapter.fit(_panel()).predict(3, level=[95, 80])
    assert adapter._sf.calls[-1]["level"] == [80, 95]


# -- frequency + model resolution ---------------------------------------------


def test_freq_inferred_and_passed_to_backend(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter()
    adapter.fit(_panel(freq="D"))
    assert adapter._sf.freq == "D"


def test_season_length_forwarded_to_model(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter(model="AutoETS", season_length=12)
    adapter.fit(_panel())
    assert adapter._sf.models[0].season_length == 12


def test_model_without_season_length_uses_fallback(monkeypatch):
    # HistoricAverage's __init__ rejects season_length; the adapter must fall
    # back to constructing the model with no arguments (no TypeError leak).
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter(model="HistoricAverage", season_length=7)
    out = adapter.fit(_panel()).predict(2)
    assert len(out) == 4


def test_unknown_model_raises_valueerror(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter(model="NopeNotAModel")
    with pytest.raises(ValueError, match="NopeNotAModel"):
        adapter.fit(_panel())


def test_older_backend_indexing_by_unique_id(monkeypatch):
    """When the backend returns unique_id as the index, the adapter resets it."""
    module = _install_fake_statsforecast(monkeypatch)

    class IndexedStatsForecast(module.StatsForecast):
        def forecast(self, df, h, level=None):
            fc = super().forecast(df, h, level=level)
            return fc.set_index("unique_id")

    monkeypatch.setattr(module, "StatsForecast", IndexedStatsForecast)
    out = StatsForecastAdapter().fit(_panel()).predict(3)
    assert list(out.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert set(out["unique_id"].unique()) == {"a", "b"}


# -- import guard --------------------------------------------------------------


def test_construction_without_statsforecast_raises_hint(monkeypatch):
    monkeypatch.setattr(sf, "_HAS_STATSFORECAST", False)
    with pytest.raises(ImportError, match=r"forecast-os\[nixtla\]"):
        StatsForecastAdapter()


# -- lifecycle -----------------------------------------------------------------


def test_predict_before_fit_raises(monkeypatch):
    _install_fake_statsforecast(monkeypatch)
    adapter = StatsForecastAdapter()
    with pytest.raises(ForecastOSError):
        adapter.predict(3)


# -- registration --------------------------------------------------------------


def test_register_adapters_noop_without_statsforecast(monkeypatch):
    monkeypatch.setattr(sf, "_HAS_STATSFORECAST", False)
    assert register_adapters() == []
    assert "statsforecast" not in _REGISTRY


def test_register_adapters_registers_when_present(monkeypatch):
    monkeypatch.setattr(sf, "_HAS_STATSFORECAST", True)
    assert register_adapters() == ["statsforecast"]
    spec = _REGISTRY["statsforecast"]
    assert spec.cls is StatsForecastAdapter
    assert spec.family == "adapter"
