"""Tests for the import-guarded Nixtla ``neuralforecast`` backend adapter.

Everything here runs offline with no torch/neuralforecast installed. A fake
``neuralforecast`` module (a ``FakeNeuralForecast`` class plus a ``models``
namespace) is injected into ``sys.modules`` and the module-level
``_HAS_NEURALFORECAST`` flag is monkeypatched True, mirroring the
``FakeNixtlaClient`` harness in ``test_timegpt_adapter.py``. Because
neuralforecast binds the horizon at construction, the adapter defers training
to :meth:`predict`; the fake's ``.fit()/.predict()`` return a Nixtla-shaped
frame so the rename onto the ``(unique_id, ds, yhat, lo-{l}, hi-{l})`` contract
— and the Gaussian fallback for point-only backends — are both exercised.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecast_os.adapters import neuralforecast_adapter as nf
from forecast_os.adapters.neuralforecast_adapter import (
    NeuralForecastAdapter,
    register_adapters,
)
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.registry import _REGISTRY


def _panel(freq: str = "D", n: int = 40) -> pd.DataFrame:
    ds = pd.date_range("2021-01-01", periods=n, freq=freq)
    frames = [
        pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.arange(n, dtype=float)})
        for uid in ("a", "b")
    ]
    return pd.concat(frames, ignore_index=True)


def _noisy_panel(n: int = 40) -> pd.DataFrame:
    """Panel whose first differences have positive std (for fallback intervals)."""
    rng = np.random.default_rng(0)
    ds = pd.date_range("2021-01-01", periods=n, freq="D")
    frames = [
        pd.DataFrame(
            {"unique_id": uid, "ds": ds, "y": 100.0 + np.cumsum(rng.normal(size=n))}
        )
        for uid in ("a", "b")
    ]
    return pd.concat(frames, ignore_index=True)


def _install_fake_neuralforecast(
    monkeypatch, emit_intervals=False, recorder=None, point_name="NHITS"
):
    """Inject a fake ``neuralforecast`` package and flip the import flag True.

    ``emit_intervals`` toggles whether ``.predict()`` returns interval columns
    (the realistic default point-loss models return none). ``recorder`` is an
    optional dict the fake fills with the created backend and its training
    frame, since the adapter constructs the backend internally.
    """
    module = types.ModuleType("neuralforecast")
    models_mod = types.ModuleType("neuralforecast.models")

    class _Model:
        def __init__(self, h, max_steps=None, **kw):
            self.h = h
            self.max_steps = max_steps

    models_mod.NHITS = _Model
    models_mod.NBEATS = _Model

    class FakeNeuralForecast:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq
            self._df = None
            if recorder is not None:
                recorder["nf"] = self

        def fit(self, df):
            self._df = df
            if recorder is not None:
                recorder["fit_df"] = df
            return self

        def predict(self):
            df = self._df
            h = self.models[0].h
            ids = list(dict.fromkeys(df["unique_id"].tolist()))
            last = pd.Timestamp(pd.to_datetime(df["ds"]).max())
            future = pd.date_range(last, periods=h + 1, freq="D")[1:]
            rows = []
            for uid in ids:
                for i, ts in enumerate(future, start=1):
                    row = {"unique_id": uid, "ds": ts, point_name: 50.0 + i}
                    if emit_intervals:
                        for lvl in (80, 90):
                            row[f"{point_name}-lo-{lvl}"] = 50.0 + i - lvl
                            row[f"{point_name}-hi-{lvl}"] = 50.0 + i + lvl
                    rows.append(row)
            return pd.DataFrame(rows)

    module.NeuralForecast = FakeNeuralForecast
    module.models = models_mod
    monkeypatch.setitem(sys.modules, "neuralforecast", module)
    monkeypatch.setitem(sys.modules, "neuralforecast.models", models_mod)
    monkeypatch.setattr(nf, "_HAS_NEURALFORECAST", True)
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
    _install_fake_neuralforecast(monkeypatch)
    adapter = NeuralForecastAdapter()
    assert adapter.fit(_panel()) is adapter


def test_fit_predict_maps_backend_intervals_to_contract(monkeypatch):
    recorder: dict = {}
    _install_fake_neuralforecast(monkeypatch, emit_intervals=True, recorder=recorder)
    adapter = NeuralForecastAdapter(model="NHITS")
    out = adapter.fit(_panel()).predict(6, level=[80, 90])

    assert list(out.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert {"lo-80", "hi-80", "lo-90", "hi-90"} <= set(out.columns)
    assert (out.groupby("unique_id").size() == 6).all()
    assert set(out["unique_id"].unique()) == {"a", "b"}
    assert np.isfinite(out["yhat"]).all()

    # backend-supplied intervals bracket the point forecast and widen by level.
    assert (out["lo-80"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["hi-80"]).all()
    assert (out["lo-90"] <= out["lo-80"]).all()
    assert (out["hi-90"] >= out["hi-80"]).all()

    # the validated (unique_id, ds, y) panel reached the backend fit call.
    assert list(recorder["fit_df"].columns[:3]) == ["unique_id", "ds", "y"]
    assert recorder["nf"].models[0].h == 6


def test_yhat_carries_the_point_column_values(monkeypatch):
    _install_fake_neuralforecast(monkeypatch)
    out = NeuralForecastAdapter().fit(_panel()).predict(3)
    # the fake sets the point column to 50 + step for steps 1..h.
    for _, g in out.groupby("unique_id"):
        assert list(g["yhat"]) == [51.0, 52.0, 53.0]


def test_predict_without_level_returns_point_only(monkeypatch):
    _install_fake_neuralforecast(monkeypatch)
    out = NeuralForecastAdapter().fit(_panel()).predict(4)
    assert list(out.columns) == ["unique_id", "ds", "yhat"]


def test_gaussian_fallback_when_backend_gives_no_intervals(monkeypatch):
    # Default point-loss neuralforecast models return no quantiles; the adapter
    # must synthesize Gaussian random-walk intervals from the training series.
    _install_fake_neuralforecast(monkeypatch, emit_intervals=False)
    out = NeuralForecastAdapter().fit(_noisy_panel()).predict(5, level=[80, 90])

    assert {"lo-80", "hi-80", "lo-90", "hi-90"} <= set(out.columns)
    assert (out["lo-80"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["hi-80"]).all()
    # 90% band strictly wider than 80% once sigma is nonzero.
    assert (out["lo-90"] < out["lo-80"]).all()
    assert (out["hi-90"] > out["hi-80"]).all()
    # random-walk fallback widens with the horizon within each series.
    for _, g in out.groupby("unique_id"):
        widths = (g["hi-80"] - g["lo-80"]).to_numpy()
        assert np.all(np.diff(widths) > 0)


def test_max_steps_and_freq_forwarded_to_backend(monkeypatch):
    recorder: dict = {}
    _install_fake_neuralforecast(monkeypatch, recorder=recorder)
    NeuralForecastAdapter(model="NBEATS", max_steps=42).fit(_panel(freq="D")).predict(3)
    assert recorder["nf"].models[0].max_steps == 42
    assert recorder["nf"].freq == "D"


def test_unknown_model_raises_valueerror(monkeypatch):
    _install_fake_neuralforecast(monkeypatch)
    adapter = NeuralForecastAdapter(model="NopeNotAModel").fit(_panel())
    with pytest.raises(ValueError, match="NopeNotAModel"):
        adapter.predict(3)


# -- import guard --------------------------------------------------------------


def test_construction_without_neuralforecast_raises_hint(monkeypatch):
    monkeypatch.setattr(nf, "_HAS_NEURALFORECAST", False)
    with pytest.raises(ImportError, match=r"forecast-os\[neural\]"):
        NeuralForecastAdapter()


# -- lifecycle -----------------------------------------------------------------


def test_predict_before_fit_raises(monkeypatch):
    _install_fake_neuralforecast(monkeypatch)
    adapter = NeuralForecastAdapter()
    with pytest.raises(ForecastOSError):
        adapter.predict(3)


# -- registration --------------------------------------------------------------


def test_register_adapters_noop_without_neuralforecast(monkeypatch):
    monkeypatch.setattr(nf, "_HAS_NEURALFORECAST", False)
    assert register_adapters() == []
    assert "neuralforecast" not in _REGISTRY


def test_register_adapters_registers_when_present(monkeypatch):
    monkeypatch.setattr(nf, "_HAS_NEURALFORECAST", True)
    assert register_adapters() == ["neuralforecast"]
    spec = _REGISTRY["neuralforecast"]
    assert spec.cls is NeuralForecastAdapter
    assert spec.family == "adapter"
