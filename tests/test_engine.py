"""ForecastEngine facade + adapter tests.

Self-sufficient: uses locally defined dummy forecasters (registered under
``_test_``-prefixed names at test run time), never sibling model modules.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.adapters import neuralforecast_adapter, statsforecast_adapter
from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.registry import register
from forecast_os.engine import ForecastEngine


class _ConstA(PerSeriesForecaster):
    """Constant-forecast dummy."""

    def __init__(self, value: float = 5.0):
        self.value = value

    def _fit_series(self, y):
        return {}

    def _predict_series(self, state, h):
        return np.full(h, float(self.value))


class _ConstB(_ConstA):
    """Second constant dummy so two models have distinct names."""

    def __init__(self, value: float = 100.0):
        self.value = value


def _register_dummies():
    register("_test_engine_const", family="baseline")(_ConstA)


def _make_panel(n_series=2, length=30, const=None, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        if const is not None:
            y = np.full(length, float(const))
        else:
            y = 50.0 + rng.normal(0, 1.0, length)
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": np.arange(length), "y": y}))
    return pd.concat(frames, ignore_index=True)


# -- ForecastEngine.forecast ---------------------------------------------------


def test_forecast_wide_frame_with_levels():
    df = _make_panel()
    out = ForecastEngine().forecast(df, h=5, models=[_ConstA(), _ConstB()], level=[80])
    assert list(out.columns) == [
        "unique_id",
        "ds",
        "_ConstA",
        "_ConstA-lo-80",
        "_ConstA-hi-80",
        "_ConstB",
        "_ConstB-lo-80",
        "_ConstB-hi-80",
    ]
    assert len(out) == 2 * 5
    assert (out.groupby("unique_id").size() == 5).all()
    assert (out["_ConstA"] == 5.0).all()
    assert (out["_ConstB"] == 100.0).all()
    assert (out["_ConstA-lo-80"] <= out["_ConstA"]).all()
    assert (out["_ConstA"] <= out["_ConstA-hi-80"]).all()
    # integer ds continues right after training (last train ds is 29)
    assert (out.groupby("unique_id")["ds"].min() == 30).all()
    assert (out.groupby("unique_id")["ds"].max() == 34).all()


def test_forecast_without_level_has_no_interval_columns():
    df = _make_panel()
    out = ForecastEngine().forecast(df, h=3, models=[_ConstA()])
    assert list(out.columns) == ["unique_id", "ds", "_ConstA"]


def test_forecast_resolves_registry_names_and_constructor_default():
    _register_dummies()
    df = _make_panel()
    engine = ForecastEngine(models=("_test_engine_const",))
    out = engine.forecast(df, h=3)
    assert "_ConstA" in out.columns
    assert (out["_ConstA"] == 5.0).all()


def test_forecast_constructor_level_used_when_method_level_none():
    df = _make_panel()
    engine = ForecastEngine(level=[90])
    out = engine.forecast(df, h=2, models=[_ConstA()])
    assert {"_ConstA-lo-90", "_ConstA-hi-90"} <= set(out.columns)


def test_forecast_duplicate_model_names_raise():
    df = _make_panel()
    with pytest.raises(ValueError, match="duplicate"):
        ForecastEngine().forecast(df, h=2, models=[_ConstA(), _ConstA(3.0)])


def test_forecast_unknown_model_name_raises():
    df = _make_panel()
    with pytest.raises(ValueError, match="unknown model"):
        ForecastEngine().forecast(df, h=2, models=["_test_engine_no_such_model"])


def test_engine_stores_constructor_params_as_attributes():
    engine = ForecastEngine(models=("_test_engine_const",), level=[80])
    assert engine.models == ("_test_engine_const",)
    assert engine.level == [80]


# -- ForecastEngine.cross_validate ---------------------------------------------


def test_cross_validate_thin_wrapper():
    df = _make_panel(length=40)
    out = ForecastEngine().cross_validate(df, h=4, n_windows=2, models=[_ConstA()])
    assert {"unique_id", "ds", "cutoff", "y", "_ConstA"} <= set(out.columns)
    assert len(out) == 2 * 4 * 2  # series x h x windows
    assert (out["_ConstA"] == 5.0).all()


def test_cross_validate_passes_level_through():
    df = _make_panel(length=40)
    engine = ForecastEngine(level=[80])
    out = engine.cross_validate(df, h=3, n_windows=2, models=[_ConstA()])
    assert {"_ConstA-lo-80", "_ConstA-hi-80"} <= set(out.columns)


# -- ForecastEngine.compare ----------------------------------------------------


def test_compare_leaderboard_sorted_by_first_metric():
    df = _make_panel(const=5.0, length=40)
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("mae", "rmse"),
        models=[_ConstB(), _ConstA()],  # bad model listed first on purpose
    )
    assert list(board.columns) == ["mae", "rmse"]
    assert board.shape == (2, 2)
    assert board.index.name == "model"
    assert list(board.index) == ["_ConstA", "_ConstB"]  # exact model wins, sorted ascending
    assert board.loc["_ConstA", "mae"] == pytest.approx(0.0, abs=1e-12)
    assert board.loc["_ConstB", "mae"] == pytest.approx(95.0, abs=1e-9)
    assert board["mae"].is_monotonic_increasing


def test_compare_default_metrics_one_row_per_model():
    df = _make_panel(length=40)
    board = ForecastEngine().compare(df, h=4, n_windows=2, models=[_ConstA(), _ConstB()])
    assert list(board.columns) == ["mae", "rmse", "smape"]
    assert sorted(board.index) == ["_ConstA", "_ConstB"]


# -- adapters --------------------------------------------------------------------


@pytest.mark.skipif(
    statsforecast_adapter._HAS_STATSFORECAST, reason="statsforecast is installed"
)
def test_statsforecast_adapter_raises_importerror_without_backend():
    with pytest.raises(ImportError, match=r"forecast-os\[nixtla\]"):
        statsforecast_adapter.StatsForecastAdapter()


@pytest.mark.skipif(
    neuralforecast_adapter._HAS_NEURALFORECAST, reason="neuralforecast is installed"
)
def test_neuralforecast_adapter_raises_importerror_without_backend():
    with pytest.raises(ImportError, match=r"forecast-os\[neural\]"):
        neuralforecast_adapter.NeuralForecastAdapter()


def test_register_adapters_noop_when_backend_missing():
    if not statsforecast_adapter._HAS_STATSFORECAST:
        assert statsforecast_adapter.register_adapters() == []
    if not neuralforecast_adapter._HAS_NEURALFORECAST:
        assert neuralforecast_adapter.register_adapters() == []
