"""ForecastEngine facade + adapter tests.

Self-sufficient: uses locally defined dummy forecasters (registered under
``_test_``-prefixed names at test run time), never sibling model modules.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.adapters import neuralforecast_adapter, statsforecast_adapter
from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.exceptions import ForecastOSError
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


class _Broken(PerSeriesForecaster):
    """Dummy whose fit always raises (compare() failure-tolerance tests)."""

    def _fit_series(self, y):
        raise RuntimeError("boom")

    def _predict_series(self, state, h):  # pragma: no cover - fit always raises
        return np.full(h, 0.0)


def _register_dummies():
    register("_test_engine_const", family="baseline")(_ConstA)
    register("_test_engine_broken", family="baseline")(_Broken)


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
    _register_dummies()  # _ConstA's registry name becomes its board index entry
    df = _make_panel(const=5.0, length=40)
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("mae", "rmse"),
        models=[_ConstB(), _ConstA()],  # bad model listed first on purpose
    )
    assert list(board.columns) == ["mae", "rmse"]
    assert board.shape == (2, 2)
    assert board.index.name == "model"
    # exact model wins, sorted ascending; registered classes index by registry name
    assert list(board.index) == ["_test_engine_const", "_ConstB"]
    assert board.loc["_test_engine_const", "mae"] == pytest.approx(0.0, abs=1e-12)
    assert board.loc["_ConstB", "mae"] == pytest.approx(95.0, abs=1e-9)
    assert board["mae"].is_monotonic_increasing


def test_compare_default_metrics_one_row_per_model():
    _register_dummies()
    df = _make_panel(length=40)
    board = ForecastEngine().compare(df, h=4, n_windows=2, models=[_ConstA(), _ConstB()])
    assert list(board.columns) == ["mae", "rmse", "smape"]
    assert sorted(board.index) == ["_ConstB", "_test_engine_const"]


# -- ForecastEngine.compare failure tolerance ------------------------------------


def test_compare_survives_broken_model_and_warns():
    _register_dummies()
    df = _make_panel(const=5.0, length=40)
    with pytest.warns(UserWarning, match="model _test_engine_broken failed"):
        board = ForecastEngine().compare(
            df, h=4, n_windows=2, metrics=("mae",),
            models=["_test_engine_broken", "_test_engine_const"],
        )
    # the healthy model still gets a full board row
    assert list(board.index) == ["_test_engine_const"]
    assert board.loc["_test_engine_const", "mae"] == pytest.approx(0.0, abs=1e-12)


def test_compare_all_models_failing_raises():
    _register_dummies()
    df = _make_panel(length=40)
    with pytest.raises(ForecastOSError, match="all models failed"), pytest.warns(UserWarning):
        ForecastEngine().compare(df, h=4, n_windows=2, models=["_test_engine_broken"])


def test_compare_empty_metrics_raises_value_error_not_key_error():
    """compare(metrics=[]) must fail cleanly, not with a bare pandas KeyError.

    What went wrong: an empty metric list produced a score frame with no
    'metric' column, so ``list(dict.fromkeys(scores["metric"]))`` raised
    ``KeyError: 'metric'``. On the library path that is an opaque traceback;
    through the CLI (``--metrics ""`` or ``--metrics ,``) it escaped main()'s
    ``(ForecastOSError, ValueError, OSError, ImportError)`` handler entirely
    and printed a pandas traceback with exit status 1, breaking the CLI
    module docstring's promise of ``error: ...`` on stderr and exit 2.
    """
    df = _make_panel(length=30)
    with pytest.raises(ValueError, match="no metrics"):
        ForecastEngine().compare(df, h=4, n_windows=2, metrics=[], models=[_ConstA()])


def test_compare_board_index_uses_registry_name_not_inherited():
    _register_dummies()
    df = _make_panel(const=5.0, length=40)
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("mae",),
        models=["_test_engine_const", _ConstB()],
    )
    # _ConstA is registered -> registry name; _ConstB merely inherits the class
    # attribute and must keep its own model name (no duplicate index labels)
    assert sorted(board.index) == ["_ConstB", "_test_engine_const"]


def test_compare_distinct_parameterizations_keep_distinct_labels():
    """Two parameterizations of one registered class must not share a label.

    What went wrong: ``_display_name`` mapped every resolved model onto its
    class's registry name, so ``_ConstA(value=1, alias="const_low")`` and
    ``_ConstA(value=99, alias="const_high")`` both landed on
    ``'_test_engine_const'``. The board came back with a non-unique index
    carrying genuinely different scores, ``board.loc['_test_engine_const']``
    returned two rows, and compare()'s documented promise that "an index entry
    can be fed straight back into forecast()" resolved to the default-parameter
    model rather than the winner. ``alias`` is the only supported way to
    compare two parameterizations of one class (``_resolve`` rejects two
    ``(name, params)`` specs of the same model as duplicates), and
    ``cross_validation`` already keys its columns on ``model.name``.

    Right behaviour: when a registry name is claimed by more than one resolved
    model, those models keep their own ``model.name`` (the alias); an
    unambiguous registry name is still used, so existing boards are unchanged.
    """
    _register_dummies()
    df = _make_panel(const=5.0, length=40)
    low = _ConstA(value=1.0)
    low.alias = "const_low"
    high = _ConstA(value=99.0)
    high.alias = "const_high"
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("mae",), models=[low, high, _ConstB()]
    )
    assert board.index.is_unique
    assert sorted(board.index) == ["_ConstB", "const_high", "const_low"]
    # and the scores really are distinct, i.e. the collapse was lossy
    assert board.loc["const_low", "mae"] != board.loc["const_high", "mae"]


# -- parameterized model specs ---------------------------------------------------


def test_forecast_accepts_param_spec_and_param_reaches_model():
    df = _make_panel(length=20)
    out = ForecastEngine().forecast(df, h=3, models=[("ridge_lag", {"lags": 6})])
    assert "RidgeLag" in out.columns
    assert len(out) == 2 * 3
    # proof the lags override reached the model: the default (lags=14) needs
    # 24 observations per series and fails on this 20-row panel
    with pytest.raises(ForecastOSError):
        ForecastEngine().forecast(df, h=3, models=["ridge_lag"])


def test_compare_accepts_param_spec():
    df = _make_panel(length=30)  # too short for ridge_lag's default lags=14 in CV
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("mae",), models=[("ridge_lag", {"lags": 6})]
    )
    assert list(board.index) == ["ridge_lag"]
    assert np.isfinite(board["mae"]).all()


def test_engine_single_spec_tuple_unwrapped():
    df = _make_panel(length=30)
    out = ForecastEngine(models=("ridge_lag", {"lags": 6})).forecast(df, h=2)
    assert "RidgeLag" in out.columns
    assert len(out) == 2 * 2


def test_engine_rejects_malformed_spec():
    df = _make_panel()
    with pytest.raises(ValueError, match="resolve model spec"):
        ForecastEngine().forecast(df, h=2, models=[("ridge_lag", 6)])


# -- covariate panels: forecast() must be able to run what compare() ranked ------


def _exog_panel(n=60, n_series=2, seed=3):
    """Panel with a numeric ``spend`` driver that y depends on."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        spend = 50.0 + rng.normal(0, 5.0, n)
        y = 10.0 + 0.4 * np.arange(n) + 2.0 * spend + rng.normal(0, 1.0, n)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": pd.date_range("2020-01-01", periods=n, freq="MS"),
                    "y": y,
                    "spend": spend,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _future_x(panel, h, freq="MS"):
    """Known future covariate rows: h periods past each series' last ds."""
    frames = []
    for uid, g in panel.groupby("unique_id", sort=True):
        last = g["ds"].max()
        future = pd.date_range(last, periods=h + 1, freq=freq)[1:]
        frames.append(
            pd.DataFrame({"unique_id": uid, "ds": future, "spend": [50.0] * h})
        )
    return pd.concat(frames, ignore_index=True)


def test_forecast_threads_known_future_covariates():
    """forecast() must be able to run the exog models compare() ranks.

    What went wrong: on a covariate panel, ``cross_validation`` (and so
    ``compare``) fits exog-capable models and predicts with an X_df built
    from the held-out rows — the documented known-future-inputs convention.
    But ``ForecastEngine.forecast`` had no ``X_df`` parameter at all and just
    called ``model.clone().fit(df).predict(h, level=level)``, so the winning
    model raised "predict() requires X_df with h future rows per series".
    compare()'s docstring promises an index entry "can be fed straight back
    into forecast()", and on covariate panels it could not be.
    """
    panel = _exog_panel()
    h = 4
    out = ForecastEngine().forecast(
        panel, h=h, models=[("ridge_lag", {"lags": 6})], X_df=_future_x(panel, h)
    )
    assert len(out) == 2 * h
    assert np.isfinite(out["RidgeLag"]).all()


def test_forecast_without_x_df_unchanged_for_plain_models():
    """X_df defaults to None and non-exog panels/models are untouched."""
    df = _make_panel(length=30)
    out = ForecastEngine().forecast(df, h=3, models=[("ridge_lag", {"lags": 6})])
    assert len(out) == 2 * 3


def test_forecast_ignores_x_df_for_models_that_cannot_use_it():
    """A model that never consumed covariates must not be handed X_df."""
    panel = _exog_panel()
    h = 3
    out = ForecastEngine().forecast(
        panel, h=h, models=[_ConstA()], X_df=_future_x(panel, h)
    )
    assert len(out) == 2 * h
    assert (out["_ConstA"] == 5.0).all()


# -- ForecastEngine.compare with interval metrics --------------------------------


class _WideIntervals(PerSeriesForecaster):
    """Mean forecaster with huge intervals: empirical coverage ~1.0."""

    def _fit_series(self, y):
        return {"mean": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mean"])

    def _predict_sigma(self, state, h):
        return np.full(h, 1000.0)


class _NarrowMissIntervals(PerSeriesForecaster):
    """Biased forecaster with near-zero-width intervals: coverage ~0.0."""

    def _fit_series(self, y):
        return {"mean": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mean"] + 10.0)

    def _predict_sigma(self, state, h):
        return np.full(h, 1e-9)


def test_compare_level_board_has_coverage_and_sorts_by_mase():
    _register_dummies()
    df = _make_panel(length=40)
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("mase", "coverage"),
        models=[_ConstB(), _ConstA()], level=[80],
    )
    assert list(board.columns) == ["mase", "coverage-80"]
    # first metric is mase (not coverage), so plain ascending sort applies
    assert board["mase"].is_monotonic_increasing
    assert list(board.index) == ["_test_engine_const", "_ConstB"]
    assert board["coverage-80"].between(0.0, 1.0).all()


def test_compare_coverage_first_sorting_closest_to_nominal_wins():
    df = _make_panel(length=40)
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=("coverage",),
        models=[_NarrowMissIntervals(), _WideIntervals()], level=[80],
    )
    # wide covers everything (|1.0 - 0.8| = 0.2) and beats narrow-missing
    # (|0.0 - 0.8| = 0.8) even though a plain ascending sort would flip them
    assert list(board.index) == ["_WideIntervals", "_NarrowMissIntervals"]
    assert board.loc["_WideIntervals", "coverage-80"] == pytest.approx(1.0)
    assert board.loc["_NarrowMissIntervals", "coverage-80"] == pytest.approx(0.0)


class _BiasSmall(PerSeriesForecaster):
    """Forecasts the training mean minus 0.5 (bias -0.5 on a constant panel)."""

    def _fit_series(self, y):
        return {"mean": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mean"] - 0.5)


class _BiasLarge(_BiasSmall):
    """Forecasts the training mean minus 31.1 (bias -31.1 on a constant panel)."""

    def _predict_series(self, state, h):
        return np.full(h, state["mean"] - 31.1)


@pytest.mark.parametrize(
    "metric, small, large",
    [("bias", -0.5, -31.1), ("pct_bias", -0.1, -6.22)],
)
def test_compare_signed_metric_first_sorts_by_abs_closest_to_zero(metric, small, large):
    """Signed governance metrics sort by |value|: a plain ascending sort would
    rank the -31.1-bias model above the -0.5-bias one."""
    df = _make_panel(const=5.0, length=40)
    board = ForecastEngine().compare(
        df, h=4, n_windows=2, metrics=(metric,),
        models=[_BiasLarge(), _BiasSmall()],  # heavy sandbagger listed first
    )
    assert list(board.index) == ["_BiasSmall", "_BiasLarge"]
    assert board.loc["_BiasSmall", metric] == pytest.approx(small)
    assert board.loc["_BiasLarge", metric] == pytest.approx(large)


def test_compare_interval_metric_without_level_raises():
    df = _make_panel(length=40)
    with pytest.raises(ValueError, match=r"level=\["):
        ForecastEngine().compare(
            df, h=4, n_windows=2, metrics=("coverage",), models=[_ConstA()]
        )


def test_compare_uses_constructor_level_by_default():
    df = _make_panel(length=40)
    engine = ForecastEngine(level=[90])
    board = engine.compare(
        df, h=4, n_windows=2, metrics=("mae", "coverage"), models=[_ConstA()]
    )
    assert list(board.columns) == ["mae", "coverage-90"]


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
