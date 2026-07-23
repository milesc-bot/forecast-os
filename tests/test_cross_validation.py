"""Tests for evaluation.backtest.cross_validation, including a no-leakage probe."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.exceptions import DataContractError
from forecast_os.core.registry import _REGISTRY, register
from forecast_os.datasets.synthetic import generate_series
from forecast_os.evaluation.backtest import cross_validation

N_SERIES = 3
LENGTH = 50


def _probe_cls():
    """A fresh probe class per test: records the length of every fitted series.

    The record list is a class attribute so it survives the ``clone()`` that
    cross_validation performs per window.
    """

    class Probe(PerSeriesForecaster):
        """Probe forecaster: predicts the last value, records fit lengths."""

        fit_lengths: list = []

        def _fit_series(self, y):
            type(self).fit_lengths.append(len(y))
            return {}

        def _predict_series(self, state, h):
            return np.full(h, float(state["_y"][-1]))

    return Probe


def _panel():
    return generate_series(n_series=N_SERIES, length=LENGTH, seasonality=None, seed=1)


# -- no leakage ---------------------------------------------------------------


def test_no_leakage_default_step():
    probe = _probe_cls()
    cross_validation(_panel(), [probe()], h=5, n_windows=3)
    # windows oldest-first; offset = (n_windows-1-i)*step, step defaults to h
    expected = []
    for offset in (10, 5, 0):
        expected += [LENGTH - 5 - offset] * N_SERIES
    assert probe.fit_lengths == expected


def test_no_leakage_custom_step():
    probe = _probe_cls()
    cross_validation(_panel(), [probe()], h=5, n_windows=3, step_size=2)
    expected = []
    for offset in (4, 2, 0):
        expected += [LENGTH - 5 - offset] * N_SERIES
    assert probe.fit_lengths == expected


def test_cutoff_spacing_default_and_custom():
    df = _panel()
    for step, kwargs in [(5, {}), (2, {"step_size": 2})]:
        cv = cross_validation(df, [_probe_cls()()], h=5, n_windows=3, **kwargs)
        for _, g in cv.groupby("unique_id"):
            cutoffs = pd.Series(sorted(g["cutoff"].unique()))
            assert len(cutoffs) == 3
            assert (cutoffs.diff().dropna() == pd.Timedelta(days=step)).all()


def test_test_rows_are_h_rows_immediately_after_cutoff():
    df = _panel()
    h = 5
    cv = cross_validation(df, [_probe_cls()()], h=h, n_windows=3)
    for (_uid, cutoff), g in cv.groupby(["unique_id", "cutoff"]):
        assert len(g) == h
        expected_ds = cutoff + pd.to_timedelta(np.arange(1, h + 1), unit="D")
        assert list(g.sort_values("ds")["ds"]) == list(expected_ds)
    # y values match the source panel
    merged = cv.merge(df, on=["unique_id", "ds"], suffixes=("", "_src"))
    assert len(merged) == len(cv)
    np.testing.assert_allclose(merged["y"], merged["y_src"])


# -- output shape and columns -------------------------------------------------


def test_output_shape_and_forecast_values():
    df = _panel()
    cv = cross_validation(df, [_probe_cls()()], h=5, n_windows=3)
    assert len(cv) == N_SERIES * 3 * 5
    assert {"unique_id", "ds", "cutoff", "y", "Probe"} <= set(cv.columns)
    assert np.isfinite(cv["Probe"]).all()
    # probe repeats the last training value == y at the cutoff
    for (uid, cutoff), g in cv.groupby(["unique_id", "cutoff"]):
        last_train = df[(df["unique_id"] == uid) & (df["ds"] == cutoff)]["y"].iloc[0]
        np.testing.assert_allclose(g["Probe"], last_train)


def test_interval_columns_per_model_when_level_passed():
    cv = cross_validation(_panel(), [_probe_cls()()], h=4, n_windows=2, level=[80])
    assert {"Probe-lo-80", "Probe-hi-80"} <= set(cv.columns)
    assert "lo-80" not in cv.columns  # renamed, not raw
    assert (cv["Probe-lo-80"] <= cv["Probe"]).all()
    assert (cv["Probe"] <= cv["Probe-hi-80"]).all()


def test_multiple_models_one_column_each():
    class Zero(PerSeriesForecaster):
        """Predicts zero."""

        def _fit_series(self, y):
            return {}

        def _predict_series(self, state, h):
            return np.zeros(h)

    cv = cross_validation(_panel(), [_probe_cls()(), Zero()], h=3, n_windows=2)
    assert {"Probe", "Zero"} <= set(cv.columns)
    assert (cv["Zero"] == 0).all()


# -- validation and errors ----------------------------------------------------


def test_too_short_series_raises():
    # need > h + (n_windows-1)*step_size = 15 rows; give exactly 15
    df = generate_series(n_series=1, length=15, seasonality=None, seed=2)
    with pytest.raises(DataContractError, match="too short"):
        cross_validation(df, [_probe_cls()()], h=5, n_windows=3)


def test_string_model_names_resolve():
    probe = _probe_cls()
    register("_test_cv_probe", family="baseline")(probe)
    try:
        cv = cross_validation(_panel(), ["_test_cv_probe"], h=3, n_windows=2)
        assert "Probe" in cv.columns
        assert probe.fit_lengths  # the registered class was actually fitted
    finally:
        _REGISTRY.pop("_test_cv_probe", None)


def test_duplicate_model_names_raise():
    probe = _probe_cls()
    with pytest.raises(ValueError, match="duplicate model names"):
        cross_validation(_panel(), [probe(), probe()], h=3, n_windows=2)


@pytest.mark.parametrize(
    "kwargs",
    [{"h": 0}, {"h": 3, "n_windows": 0}, {"h": 3, "step_size": 0}],
)
def test_invalid_window_args_raise(kwargs):
    kwargs = {"n_windows": 2, **kwargs}
    with pytest.raises(ValueError):
        cross_validation(_panel(), [_probe_cls()()], **kwargs)


def test_shuffled_predict_output_raises():
    """A model returning out-of-order ds must fail loudly, not merge silently."""

    class Shuffled(PerSeriesForecaster):
        """Returns its prediction rows in reversed ds order."""

        def _fit_series(self, y):
            return {}

        def _predict_series(self, state, h):
            return np.zeros(h)

        def predict(self, h, level=None):
            out = super().predict(h, level=level)
            return out.iloc[::-1].reset_index(drop=True)

    with pytest.raises(DataContractError, match="strictly increasing"):
        cross_validation(_panel(), [Shuffled()], h=3, n_windows=2)


# -- PerSeriesForecaster fallback sigma (core.base, used by every CV model) ----


def test_fallback_sigma_is_uncentered_rms_of_residuals():
    """Systematic forecast bias must widen intervals, not vanish by centering."""

    class Biased(PerSeriesForecaster):
        """Fitted values are y - 2: constant residual of +2, zero variance."""

        def _fit_series(self, y):
            return {"fitted": y - 2.0}

        def _predict_series(self, state, h):
            return np.full(h, float(state["_y"][-1]))

    df = pd.DataFrame({"unique_id": "a", "ds": range(10), "y": np.arange(10.0)})
    m = Biased().fit(df)
    resid = np.full(10, 2.0)
    expected = np.sqrt(np.sum(resid**2) / (resid.size - 1))
    assert m._series_state["a"]["_sigma"] == pytest.approx(expected)
    assert m._series_state["a"]["_sigma"] > 1.0  # centered std would be ~0


def test_fallback_sigma_diff_path_is_uncentered_too():
    class NoFits(PerSeriesForecaster):
        """No usable fitted values -> diff-based sigma fallback."""

        def _fit_series(self, y):
            return {"fitted": np.full(len(y), np.nan)}

        def _predict_series(self, state, h):
            return np.full(h, float(state["_y"][-1]))

    df = pd.DataFrame({"unique_id": "a", "ds": range(10), "y": np.arange(10.0)})
    m = NoFits().fit(df)
    d = np.diff(np.arange(10.0))  # all ones; centered std would be 0
    expected = np.sqrt(np.sum(d**2) / (d.size - 1))
    assert m._series_state["a"]["_sigma"] == pytest.approx(expected)
