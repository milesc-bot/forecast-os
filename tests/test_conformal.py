"""Tests for the split-conformal interval wrapper.

Member models are local dummies; registry lookups use throwaway
``_test_``-prefixed names registered inside the tests.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import register
from forecast_os.core.types import ID_COL, to_panel
from forecast_os.uncertainty.conformal import ConformalForecaster


class _Mean(PerSeriesForecaster):
    """Dummy member: predicts the in-sample mean."""

    def _fit_series(self, y):
        return {"mu": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mu"])


class _Zero(PerSeriesForecaster):
    """Dummy member: always predicts zero."""

    def _fit_series(self, y):
        return {}

    def _predict_series(self, state, h):
        return np.zeros(h)


def test_empirical_coverage_of_80_intervals():
    rng = np.random.default_rng(11)
    y = 10.0 + rng.normal(0.0, 1.0, size=200)
    train, test = y[:160], y[160:]
    cf = ConformalForecaster(model=_Mean()).fit(to_panel(train))
    pred = cf.predict(40, level=[80])
    inside = (test >= pred["lo-80"].to_numpy()) & (test <= pred["hi-80"].to_numpy())
    assert 0.6 <= inside.mean() <= 0.95


def test_bounds_ordering(panel):
    cf = ConformalForecaster(model=_Mean()).fit(panel)
    pred = cf.predict(10, level=[80, 95])
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert (pred["lo-80"] <= pred["yhat"]).all()
    assert (pred["yhat"] <= pred["hi-80"]).all()
    assert (pred["lo-95"] <= pred["lo-80"]).all()
    assert (pred["hi-80"] <= pred["hi-95"]).all()


def test_width_equals_corrected_calibration_quantile():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 2.0, size=40)  # n_cal = max(8, 40 // 4) = 10
    cf = ConformalForecaster(model=_Zero()).fit(to_panel(y))
    pred = cf.predict(5, level=[80])
    # finite-sample-corrected order statistic: min(1, (n+1)*0.8/n) = 0.88
    expected = np.quantile(np.abs(y[-10:]), 0.88, method="higher")
    assert np.allclose(pred["yhat"], 0.0)
    assert np.allclose(pred["hi-80"] - pred["yhat"], expected)
    assert np.allclose(pred["yhat"] - pred["lo-80"], expected)


def test_nominal_90_coverage_on_iid_gaussian_is_at_least_085():
    rng = np.random.default_rng(21)
    y = 5.0 + rng.normal(0.0, 1.0, size=400)
    train, test = y[:300], y[300:]
    cf = ConformalForecaster(model=_Mean()).fit(to_panel(train))
    pred = cf.predict(100, level=[90])
    inside = (test >= pred["lo-90"].to_numpy()) & (test <= pred["hi-90"].to_numpy())
    assert inside.mean() >= 0.85


def test_per_series_widths_reflect_noise():
    rng = np.random.default_rng(7)
    df = pd.concat(
        [
            to_panel(rng.normal(0.0, 0.5, size=120), unique_id="calm"),
            to_panel(rng.normal(0.0, 5.0, size=120), unique_id="wild"),
        ],
        ignore_index=True,
    )
    cf = ConformalForecaster(model=_Mean()).fit(df)
    pred = cf.predict(6, level=[80])
    width = (pred["hi-80"] - pred["lo-80"]).groupby(pred[ID_COL]).mean()
    assert width["wild"] > 2 * width["calm"]


def test_string_member_resolved_at_fit():
    register("_test_conf_mean", family="baseline")(_Mean)
    rng = np.random.default_rng(1)
    cf = ConformalForecaster(model="_test_conf_mean")
    cf.fit(to_panel(rng.normal(size=60)))
    pred = cf.predict(4, level=[80])
    assert np.isfinite(pred[["yhat", "lo-80", "hi-80"]].to_numpy()).all()


def test_unknown_string_member_fails_at_fit_not_construct():
    cf = ConformalForecaster(model="_test_conf_never_registered")  # must not raise here
    with pytest.raises(ValueError, match="unknown model"):
        cf.fit(to_panel(np.arange(60.0)))


def test_series_too_short_for_calibration_raises():
    with pytest.raises(ForecastOSError):
        ConformalForecaster(model=_Mean()).fit(to_panel(np.arange(8.0)))


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        ConformalForecaster(model=_Mean()).predict(3)


@pytest.mark.parametrize("h", [0, -1, 2.5, "3"])
def test_predict_invalid_h_raises(panel, h):
    cf = ConformalForecaster(model=_Mean()).fit(panel)
    with pytest.raises(ValueError, match="h must be"):
        cf.predict(h)


@pytest.mark.parametrize("level", [[150], [0], [100], [-5]])
def test_predict_invalid_level_raises(panel, level):
    cf = ConformalForecaster(model=_Mean()).fit(panel)
    with pytest.raises(ValueError, match="confidence levels"):
        cf.predict(3, level=level)


def test_clone_deep_clones_member_instance():
    inst = _Mean()
    cf = ConformalForecaster(model=inst, level_calibration_fraction=0.3, min_calibration=5)
    clone = cf.clone()
    assert clone is not cf
    assert clone.model is not inst and isinstance(clone.model, _Mean)
    assert clone.level_calibration_fraction == 0.3
    assert clone.min_calibration == 5


def test_clone_passes_string_member_through():
    cf = ConformalForecaster(model="_test_conf_mean")
    assert cf.clone().model == "_test_conf_mean"


# --- v0.10.0 audit regressions ---------------------------------------------


def test_unattainable_level_warns_and_documents_the_coverage_cap():
    """A level split conformal cannot reach was silently clamped to the max score.

    ``q_level = min(1.0, (n + 1) * (lvl / 100) / n)`` substitutes the largest
    calibration residual whenever ``ceil((n + 1) * lvl/100) > n``, so with
    n_cal = 8 the 90/95/99% bands are all the *same* band and all cover at
    most 8/9 = 88.9%. That is a real miscalibration for a class whose whole
    value proposition is a finite-sample coverage guarantee, so it must not be
    silent. The band itself is still returned (the widest one available) —
    refusing the call would break short real-world series.
    """
    rng = np.random.default_rng(0)
    cf = ConformalForecaster(model=_Zero()).fit(to_panel(rng.normal(size=32)))
    assert len(cf._abs_resid_["series-0"]) == 8

    with pytest.warns(UserWarning, match="level 99 needs >= 99 calibration residuals"):
        pred = cf.predict(1, level=[99])
    assert np.isfinite(pred[["lo-99", "hi-99"]].to_numpy()).all()

    # level 80 needs ceil(9 * 0.8) = 8 <= 8 residuals: attainable, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cf.predict(1, level=[80])


def test_unattainable_level_warning_names_the_series_and_the_cap():
    rng = np.random.default_rng(4)
    cf = ConformalForecaster(model=_Zero()).fit(to_panel(rng.normal(size=32)))
    with pytest.warns(UserWarning) as rec:
        cf.predict(1, level=[90, 95])
    msg = str(rec[0].message)
    assert "'series-0'" in msg and "has 8" in msg and "88.9%" in msg


# --- remediation round 2 ----------------------------------------------------


class _ExplodesOnFullRefit(PerSeriesForecaster):
    """Member whose fit raises once the calibration head has been consumed.

    ``ConformalForecaster.fit`` fits the member twice: once on the head to get
    calibration residuals, once on the full panel to get the point forecasts.
    This dummy lets the first fit through and blows up on the second (it fires
    on the longest panel it is asked to fit), reproducing a refit that fails
    only on the full series.
    """

    def __init__(self, boom_at: int = 0):
        self.boom_at = boom_at

    def _fit_series(self, y):
        if len(y) >= self.boom_at:
            raise ForecastOSError("full refit exploded")
        return {"mu": float(np.mean(y))}

    def _predict_series(self, state, h):
        return np.full(h, state["mu"])


def test_failed_refit_does_not_leave_a_half_updated_conformal_model():
    """A fit that raises must not leave old forecasts wearing new interval widths.

    ``fit`` published ``_abs_resid_`` (the NEW calibration) before the final
    full refit and never cleared ``_is_fitted``. When that refit raised, the
    object kept ``_is_fitted = True``, the OLD ``_model_`` and the NEW
    residuals, so ``predict()`` still succeeded and returned stale point
    forecasts wrapped in freshly calibrated widths — a silently wrong number
    rather than a loud failure. A failed refit must leave the object unfitted
    (and must not mutate the state of the previous successful fit).
    """
    first = to_panel(np.full(40, 100.0) + np.arange(40) % 3)
    cf = ConformalForecaster(model=_Mean()).fit(first)
    good = cf.predict(2, level=[80])
    resid_before = {k: v.copy() for k, v in cf._abs_resid_.items()}

    # 40 rows, 10 held out for calibration -> head is 30 rows, full panel is 40
    cf.model = _ExplodesOnFullRefit(boom_at=40)
    with pytest.raises(ForecastOSError, match="full refit exploded"):
        cf.fit(to_panel(np.full(40, 500.0) + np.arange(40) % 3))

    assert cf._is_fitted is False
    with pytest.raises(NotFittedError):
        cf.predict(2, level=[80])
    # the previous fit's calibration was not overwritten by the failed one
    for uid, r in resid_before.items():
        np.testing.assert_allclose(cf._abs_resid_[uid], r)
    cf._is_fitted = True
    pd.testing.assert_frame_equal(cf.predict(2, level=[80]), good)
