"""Tests for the Ensemble meta-forecaster (mean/median/weighted combination).

Member models are local dummies (never sibling model modules); registry
lookups use throwaway ``_test_``-prefixed names registered inside the tests.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.exceptions import NotFittedError
from forecast_os.core.registry import register
from forecast_os.core.types import to_panel
from forecast_os.models.ensemble import Ensemble


class _Const(PerSeriesForecaster):
    """Dummy member: always predicts a fixed constant."""

    def __init__(self, value=1.0):
        self.value = value

    def _fit_series(self, y):
        return {}

    def _predict_series(self, state, h):
        return np.full(h, self.value)


class _Sigma(PerSeriesForecaster):
    """Dummy member: repeats the last value with a fixed forecast sigma."""

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def _fit_series(self, y):
        return {"last": float(y[-1])}

    def _predict_series(self, state, h):
        return np.full(h, state["last"])

    def _predict_sigma(self, state, h):
        return np.full(h, self.sigma)


@pytest.fixture
def small_panel():
    rng = np.random.default_rng(0)
    return pd.concat(
        [to_panel(rng.normal(size=30), unique_id=uid) for uid in ("a", "b")],
        ignore_index=True,
    )


def test_mean_of_two_constant_members(small_panel):
    ens = Ensemble(models=(_Const(2.0), _Const(4.0))).fit(small_panel)
    pred = ens.predict(5)
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert len(pred) == 2 * 5
    assert np.allclose(pred["yhat"], 3.0)


def test_median_of_three_members(small_panel):
    ens = Ensemble(models=(_Const(1.0), _Const(2.0), _Const(10.0)), mode="median")
    pred = ens.fit(small_panel).predict(4)
    assert np.allclose(pred["yhat"], 2.0)


def test_weights_respected(small_panel):
    ens = Ensemble(models=(_Const(2.0), _Const(4.0)), weights=(3, 1))
    pred = ens.fit(small_panel).predict(4)
    assert np.allclose(pred["yhat"], 2.5)


def test_interval_columns_combined_like_yhat(small_panel):
    m1, m2 = _Const(2.0), _Const(4.0)
    pred = Ensemble(models=(m1, m2)).fit(small_panel).predict(4, level=[80])
    p1 = m1.clone().fit(small_panel).predict(4, level=[80])
    p2 = m2.clone().fit(small_panel).predict(4, level=[80])
    for col in ("yhat", "lo-80", "hi-80"):
        assert np.allclose(pred[col], (p1[col] + p2[col]) / 2)
    assert (pred["lo-80"] <= pred["hi-80"]).all()


def test_negative_weights_keep_intervals_ordered_and_nested(small_panel):
    """Negative combination weights must not invert the prediction intervals.

    Ensemble used to push lo/hi through the same signed weight vector as
    yhat, so the combined width was ``sum_i w_i * (hi_i - lo_i)``. With a
    negatively-weighted wide member that width went negative and predict()
    returned lo > yhat > hi (and a 95% band strictly inside the 80% one) —
    silently violating the lo <= yhat <= hi invariant that the rest of the
    library asserts. Negative weights are legitimate (Granger-Ramanathan /
    OLS forecast combination routinely produces them), so the fix is to
    keep the signed weighted mean for yhat and combine the members' *half
    widths* with ``|w_i|``, which is ordered and monotone by construction
    and identical to the old arithmetic whenever every weight is >= 0.
    """
    wide, narrow = _Sigma(20.0), _Sigma(1.0)
    ens = Ensemble(models=(wide, narrow), weights=[-1.0, 2.0]).fit(small_panel)
    pred = ens.predict(3, level=[80, 95])
    assert (pred["lo-80"] <= pred["yhat"]).all()
    assert (pred["yhat"] <= pred["hi-80"]).all()
    assert (pred["lo-95"] <= pred["lo-80"]).all()
    assert (pred["hi-80"] <= pred["hi-95"]).all()
    # yhat is still the plain (normalised) signed weighted mean.
    pw = wide.clone().fit(small_panel).predict(3)
    pn = narrow.clone().fit(small_panel).predict(3)
    assert np.allclose(pred["yhat"], -1.0 * pw["yhat"] + 2.0 * pn["yhat"])


def test_non_negative_weights_leave_interval_arithmetic_unchanged(small_panel):
    """The half-width rebuild must be a no-op for ordinary convex weights."""
    wide, narrow = _Sigma(20.0), _Sigma(1.0)
    pred = Ensemble(models=(wide, narrow), weights=[1.0, 3.0]).fit(small_panel).predict(
        3, level=[90]
    )
    pw = wide.clone().fit(small_panel).predict(3, level=[90])
    pn = narrow.clone().fit(small_panel).predict(3, level=[90])
    for col in ("yhat", "lo-90", "hi-90"):
        assert np.allclose(pred[col], 0.25 * pw[col] + 0.75 * pn[col])


def test_members_are_cloned_not_mutated(small_panel):
    m1 = _Const(2.0)
    Ensemble(models=(m1,)).fit(small_panel)
    assert not getattr(m1, "_is_fitted", False)


def test_string_members_resolved_at_fit(small_panel):
    register("_test_ens_const", family="baseline")(_Const)
    ens = Ensemble(models=("_test_ens_const", _Const(5.0)))
    pred = ens.fit(small_panel).predict(3)
    assert np.allclose(pred["yhat"], 3.0)  # (1.0 + 5.0) / 2


def test_unknown_string_member_fails_at_fit_not_construct(small_panel):
    ens = Ensemble(models=("_test_ens_never_registered",))  # must not raise here
    with pytest.raises(ValueError, match="unknown model"):
        ens.fit(small_panel)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Ensemble(models=(_Const(),), mode="bogus")


def test_empty_models_raises():
    with pytest.raises(ValueError):
        Ensemble(models=())


def test_weights_validation():
    with pytest.raises(ValueError):
        Ensemble(models=(_Const(), _Const()), weights=(1,))
    with pytest.raises(ValueError):
        Ensemble(models=(_Const(), _Const()), weights=(0, 0))
    with pytest.raises(ValueError):
        Ensemble(models=(_Const(), _Const()), mode="median", weights=(1, 1))


def test_clone_deep_clones_member_instances():
    inst = _Const(7.0)
    ens = Ensemble(models=(inst, "_test_ens_const"), weights=(1, 3))
    clone = ens.clone()
    assert clone is not ens
    assert clone.models[0] is not inst
    assert isinstance(clone.models[0], _Const) and clone.models[0].value == 7.0
    assert clone.models[1] == "_test_ens_const"
    assert tuple(clone.weights) == (1, 3)


def test_name_and_predict_before_fit():
    ens = Ensemble(models=(_Const(),))
    assert ens.name == "Ensemble"
    with pytest.raises(NotFittedError):
        ens.predict(3)
