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
