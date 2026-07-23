"""Tests for AutoSelect (per-series CV-based model selection).

Candidates are local dummy forecasters; registry names use throwaway
``_test_``-prefixed registrations made inside the tests.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.exceptions import NotFittedError
from forecast_os.core.registry import register
from forecast_os.core.types import ID_COL, to_panel
from forecast_os.models.auto import AutoSelect


class _LastDummy(PerSeriesForecaster):
    """Dummy candidate: repeats the last observed value."""

    alias = "_test_last"

    def _fit_series(self, y):
        return {"last": float(y[-1])}

    def _predict_series(self, state, h):
        return np.full(h, state["last"])


class _IncDummy(PerSeriesForecaster):
    """Dummy candidate: last value plus one per step."""

    alias = "_test_inc"

    def _fit_series(self, y):
        return {"last": float(y[-1])}

    def _predict_series(self, state, h):
        return state["last"] + np.arange(1, h + 1, dtype=float)


def _two_series(length: int) -> pd.DataFrame:
    a = to_panel(np.full(length, 5.0), unique_id="A")  # _LastDummy is exact
    b = to_panel(np.arange(length, dtype=float), unique_id="B")  # _IncDummy is exact
    return pd.concat([a, b], ignore_index=True)


def test_picks_exact_model_per_series_via_cv():
    df = _two_series(40)
    auto = AutoSelect(candidates=(_LastDummy(), _IncDummy()), val_h=6, n_windows=2)
    auto.fit(df)
    assert auto.best_models_ == {"A": "_test_last", "B": "_test_inc"}

    pred = auto.predict(5)
    a = pred.loc[pred[ID_COL] == "A", "yhat"].to_numpy()
    b = pred.loc[pred[ID_COL] == "B", "yhat"].to_numpy()
    assert np.allclose(a, 5.0)
    assert np.allclose(b, np.arange(40.0, 45.0))


def test_short_panel_falls_back_to_holdout():
    df = _two_series(20)  # span = 24 > 20 rows -> 75/25 holdout path
    auto = AutoSelect(candidates=(_LastDummy(), _IncDummy()), val_h=12, n_windows=2)
    auto.fit(df)
    assert auto.best_models_ == {"A": "_test_last", "B": "_test_inc"}


def test_end_to_end_on_panel_fixture(panel):
    auto = AutoSelect(candidates=(_LastDummy(), _IncDummy()), val_h=6, n_windows=2)
    auto.fit(panel)
    uids = set(panel[ID_COL].unique())
    assert set(auto.best_models_) == uids

    pred = auto.predict(7, level=[80])
    assert list(pred.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert {"lo-80", "hi-80"} <= set(pred.columns)
    assert (pred.groupby(ID_COL).size() == 7).all()
    assert np.isfinite(pred["yhat"]).all()


def test_string_candidates_resolved_at_fit():
    register("_test_auto_last", family="baseline")(_LastDummy)
    register("_test_auto_inc", family="baseline")(_IncDummy)
    df = _two_series(40)
    auto = AutoSelect(candidates=("_test_auto_last", "_test_auto_inc"), val_h=6, n_windows=2)
    auto.fit(df)
    assert auto.best_models_ == {"A": "_test_last", "B": "_test_inc"}


def test_unknown_string_candidate_fails_at_fit_not_construct():
    auto = AutoSelect(candidates=("_test_auto_missing",))  # must not raise here
    with pytest.raises(ValueError, match="unknown model"):
        auto.fit(_two_series(40))


def test_duplicate_candidate_names_raise():
    auto = AutoSelect(candidates=(_LastDummy(), _LastDummy()), val_h=6, n_windows=2)
    with pytest.raises(ValueError, match="duplicate"):
        auto.fit(_two_series(40))


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        AutoSelect(candidates=(_LastDummy(),)).predict(3)


def test_clone_deep_clones_candidate_instances():
    inst = _LastDummy()
    auto = AutoSelect(candidates=(inst, "_test_auto_last"), metric="mae", val_h=4, n_windows=1)
    clone = auto.clone()
    assert clone is not auto
    assert clone.candidates[0] is not inst
    assert isinstance(clone.candidates[0], _LastDummy)
    assert clone.candidates[1] == "_test_auto_last"
    assert (clone.metric, clone.val_h, clone.n_windows) == ("mae", 4, 1)
