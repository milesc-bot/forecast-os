"""Tests for AutoSelect (per-series CV-based model selection).

Candidates are local dummy forecasters; registry names use throwaway
``_test_``-prefixed registrations made inside the tests.
"""

import numpy as np
import pandas as pd
import pytest

import forecast_os as fos
import forecast_os.models.auto as auto_module
from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.exceptions import NotFittedError
from forecast_os.core.registry import register
from forecast_os.core.types import ID_COL, TIME_COL, to_panel
from forecast_os.datasets.synthetic import generate_returns, generate_series
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


# -- seasonal defaults + honest metric ------------------------------------------

_SEASONAL_WINNERS = {"SeasonalNaive", "Theta", "AutoETS"}
_OLD_DEFAULT_POOL = {"Naive", "Drift", "SES", "Holt", "Theta", "AutoETS", "WindowAverage"}
_SEASONAL_DEFAULT_POOL = {
    "Naive",
    "Drift",
    "SES",
    "WindowAverage",
    "SeasonalNaive",
    "Theta",
    "AutoETS",
}


def _spy_evaluate(monkeypatch):
    """Wrap evaluate() in the auto module, capturing its inputs and output."""
    captured = {}
    real_evaluate = auto_module.evaluate

    def spy(cv_df, metrics, train_df=None, seasonality=1):
        out = real_evaluate(cv_df, metrics=metrics, train_df=train_df, seasonality=seasonality)
        captured.update(train_df=train_df, seasonality=seasonality, scores=out)
        return out

    monkeypatch.setattr(auto_module, "evaluate", spy)
    return captured


def _score_columns(scores: pd.DataFrame) -> set:
    return {c for c in scores.columns if c not in (ID_COL, "metric")}


def test_default_metric_is_mase_and_season_length_auto():
    auto = AutoSelect()
    assert auto.metric == "mase"
    assert auto.season_length == "auto"
    assert auto.candidates is None


def test_defaults_infer_monthly_seasonality_on_air_passengers(monkeypatch):
    captured = _spy_evaluate(monkeypatch)
    auto = AutoSelect()
    auto.fit(fos.load_air_passengers())

    assert captured["seasonality"] == 12  # MS frequency -> m=12
    assert _score_columns(captured["scores"]) == _SEASONAL_DEFAULT_POOL
    winner = auto.best_models_["AirPassengers"]
    assert winner in _SEASONAL_WINNERS
    assert auto._fitted_[winner].season_length == 12


@pytest.mark.parametrize("season_length", [1, None])
def test_nonseasonal_season_length_keeps_old_default_pool(monkeypatch, season_length):
    captured = _spy_evaluate(monkeypatch)
    df = generate_series(n_series=2, length=60, freq="D", seasonality=7, seed=2)
    AutoSelect(val_h=6, n_windows=2, season_length=season_length).fit(df)
    assert _score_columns(captured["scores"]) == _OLD_DEFAULT_POOL
    assert captured["seasonality"] == 1


def test_explicit_int_season_length_overrides_inference(monkeypatch):
    captured = _spy_evaluate(monkeypatch)
    df = generate_series(n_series=2, length=60, freq="D", seasonality=4, seed=2)  # D infers 7
    auto = AutoSelect(val_h=6, n_windows=2, season_length=4)
    auto.fit(df)
    assert captured["seasonality"] == 4
    assert _score_columns(captured["scores"]) == _SEASONAL_DEFAULT_POOL
    assert set(auto.best_models_) == set(df[ID_COL].unique())


def test_explicit_candidates_used_verbatim_no_inference(monkeypatch):
    captured = _spy_evaluate(monkeypatch)
    cands = (_LastDummy(), _IncDummy())
    auto = AutoSelect(candidates=cands, val_h=6, n_windows=2)
    auto.fit(fos.load_air_passengers())  # default candidates would infer m=12 here
    assert _score_columns(captured["scores"]) == {"_test_last", "_test_inc"}
    assert captured["seasonality"] == 1  # no inference with an explicit candidate list
    assert auto.candidates == cands  # stored verbatim, never modified


@pytest.mark.parametrize("seed", [5, 11])
def test_zero_crossing_returns_scores_finite_and_discriminating(monkeypatch, seed):
    # smape saturates near 2.0 on zero-crossing returns (all candidates tie);
    # the mase default must yield finite scores that differ across candidates.
    captured = _spy_evaluate(monkeypatch)
    df = generate_returns(n_series=2, length=120, mu=0.0, sigma=0.01, seed=seed)
    assert (df.groupby(ID_COL)["y"].agg(lambda s: (s > 0).any() and (s < 0).any())).all()

    auto = AutoSelect()
    auto.fit(df)

    scores = captured["scores"].set_index(ID_COL)[sorted(_score_columns(captured["scores"]))]
    vals = scores.to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    for _, row in scores.iterrows():
        assert float(row.max()) - float(row.min()) > 1e-9
    assert set(auto.best_models_) == set(df[ID_COL].unique())


def test_cv_mase_scaling_uses_train_only_slice(monkeypatch):
    captured = _spy_evaluate(monkeypatch)
    df = _two_series(60)  # numeric ds 0..59; span = 12 -> first-fold train is ds 0..47
    AutoSelect(candidates=(_LastDummy(), _IncDummy()), val_h=6, n_windows=2).fit(df)
    train_df = captured["train_df"]
    assert (train_df.groupby(ID_COL).size() == 48).all()
    assert (train_df.groupby(ID_COL)[TIME_COL].max() == 47).all()


def test_holdout_mase_scaling_uses_train_only_slice(monkeypatch):
    captured = _spy_evaluate(monkeypatch)
    df = _two_series(20)  # span 24 >= 20 rows -> holdout path, hold = 5
    AutoSelect(candidates=(_LastDummy(), _IncDummy()), val_h=12, n_windows=2).fit(df)
    train_df = captured["train_df"]
    assert (train_df.groupby(ID_COL).size() == 15).all()
    assert (train_df.groupby(ID_COL)[TIME_COL].max() == 14).all()


def test_season_length_stored_and_clone_safe():
    auto = AutoSelect(season_length=12)
    assert auto.season_length == 12
    clone = auto.clone()
    assert clone.season_length == 12
    assert clone.candidates is None

    default = AutoSelect()
    assert default.clone().get_params() == default.get_params()


def test_invalid_season_length_string_raises():
    with pytest.raises(ValueError, match="season_length"):
        AutoSelect(season_length="monthly")
