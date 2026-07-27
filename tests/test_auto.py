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
from forecast_os.models.baselines import SeasonalNaive
from forecast_os.models.ml import RidgeLag


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


# --- v0.2.0 verifier regressions ------------------------------------------


def _freq_panel(n, freq, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            ID_COL: "s1",
            TIME_COL: pd.date_range("2020-01-06", periods=n, freq=freq),
            "y": 10 + np.sin(np.arange(n)) + rng.normal(0, 0.1, n),
        }
    )


@pytest.mark.parametrize(
    "n,freq",
    [(60, "W"), (30, "MS"), (16, "MS"), (28, "D")],
)
def test_auto_inference_capped_on_short_seasonal_panels(n, freq):
    """Auto-inferred m larger than the train slice falls back to m=1 (no crash)."""
    sel = AutoSelect().fit(_freq_panel(n, freq))
    assert sel.best_models_["s1"]
    assert len(sel.predict(3)) == 3


def test_perfectly_periodic_series_picks_seasonal_winner():
    """All-NaN mase (zero seasonal-naive scale) falls back to mae ranking."""
    pattern = np.array([10.0, 12.0, 15.0, 11.0, 9.0, 20.0, 25.0])
    y = np.tile(pattern, 12)[:80]
    df = pd.DataFrame(
        {
            ID_COL: "s1",
            TIME_COL: pd.date_range("2024-01-01", periods=80, freq="D"),
            "y": y,
        }
    )
    sel = AutoSelect().fit(df)
    assert sel.best_models_["s1"] == "SeasonalNaive"
    pred = sel.predict(7)["yhat"].to_numpy()
    np.testing.assert_allclose(pred, np.tile(pattern, 2)[80 - 77 : 80 - 77 + 7], atol=1e-9)


# --- v0.9.0 verifier regressions ------------------------------------------


def _monthly_panel(lengths: dict[str, int]) -> pd.DataFrame:
    """Panel of monthly series with the given per-series lengths."""
    rng = np.random.default_rng(7)
    frames = [
        pd.DataFrame(
            {
                ID_COL: uid,
                TIME_COL: pd.date_range("2020-01-31", periods=n, freq="ME"),
                "y": 100.0 + np.arange(n) + rng.normal(0, 0.5, n),
            }
        )
        for uid, n in lengths.items()
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("val_h,n_windows", [(12, 2), (6, 2), (4, 3)])
@pytest.mark.parametrize("offset", [0, 1, 2, 3, 4, 5])
def test_default_pool_fits_across_the_cv_span_boundary(val_h, n_windows, offset):
    """Series just longer than the CV span must fit, not crash.

    Through v0.8.0 ``use_cv`` was ``sizes.min() > span``, which only guarantees
    cross_validation()'s own precondition. It never checked that the first
    fold's training slice (``n - span`` rows) was long enough for the
    candidates, so for ``n`` in ``span+1 .. span+3`` the pool was fitted on 1-3
    rows and Drift/Holt/AutoETS raised -- e.g. ``AutoSelect().fit`` on 25
    monthly points died with "Drift requires at least 2 observations per
    series; series 's1' has 1". The module contract is a 75/25 holdout fallback
    whenever the panel is too short for CV, and "too short" has to include a
    first fold the candidates cannot be fitted on.
    """
    n = val_h * n_windows + offset
    df = _monthly_panel({"s1": n})
    sel = AutoSelect(val_h=val_h, n_windows=n_windows).fit(df)
    assert sel.best_models_["s1"]
    assert len(sel.predict(3)) == 3


def test_one_short_series_does_not_poison_a_long_panel():
    """A single short series must not break the whole panel's fit.

    ``sizes.min()`` is a panel-wide reduction, so before the fix a 3-series
    panel of lengths [120, 120, 25] chose CV for everyone and then died on the
    25-row series with a message naming 1 observation -- unactionable for a user
    whose shortest series has 25 rows.
    """
    df = _monthly_panel({"a": 120, "b": 120, "c": 25})
    sel = AutoSelect().fit(df)
    assert set(sel.best_models_) == {"a", "b", "c"}
    assert set(sel.predict(3)[ID_COL]) == {"a", "b", "c"}


def test_cv_span_boundary_falls_back_to_the_documented_holdout(monkeypatch):
    """The boundary band scores on the 75/25 holdout, not on a 1-row CV fold."""
    captured = _spy_evaluate(monkeypatch)
    df = _monthly_panel({"s1": 25})  # span = 24 -> first CV fold would train on 1 row
    AutoSelect().fit(df)
    train_df = captured["train_df"]
    assert (train_df.groupby(ID_COL).size() == 25 - 25 // 4).all()  # holdout split, not n - span


def test_cv_still_used_once_the_first_fold_can_fit_the_pool(monkeypatch):
    """CV must not be abandoned wholesale: it still runs as soon as it is sound."""
    captured = _spy_evaluate(monkeypatch)
    df = _monthly_panel({"s1": 28})  # span = 24 -> first fold trains on 4 rows
    AutoSelect().fit(df)
    assert (captured["train_df"].groupby(ID_COL).size() == 4).all()


def _daily_seasonal_panel(n: int) -> pd.DataFrame:
    t = np.arange(n)
    return pd.DataFrame(
        {
            ID_COL: "s1",
            TIME_COL: pd.date_range("2020-01-05", periods=n, freq="D"),
            "y": 100.0 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 12),
        }
    )


@pytest.mark.parametrize("n,val_h,n_windows", [(13, 1, 1), (14, 2, 1), (14, 1, 2)])
def test_feasible_cv_is_not_swapped_for_a_narrower_holdout(n, val_h, n_windows):
    """The short-panel fallback must only fire when the holdout is actually wider.

    The ``_CV_TRAIN_MARGIN`` gate abandoned CV whenever the first fold trained on
    fewer than ``max(min_train_size) + 1`` rows, and jumped to a 75/25 holdout
    that nothing revalidated. But ``holdout_train_len = n - n // 4`` is *smaller*
    than ``cv_train_len = n - span`` whenever ``n // 4 > span``, so a feasible CV
    was traded for an infeasible holdout: 13 daily rows with
    ``season_length=12`` fold-train on exactly 12 rows -- SeasonalNaive's
    declared minimum -- while the holdout keeps only 10, and the fit raised
    "SeasonalNaive requires at least 12 observations per series". A fix for a
    crash must not crash on data that fitted before, so the fallback is now
    conditional on the holdout keeping more training rows.
    """
    sel = AutoSelect(val_h=val_h, n_windows=n_windows, season_length=12)
    sel.fit(_daily_seasonal_panel(n))
    assert sel.best_models_["s1"]
    assert len(sel.predict(3)) == 3


@pytest.mark.parametrize("candidate", [SeasonalNaive(season_length=176), RidgeLag(lags=166)])
def test_wide_candidate_keeps_cv_when_the_holdout_is_too_narrow(candidate, monkeypatch):
    """Same regression on long panels: a hungry candidate plus a wide CV window.

    200 rows with ``val_h=12, n_windows=2`` train the first fold on 176 rows,
    which is exactly what ``SeasonalNaive(176)``/``RidgeLag(166)`` need; the
    75/25 holdout keeps only 150 and cannot fit either. The margin sent both to
    the holdout and the fit raised. CV must be kept here -- and it must really be
    CV, not a silently widened holdout.
    """
    captured = _spy_evaluate(monkeypatch)
    sel = AutoSelect(candidates=["naive", candidate], val_h=12, n_windows=2)
    sel.fit(_daily_seasonal_panel(200))
    assert (captured["train_df"].groupby(ID_COL).size() == 176).all()
    assert sel.best_models_["s1"]
