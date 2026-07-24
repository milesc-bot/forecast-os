"""Tests for exogenous covariate support in the core contract and RidgeLag.

Covariates are extra numeric columns in the fit panel. Models opt in via
``supports_exog``; everything else warns (never silently ignores) and keeps
working exactly as before.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.base import IgnoredCovariatesWarning
from forecast_os.core.exceptions import DataContractError, ForecastOSError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL
from forecast_os.evaluation.backtest import cross_validation
from forecast_os.evaluation.metrics import mae
from forecast_os.models.baselines import Naive
from forecast_os.models.ml import RidgeLag


def _driver_panel(
    n: int = 160, n_series: int = 1, beta: float = 2.5, noise: float = 0.4, seed: int = 3
) -> pd.DataFrame:
    """Panel where y is driven by an i.i.d. 'spend' covariate: y_t = f(spend_t) + eps.

    Because spend is i.i.d., lags of y carry no information about future y;
    only a model that consumes the covariate can forecast well.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        spend = rng.uniform(0.0, 10.0, n)
        y = 20.0 + 5.0 * i + beta * spend + noise * rng.standard_normal(n)
        frames.append(
            pd.DataFrame(
                {
                    ID_COL: f"s{i}",
                    TIME_COL: pd.date_range("2024-01-01", periods=n, freq="D"),
                    TARGET_COL: y,
                    "spend": spend,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _split(df: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = df.groupby(ID_COL)[TARGET_COL].transform("size").to_numpy()
    pos = df.groupby(ID_COL).cumcount().to_numpy()
    return df[pos < n - h], df[pos >= n - h]


class PlainRidge(RidgeLag):
    """RidgeLag with covariate support switched off (control model)."""

    supports_exog = False
    alias = "ridge_plain"


# -- flags ---------------------------------------------------------------------


def test_supports_exog_flags():
    assert Naive.supports_exog is False
    assert RidgeLag.supports_exog is True


# -- exog improves accuracy ----------------------------------------------------


def test_ridge_lag_exog_improves_holdout_mae():
    h = 20
    df = _driver_panel(n=160, seed=3)
    train, test = _split(df, h)

    with_exog = RidgeLag(lags=5).fit(train)
    pred_exog = with_exog.predict(h, X_df=test[[ID_COL, TIME_COL, "spend"]])

    without = RidgeLag(lags=5).fit(train[[ID_COL, TIME_COL, TARGET_COL]])
    pred_plain = without.predict(h)

    y_true = test[TARGET_COL].to_numpy()
    mae_exog = mae(y_true, pred_exog["yhat"].to_numpy())
    mae_plain = mae(y_true, pred_plain["yhat"].to_numpy())
    assert mae_exog < 0.6 * mae_plain


def test_exog_fit_stores_sorted_feature_names():
    df = _driver_panel(n=60)
    df["adspend_b"] = df["spend"] * 0.5
    model = RidgeLag(lags=5).fit(df)
    assert model._exog_features == ["adspend_b", "spend"]


# -- the silent-ignore trap ----------------------------------------------------


def test_non_exog_model_warns_naming_ignored_columns():
    df = _driver_panel(n=40)
    df["clicks"] = np.arange(len(df), dtype=float)
    with pytest.warns(IgnoredCovariatesWarning) as rec:
        Naive().fit(df)
    exog_warnings = [w for w in rec if issubclass(w.category, IgnoredCovariatesWarning)]
    assert len(exog_warnings) == 1, "expected exactly one warning per fit call"
    msg = str(exog_warnings[0].message)
    assert "clicks" in msg and "spend" in msg


def test_non_numeric_extras_ignored_silently():
    df = _driver_panel(n=40)[[ID_COL, TIME_COL, TARGET_COL]].copy()
    df["region"] = "emea"
    with warnings.catch_warnings():
        warnings.simplefilter("error", IgnoredCovariatesWarning)
        Naive().fit(df)


def test_no_warning_on_plain_panel():
    df = _driver_panel(n=40)[[ID_COL, TIME_COL, TARGET_COL]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", IgnoredCovariatesWarning)
        Naive().fit(df)


# -- non-finite covariates rejected up front -----------------------------------


def test_nan_in_fit_covariates_raises_naming_column():
    df = _driver_panel(n=60)
    df.loc[10, "spend"] = np.nan
    with pytest.raises(DataContractError, match="spend"):
        RidgeLag(lags=5).fit(df)


def test_nan_in_x_df_raises_naming_column():
    df = _driver_panel(n=60)
    train, test = _split(df, 5)
    model = RidgeLag(lags=5).fit(train)
    x_df = test[[ID_COL, TIME_COL, "spend"]].copy()
    x_df.iloc[2, x_df.columns.get_loc("spend")] = np.nan
    with pytest.raises(DataContractError, match="spend"):
        model.predict(5, X_df=x_df)


# -- X_df validation at predict ------------------------------------------------


def test_predict_without_x_df_after_exog_fit_raises():
    model = RidgeLag(lags=5).fit(_driver_panel(n=60))
    with pytest.raises(ForecastOSError, match="X_df"):
        model.predict(5)


def test_x_df_on_model_without_exog_warns_and_ignores():
    df = _driver_panel(n=60)
    plain = RidgeLag(lags=5).fit(df[[ID_COL, TIME_COL, TARGET_COL]])
    x_df = df.tail(5)[[ID_COL, TIME_COL, "spend"]]
    with pytest.warns(IgnoredCovariatesWarning):
        pred = plain.predict(5, X_df=x_df)
    pd.testing.assert_frame_equal(pred, plain.predict(5))


def test_x_df_wrong_row_count_raises():
    df = _driver_panel(n=60)
    train, test = _split(df, 10)
    model = RidgeLag(lags=5).fit(train)
    short = test[[ID_COL, TIME_COL, "spend"]].head(7)
    with pytest.raises(ForecastOSError, match="exactly 10"):
        model.predict(10, X_df=short)


def test_x_df_missing_feature_column_raises():
    df = _driver_panel(n=60)
    train, test = _split(df, 5)
    model = RidgeLag(lags=5).fit(train)
    with pytest.raises(ForecastOSError, match="spend"):
        model.predict(5, X_df=test[[ID_COL, TIME_COL]])


def test_x_df_missing_series_raises():
    df = _driver_panel(n=80, n_series=2)
    train, test = _split(df, 5)
    model = RidgeLag(lags=5).fit(train)
    only_s0 = test[test[ID_COL] == "s0"][[ID_COL, TIME_COL, "spend"]]
    with pytest.raises(ForecastOSError, match="s1"):
        model.predict(5, X_df=only_s0)


def test_x_df_aligned_by_ds_sort_order():
    h = 6
    df = _driver_panel(n=60)
    train, test = _split(df, h)
    model = RidgeLag(lags=5).fit(train)
    x_df = test[[ID_COL, TIME_COL, "spend"]]
    shuffled = x_df.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(
        model.predict(h, X_df=shuffled), model.predict(h, X_df=x_df)
    )


# -- cross_validation auto-threads covariates ----------------------------------


def test_cv_threads_exog_and_suppresses_covariate_warnings():
    df = _driver_panel(n=90, n_series=2, seed=11)
    exog_model = RidgeLag(lags=5)
    exog_model.alias = "ridge_exog"

    with warnings.catch_warnings():
        # any IgnoredCovariatesWarning escaping cross_validation fails the test
        warnings.simplefilter("error", IgnoredCovariatesWarning)
        cv = cross_validation(
            df, models=[exog_model, PlainRidge(lags=5), "naive"], h=8, n_windows=2
        )

    for col in ("ridge_exog", "ridge_plain", "Naive"):
        assert col in cv.columns
        assert np.isfinite(cv[col]).all()

    # the exog-aware alias beats the identical model without covariates
    y = cv[TARGET_COL].to_numpy()
    assert mae(y, cv["ridge_exog"].to_numpy()) < 0.6 * mae(y, cv["ridge_plain"].to_numpy())


def test_cv_without_covariates_unchanged(panel):
    cv = cross_validation(panel, models=[RidgeLag(lags=7), "naive"], h=6, n_windows=2)
    assert {"RidgeLag", "Naive"} <= set(cv.columns)
    assert len(cv) == 3 * 6 * 2
    assert np.isfinite(cv["RidgeLag"]).all()


class BadlyDeclared(Naive):
    """Declares supports_exog=True but its ``_fit_series`` takes no ``X``.

    Such a model is fitted with no exog features; cross_validation must not
    hand it an ``X_df`` (which would emit IgnoredCovariatesWarning outside the
    per-fit suppression block, once per window).
    """

    supports_exog = True
    alias = "badly_declared"


def test_cv_badly_declared_exog_model_emits_no_escaping_warnings():
    df = _driver_panel(n=90, seed=7)
    with warnings.catch_warnings():
        # any IgnoredCovariatesWarning escaping cross_validation fails the test
        warnings.simplefilter("error", IgnoredCovariatesWarning)
        cv = cross_validation(df, models=[BadlyDeclared()], h=8, n_windows=3)
    assert "badly_declared" in cv.columns
    assert np.isfinite(cv["badly_declared"]).all()


# -- RidgeLag alpha=0 degenerate designs ---------------------------------------


def test_ridge_alpha_zero_constant_y_fits_and_predicts_finite():
    n = 40
    df = pd.DataFrame(
        {
            ID_COL: "s0",
            TIME_COL: pd.date_range("2024-01-01", periods=n, freq="D"),
            TARGET_COL: 7.0,
        }
    )
    pred = RidgeLag(lags=5, alpha=0.0).fit(df).predict(6)
    assert np.isfinite(pred["yhat"]).all()
    np.testing.assert_allclose(pred["yhat"], 7.0, atol=1e-8)


def test_ridge_alpha_zero_constant_covariate_fits_and_predicts_finite():
    df = _driver_panel(n=60)
    df["flat"] = 1.0  # constant column standardizes to all-zero: singular OLS
    train, test = _split(df, 5)
    model = RidgeLag(lags=5, alpha=0.0).fit(train)
    pred = model.predict(5, X_df=test[[ID_COL, TIME_COL, "flat", "spend"]])
    assert np.isfinite(pred["yhat"]).all()
