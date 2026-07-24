"""Exogenous-regressor support for ARIMA (regression with ARIMA errors).

These exercise the ARIMAX path: OLS on standardized covariates with the
existing CSS machinery fitted on the regression residuals, X threaded through
``predict``/``cross_validation`` exactly like :class:`RidgeLag`. The no-exog
path must stay byte-identical, so a couple of exact forecasts are pinned.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL, to_panel
from forecast_os.evaluation.backtest import cross_validation
from forecast_os.models.arima import ARIMA, AutoARIMA
from forecast_os.models.baselines import Naive


def _driver_panel(n_train, h, seed=0, beta=4.0, phi_noise=0.5, uid="series-0"):
    """Single-series panel ``y_t = beta*x_t + AR(1) noise`` with an ``x`` column.

    ``x`` is an i.i.d. driver (unpredictable from the target's own past), so a
    no-exog ARIMA cannot exploit it while an exog ARIMA that is handed the known
    future ``x`` can. Returns ``(train, test)`` split at ``n_train`` with ``h``
    held-out rows.
    """
    rng = np.random.default_rng(seed)
    total = n_train + h
    x = rng.standard_normal(total)
    eps = np.empty(total)
    eps[0] = rng.standard_normal()
    for t in range(1, total):
        eps[t] = phi_noise * eps[t - 1] + rng.standard_normal()
    y = beta * x + eps
    df = pd.DataFrame(
        {ID_COL: uid, TIME_COL: np.arange(total), TARGET_COL: y, "x": x}
    )
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


# -- opt-in flag ---------------------------------------------------------------


def test_supports_exog_flag():
    assert ARIMA.supports_exog is True
    assert AutoARIMA.supports_exog is True


# -- exog lifts accuracy -------------------------------------------------------


def test_exog_beats_no_exog_holdout():
    train, test = _driver_panel(n_train=300, h=30, seed=0)
    X_df = test[[ID_COL, TIME_COL, "x"]]

    exog = ARIMA(order=(1, 0, 0)).fit(train)
    exog_mae = _mae(test[TARGET_COL], exog.predict(30, X_df=X_df)["yhat"])

    noexog = ARIMA(order=(1, 0, 0)).fit(train.drop(columns="x"))
    noexog_mae = _mae(test[TARGET_COL], noexog.predict(30)["yhat"])

    assert exog_mae < 0.6 * noexog_mae


def test_coefficient_direction_recovered_via_prediction_deltas():
    train, _ = _driver_panel(n_train=400, h=20, seed=1, beta=4.0)
    model = ARIMA(order=(1, 0, 0)).fit(train)
    h = 12
    ds = np.arange(400, 400 + h)
    lo = pd.DataFrame({ID_COL: "series-0", TIME_COL: ds, "x": np.zeros(h)})
    hi = pd.DataFrame({ID_COL: "series-0", TIME_COL: ds, "x": np.ones(h)})

    yhat_hi = model.predict(h, X_df=hi)["yhat"].to_numpy()
    yhat_lo = model.predict(h, X_df=lo)["yhat"].to_numpy()
    delta = yhat_hi - yhat_lo
    # The ARIMA-residual forecast is identical across both calls, so the delta
    # isolates the regression effect: a +1 change in x lifts yhat by ~beta=4.
    assert np.all(delta > 0)
    assert abs(float(delta.mean()) - 4.0) < 0.5


# -- X_df validation -----------------------------------------------------------


def test_predict_without_x_df_after_exog_fit_raises():
    train, _ = _driver_panel(n_train=120, h=10, seed=2)
    model = ARIMA(order=(1, 0, 0)).fit(train)
    with pytest.raises(ForecastOSError):
        model.predict(10)


def test_x_df_missing_covariate_column_raises():
    train, test = _driver_panel(n_train=120, h=10, seed=3)
    model = ARIMA(order=(1, 0, 0)).fit(train)
    bad = test[[ID_COL, TIME_COL]].copy()  # dropped the 'x' column
    with pytest.raises(ForecastOSError):
        model.predict(10, X_df=bad)


def test_x_df_wrong_row_count_raises():
    train, test = _driver_panel(n_train=120, h=10, seed=4)
    model = ARIMA(order=(1, 0, 0)).fit(train)
    short = test[[ID_COL, TIME_COL, "x"]].iloc[:5]  # only 5 of 10 future rows
    with pytest.raises(ForecastOSError):
        model.predict(10, X_df=short)


# -- exog intervals + fitted values --------------------------------------------


def test_exog_intervals_finite_and_widen():
    train, test = _driver_panel(n_train=200, h=15, seed=5)
    model = ARIMA(order=(1, 0, 0)).fit(train)
    pred = model.predict(15, level=[80], X_df=test[[ID_COL, TIME_COL, "x"]])
    for col in ("yhat", "lo-80", "hi-80"):
        assert np.isfinite(pred[col]).all()
    assert (pred["lo-80"] <= pred["yhat"]).all()
    assert (pred["yhat"] <= pred["hi-80"]).all()
    width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
    assert np.all(np.diff(width) >= -1e-9)


def test_fitted_values_combine_regression_and_warmup():
    train, _ = _driver_panel(n_train=150, h=10, seed=6)
    model = ARIMA(order=(2, 0, 1)).fit(train)
    fv = model.fitted_values()
    fitted = fv["fitted"].to_numpy()
    # warm-up for (2,0,1): d + max(p, q) = 0 + 2 = 2 NaNs, rest finite
    assert np.isnan(fitted[:2]).all()
    assert np.isfinite(fitted[2:]).all()
    # combined fit (Xb + arima) tracks y far better than a mean-only baseline
    y = fv[TARGET_COL].to_numpy()
    resid = _mae(y[2:], fitted[2:])
    assert resid < _mae(y[2:], np.full(len(y) - 2, y.mean()))


# -- byte-identical no-exog path ----------------------------------------------


def test_no_exog_arima_forecasts_byte_identical():
    # Pinned from a fixed seed BEFORE exog support was added; the no-exog code
    # path must remain byte-for-byte unchanged.
    rng = np.random.default_rng(12345)
    n = 200
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + rng.standard_normal()
    pred = ARIMA(order=(1, 0, 0)).fit(to_panel(y)).predict(5)["yhat"].to_numpy()
    assert pred.tolist() == [
        -0.14165596579743853,
        -0.07906974006872589,
        -0.04143385149488216,
        -0.018801713316705472,
        -0.00519199934619324,
    ]

    rng2 = np.random.default_rng(7)
    y2 = 10.0 + 0.5 * np.arange(120) + rng2.standard_normal(120)
    pred2 = ARIMA(order=(1, 1, 1)).fit(to_panel(y2)).predict(4)["yhat"].to_numpy()
    assert pred2.tolist() == [
        69.76587125474003,
        69.74617769033561,
        69.75148141224142,
        69.75005305394748,
    ]


def test_no_exog_auto_arima_forecasts_byte_identical():
    rng = np.random.default_rng(2024)
    y = 5.0 + np.cumsum(rng.standard_normal(160)) + 0.3 * np.arange(160)
    model = AutoARIMA().fit(to_panel(y))
    assert model._series_state["series-0"]["order_"] == (1, 1, 1)
    pred = model.predict(5)["yhat"].to_numpy()
    assert pred.tolist() == [
        52.711554282491115,
        52.99149892837318,
        53.26864412779644,
        53.54301787522546,
        53.81464788518019,
    ]


# -- cross_validation auto-threads X ------------------------------------------


def test_cross_validation_auto_threads_exog():
    train, _ = _driver_panel(n_train=200, h=0, seed=8)  # full panel for CV
    panel = train  # 200 rows with an 'x' covariate column

    arimax = ARIMA(order=(1, 0, 0))
    arimax.alias = "arimax"
    baseline = Naive()
    baseline.alias = "no_exog"

    out = cross_validation(panel, models=[arimax, baseline], h=20, n_windows=2)
    exog_mae = _mae(out[TARGET_COL], out["arimax"])
    noexog_mae = _mae(out[TARGET_COL], out["no_exog"])
    assert exog_mae < noexog_mae

    # And the covariate specifically helps ARIMA: the same model over the panel
    # with 'x' dropped (auto-threading finds nothing) does strictly worse.
    plain = ARIMA(order=(1, 0, 0))
    plain.alias = "arima_plain"
    out_no_x = cross_validation(
        panel.drop(columns="x"), models=[plain], h=20, n_windows=2
    )
    assert exog_mae < _mae(out_no_x[TARGET_COL], out_no_x["arima_plain"])


def test_auto_arima_exog_beats_no_exog_on_driver_series():
    train, test = _driver_panel(n_train=250, h=25, seed=9)
    X_df = test[[ID_COL, TIME_COL, "x"]]

    exog = AutoARIMA(max_p=2, max_d=1, max_q=2).fit(train)
    exog_mae = _mae(test[TARGET_COL], exog.predict(25, X_df=X_df)["yhat"])

    noexog = AutoARIMA(max_p=2, max_d=1, max_q=2).fit(train.drop(columns="x"))
    noexog_mae = _mae(test[TARGET_COL], noexog.predict(25)["yhat"])

    assert exog_mae < 0.6 * noexog_mae
