"""Tests for models/mstl.py: additive multi-seasonal decomposition + MSTL."""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import get_model, list_models
from forecast_os.core.types import to_panel
from forecast_os.models.mstl import MSTL, STLResult, stl_decompose

H = 14


def _two_seasonal(n=280, seed=0, noise=0.3, slope=0.05):
    """Trend + a period-7 sine + a period-28 cosine + Gaussian noise.

    ``n`` is a multiple of 28 so both true components have exactly zero mean
    and neither aliases into the other's cycle-subseries averages.
    """
    t = np.arange(n, dtype=float)
    weekly = 5.0 * np.sin(2 * np.pi * t / 7)
    monthly = 3.0 * np.cos(2 * np.pi * t / 28)
    rng = np.random.default_rng(seed)
    y = 20.0 + slope * t + weekly + monthly + noise * rng.standard_normal(n)
    return y, weekly, monthly


def _split(y, h):
    return y[:-h], y[-h:]


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _model_mae(model, train, test, h):
    pred = model.fit(train).predict(h)
    merged = test.merge(pred, on=["unique_id", "ds"], validate="one_to_one")
    assert len(merged) == len(test)
    return float((merged["y"] - merged["yhat"]).abs().mean())


def _seasonal_naive_mae(y_train, y_test, m):
    season = y_train[-m:]
    fc = season[np.arange(len(y_test)) % m]
    return _mae(fc, y_test)


# -- stl_decompose -------------------------------------------------------------


class TestSTLDecompose:
    def test_result_shape_and_sorted_periods(self):
        y, _, _ = _two_seasonal()
        res = stl_decompose(y, [28, 7, 7])
        assert isinstance(res, STLResult)
        assert res.periods == (7, 28)
        assert list(res.seasonal) == [7, 28]
        assert res.trend.shape == y.shape
        assert res.remainder.shape == y.shape
        for s in res.seasonal.values():
            assert s.shape == y.shape

    def test_exact_recomposition_single_period(self):
        y, _, _ = _two_seasonal()
        res = stl_decompose(y, 7)
        recon = res.trend + res.seasonal[7] + res.remainder
        assert np.array_equal(recon, y)

    def test_exact_recomposition_two_periods(self):
        y, _, _ = _two_seasonal()
        res = stl_decompose(y, [7, 28])
        recon = res.trend + sum(res.seasonal.values()) + res.remainder
        assert np.array_equal(recon, y), "recomposition must be exact, not approximate"

    def test_recompose_helper_matches_input(self):
        y, _, _ = _two_seasonal(n=140, seed=3)
        for robust in (False, True):
            res = stl_decompose(y, [7, 28], robust=robust)
            assert np.array_equal(res.recompose(), y)

    def test_recomposition_error_never_exceeds_rounding(self):
        """Bit-exact recomposition is guaranteed only up to IEEE round-off.

        When ``trend + Σ seasonal`` stays within a factor of two of ``y``
        (any sanely scaled series) Sterbenz's lemma makes the residual
        subtraction exact, so recomposition is bit-exact — that is what the
        tests above assert. On a series straddling zero the ratio can leave
        that band and recomposition may land a fraction of an ulp away.
        """
        for seed in range(6):
            rng = np.random.default_rng(seed)
            for scale in (1e-6, 1.0, 1e4, 1e8):
                t = np.arange(120.0)
                y = scale * (
                    rng.standard_normal(120)
                    + 3 * np.sin(2 * np.pi * t / 7)
                    + 0.05 * t
                    + rng.uniform(-50, 50)
                )
                res = stl_decompose(y, [7, 28])
                total = res.trend + sum(res.seasonal.values())
                err = float(np.max(np.abs(res.recompose() - y)))
                ulp = float(np.spacing(float(np.max(np.abs(np.concatenate([y, total]))))))
                assert err <= 8 * ulp

    def test_seasonal_components_are_mean_centered(self):
        y, _, _ = _two_seasonal(n=283, seed=1)  # length NOT a multiple of 7 or 28
        res = stl_decompose(y, [7, 28])
        for s in res.seasonal.values():
            assert abs(float(np.mean(s))) < 1e-10

    def test_seasonal_components_are_exactly_periodic(self):
        y, _, _ = _two_seasonal()
        res = stl_decompose(y, [7, 28])
        for p, s in res.seasonal.items():
            assert np.allclose(s[:-p], s[p:], atol=1e-12)

    def test_recovers_both_seasonal_components(self):
        y, weekly, monthly = _two_seasonal()
        res = stl_decompose(y, [7, 28], iterations=3)
        assert np.max(np.abs(res.seasonal[7] - weekly)) < 0.5
        assert np.max(np.abs(res.seasonal[28] - monthly)) < 0.5

    def test_recovers_non_nested_periods(self):
        # 5 and 28 do not divide one another, so the trend window (28) only
        # attenuates rather than annihilates the period-5 harmonic.
        t = np.arange(280.0)
        five = 4.0 * np.sin(2 * np.pi * t / 5)
        long = 3.0 * np.cos(2 * np.pi * t / 28)
        rng = np.random.default_rng(1)
        y = 50.0 + 0.03 * t + five + long + 0.2 * rng.standard_normal(280)
        res = stl_decompose(y, [5, 28], iterations=3)
        assert np.max(np.abs(res.seasonal[5] - five)) < 0.5
        assert np.max(np.abs(res.seasonal[28] - long)) < 0.5

    def test_trend_tracks_linear_trend_in_interior(self):
        y, _, _ = _two_seasonal()
        res = stl_decompose(y, [7, 28], iterations=3)
        t = np.arange(len(y), dtype=float)
        truth = 20.0 + 0.05 * t
        interior = slice(30, -30)
        assert np.max(np.abs(res.trend[interior] - truth[interior])) < 0.5

    def test_remainder_is_small_relative_to_signal(self):
        y, _, _ = _two_seasonal()
        res = stl_decompose(y, [7, 28], iterations=3)
        assert np.std(res.remainder) < 0.25 * np.std(y)

    def test_robust_median_resists_a_single_outlier(self):
        y, weekly, _ = _two_seasonal(n=280, noise=0.05)
        y = y.copy()
        y[100] += 200.0  # one gross outlier inside a single cycle-subseries cell
        plain = stl_decompose(y, 7)
        robust = stl_decompose(y, 7, robust=True)
        err_plain = np.max(np.abs(plain.seasonal[7] - weekly))
        err_robust = np.max(np.abs(robust.seasonal[7] - weekly))
        assert err_robust < err_plain
        assert err_robust < 1.0

    def test_period_longer_than_series_warns_and_is_dropped(self):
        y, _, _ = _two_seasonal(n=56)
        with pytest.warns(UserWarning, match="seasonal period"):
            res = stl_decompose(y, [7, 90])
        assert res.periods == (7,)
        assert 90 not in res.seasonal
        assert np.array_equal(res.recompose(), y)

    def test_all_periods_dropped_gives_trivial_decomposition(self):
        y = np.arange(10.0)
        with pytest.warns(UserWarning, match="seasonal period"):
            res = stl_decompose(y, 20)
        assert res.periods == ()
        assert res.seasonal == {}
        assert np.array_equal(res.trend, y)
        assert np.array_equal(res.remainder, np.zeros(10))

    @pytest.mark.parametrize("bad", [1, 0, -7, [7, 1], [], 7.5, "7", None, [7, "a"]])
    def test_bad_periods_raise(self, bad):
        y, _, _ = _two_seasonal(n=56)
        with pytest.raises(ForecastOSError):
            stl_decompose(y, bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_bad_iterations_raise(self, bad):
        y, _, _ = _two_seasonal(n=56)
        with pytest.raises(ForecastOSError):
            stl_decompose(y, 7, iterations=bad)

    def test_int_and_sequence_periods_agree(self):
        y, _, _ = _two_seasonal(n=140)
        a = stl_decompose(y, 7)
        b = stl_decompose(y, [7, 7])
        assert a.periods == b.periods == (7,)
        assert np.array_equal(a.seasonal[7], b.seasonal[7])

    def test_deterministic(self):
        y, _, _ = _two_seasonal(n=140, seed=9)
        a = stl_decompose(y, [7, 28], iterations=3)
        b = stl_decompose(y, [7, 28], iterations=3)
        assert np.array_equal(a.trend, b.trend)
        assert np.array_equal(a.seasonal[28], b.seasonal[28])


# -- MSTL ----------------------------------------------------------------------


class TestMSTL:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("mstl", periods=7), MSTL)
        listed = list_models(family="statistical")
        assert "mstl" in set(listed["name"])

    def test_params_stored_and_clone(self):
        model = MSTL(periods=[28, 7, 7], base_model="naive", iterations=3)
        assert model.get_params() == {
            "periods": (7, 28),
            "base_model": "naive",
            "base_params": None,
            "iterations": 3,
        }
        clone = model.clone()
        assert type(clone) is MSTL
        assert clone.get_params() == model.get_params()

    @pytest.mark.parametrize("bad", [1, 0, [], [7, 0], "weekly", None])
    def test_bad_periods_raise(self, bad):
        with pytest.raises(ForecastOSError):
            MSTL(periods=bad)

    def test_bad_iterations_raise(self):
        with pytest.raises(ForecastOSError):
            MSTL(periods=7, iterations=0)

    def test_unknown_base_model_raises(self):
        y, _, _ = _two_seasonal(n=140)
        with pytest.raises(ForecastOSError, match="base_model"):
            MSTL(periods=7, base_model="no_such_model").fit(to_panel(y))

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            MSTL(periods=7).predict(3)

    def test_min_train_size(self):
        # Gated on the shortest period, so a too-long period can still be dropped
        # by stl_decompose instead of blocking the fit outright.
        assert MSTL(periods=[7, 28]).min_train_size == 14
        assert MSTL(periods=7).min_train_size == 14
        with pytest.raises(ForecastOSError):
            MSTL(periods=7).fit(to_panel(np.arange(13.0)))

    def test_period_too_long_for_series_is_dropped_not_fatal(self):
        """A period longer than the series warns and is dropped, rather than raising.

        Regression test: gating min_train_size on max(periods) made this path
        unreachable through fit() — MSTL(periods=(7, 365)) on a short series
        raised "requires at least 730 observations" instead of forecasting
        weekly. Only periods that actually fit should be used.
        """
        rng = np.random.default_rng(0)
        n = 120
        y = 10.0 + np.tile(np.array([3.0, -1.0, -2.0, 0.5, 1.0, -1.5, 0.0]), n // 7 + 1)[:n]
        y = y + rng.normal(0, 0.1, n)

        model = MSTL(periods=[7, 365])
        with pytest.warns(UserWarning, match="365"):
            pred = model.fit(to_panel(y)).predict(H)

        assert len(pred) == H
        assert np.isfinite(pred["yhat"]).all()
        # The weekly component survived: forecasts still swing with the cycle.
        assert pred["yhat"].std() > 0.5

    def test_panel_contract_with_levels(self, panel):
        pred = MSTL(periods=7).fit(panel).predict(H, level=[80])
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["yhat"]).all()
        assert (pred["yhat"] <= pred["hi-80"]).all()
        for _, g in pred.groupby("unique_id"):
            assert g["ds"].is_monotonic_increasing
            width = (g["hi-80"] - g["lo-80"]).to_numpy()
            assert np.all(np.diff(width) >= -1e-9), "interval width must not shrink with h"

    def test_state_decomposition_recomposes_to_y(self, panel):
        model = MSTL(periods=7).fit(panel)
        for state in model._series_state.values():
            total = sum(state["seasonal_"].values())
            assert np.array_equal(state["trend_"] + total + state["remainder_"], state["_y"])

    def test_forecast_is_base_forecast_plus_cyclic_seasonal(self):
        y, _, _ = _two_seasonal(n=280)
        model = MSTL(periods=[7, 28]).fit(to_panel(y))
        state = model._series_state["series-0"]
        n = len(y)
        base_fc = state["base_"]._predict_series(state["base_state_"], H)
        expected = base_fc.astype(float).copy()
        for p, s in state["seasonal_"].items():
            # s is exactly periodic, so future step j sits at phase (n + j) % p
            expected = expected + s[(n + np.arange(H)) % p]
        yhat = model.predict(H)["yhat"].to_numpy()
        assert np.allclose(yhat, expected, atol=1e-12)

    def test_seasonal_extension_matches_the_previous_cycle(self):
        # A pure repeating pattern with no trend: the h-step forecast must
        # repeat the pattern in the right phase.
        pattern = np.array([3.0, -1.0, 0.5, -2.0, 4.0, -1.5, -3.0])
        y = 10.0 + np.tile(pattern, 30)
        model = MSTL(periods=7, base_model="ses").fit(to_panel(y))
        yhat = model.predict(H)["yhat"].to_numpy()
        n = len(y)
        expected_shape = pattern[(n + np.arange(H)) % 7] - pattern.mean()
        # atol covers the edge-padded trend distortion only; adjacent phases are
        # >= 0.5 apart, so a phase-alignment bug could not sneak through.
        assert np.allclose(yhat - yhat.mean(), expected_shape, atol=0.01)

    def test_fitted_values_are_base_fit_plus_seasonal(self):
        y, _, _ = _two_seasonal(n=140)
        model = MSTL(periods=7, base_model="naive").fit(to_panel(y))
        state = model._series_state["series-0"]
        expected = state["base_state_"]["fitted"] + state["seasonal_"][7]
        fitted = model.fitted_values()["fitted"].to_numpy()
        assert np.allclose(fitted[1:], expected[1:])
        assert np.isnan(fitted[0])

    def test_beats_seasonal_naive_two_seasonalities(self):
        y, _, _ = _two_seasonal(n=280)
        train, test = _split(y, 28)
        df = to_panel(y)
        train_df = df.iloc[: len(train)]
        test_df = df.iloc[len(train) :]
        mstl_mae = _model_mae(MSTL(periods=[7, 28]), train_df, test_df, 28)
        assert mstl_mae < _seasonal_naive_mae(train, test, 28)
        assert mstl_mae < _seasonal_naive_mae(train, test, 7)

    def test_single_period_beats_seasonal_naive_on_panel(self, panel):
        train = panel.groupby("unique_id").head(160)
        test = panel.groupby("unique_id").tail(40)
        mstl_mae = _model_mae(MSTL(periods=7), train, test, 40)
        sn_mae = _model_mae(get_model("seasonal_naive", season_length=7), train, test, 40)
        assert mstl_mae < sn_mae

    def test_base_params_forwarded_to_base_model(self):
        y, _, _ = _two_seasonal(n=140)
        model = MSTL(
            periods=7, base_model="window_average", base_params={"window": 5}
        ).fit(to_panel(y))
        state = model._series_state["series-0"]
        assert state["base_"].window == 5
        level = float(np.mean(state["deseason_"][-5:]))
        yhat = model.predict(H)["yhat"].to_numpy()
        n = len(y)
        expected = level + state["seasonal_"][7][(n + np.arange(H)) % 7]
        assert np.allclose(yhat, expected, atol=1e-10)

    def test_multi_series_panel_is_independent(self, panel):
        model = MSTL(periods=7).fit(panel)
        assert set(model._series_state) == set(panel["unique_id"].unique())
        pred = model.predict(5)
        assert isinstance(pred, pd.DataFrame)
        assert len(pred) == 5 * panel["unique_id"].nunique()


class TestAdversarialRegressions:
    """Regressions for defects found by the v0.8.0 adversarial verification wave."""

    def test_period_needs_two_full_cycles_not_one(self):
        """A period estimated from ~1 cycle memorizes noise and collapses the intervals.

        ``_usable_periods`` originally kept any ``p < n``, so at one cycle each
        cycle-subseries cell held a single observation: the "seasonal" component
        absorbed the noise, the remainder collapsed, and sigma collapsed with it.
        Measured at n=29 with periods=[7, 28]: a nominal 95% interval covered 9%.
        """
        rng = np.random.default_rng(0)
        n = 29
        y = 100 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 5, n)

        with pytest.warns(UserWarning, match="28"):
            res = stl_decompose(y, periods=[7, 28])
        assert res.periods == (7,), "a period with fewer than two cycles must be dropped"

        # 2 * 28 == 56 is exactly two cycles, so 28 survives at n=56 but not n=55.
        assert stl_decompose(np.tile(y, 2)[:56], periods=[7, 28]).periods == (7, 28)
        with pytest.warns(UserWarning, match="28"):
            assert stl_decompose(np.tile(y, 2)[:55], periods=[7, 28]).periods == (7,)

    def test_intervals_do_not_collapse_at_short_n(self):
        """The dropped-period fix must restore sane interval widths on pure noise.

        Before the fix, MSTL(periods=[7, 28]) on 29 points of pure noise (sd=10,
        honest 95% width ~39) reported a 95% width of 3.55 — 11x too narrow.
        """
        rng = np.random.default_rng(3)
        n = 29
        y = 100 + rng.normal(0, 10, n)
        with pytest.warns(UserWarning, match="28"):
            pred = MSTL(periods=[7, 28]).fit(to_panel(y)).predict(4, level=[95])
        width = float((pred["hi-95"] - pred["lo-95"]).mean())
        assert width > 15.0, f"95% interval collapsed to width {width:.2f}"

    def test_base_model_data_requirement_is_enforced(self):
        """MSTL fitted the base via _fit_series, bypassing its min_train_size guard.

        With the default periods=7 (MSTL gate 14) and a heavier base, the fit
        silently produced all-NaN forecasts with no error and no warning.
        """
        base_min = get_model("ridge_lag").min_train_size
        model = MSTL(periods=7, base_model="ridge_lag")
        assert model.min_train_size >= base_min

        n = 14
        y = 100.0 + np.arange(n) + 5 * np.sin(2 * np.pi * np.arange(n) / 7)
        with pytest.raises(ForecastOSError, match="requires at least"):
            model.fit(to_panel(y))

    def test_base_model_failure_surfaces_as_forecast_os_error(self):
        """A base blowing up on its own terms must not leak a raw AttributeError."""
        y = 100.0 + np.arange(60.0)
        with pytest.raises(ForecastOSError, match="retention_sbg"):
            MSTL(periods=7, base_model="retention_sbg").fit(to_panel(y))
