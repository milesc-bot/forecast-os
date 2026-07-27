"""Tests for models/sarima.py: multiplicative seasonal ARIMA and AutoSARIMA.

The three numerically delicate pieces — polynomial expansion, the
differencing/integration round trip, and the seasonally-integrated psi
weights — are unit-tested directly against hand-computed values before the
model-level behaviour is exercised.
"""

import numpy as np
import pytest

from forecast_os.core.exceptions import ForecastOSError, NotFittedError
from forecast_os.core.registry import get_model, list_models
from forecast_os.core.types import to_panel
from forecast_os.datasets.synthetic import generate_series
from forecast_os.models.arima import ARIMA
from forecast_os.models.baselines import SeasonalNaive
from forecast_os.models.sarima import (
    SARIMA,
    AutoSARIMA,
    _aicc,
    _centered_ma,
    _expand_ar,
    _expand_ma,
    _fit_sarima_css,
    _sarima_psi,
    _seasonal_caps,
    _seasonal_difference,
    _seasonal_integrate,
    _seasonal_strength,
)

H = 12
EMPTY = np.zeros(0)


def _seasonal_cumsum(x: np.ndarray, m: int) -> np.ndarray:
    """Inverse of one seasonal difference with zero pre-sample values."""
    out = np.asarray(x, dtype=float).copy()
    for t in range(m, len(out)):
        out[t] += out[t - m]
    return out


def _sim_sarima(
    n, phi=(), theta=(), sphi=(), stheta=(), m=12, d=0, D=0, sigma=1.0, seed=0, burn=400
):
    """Simulate a multiplicative SARIMA under the Hamilton (+MA) convention.

    Built by applying the four factors one at a time (never via the expanded
    polynomials), so it is an independent reference for the expansion code:

        (1 - sum phi_i B^i)(1 - sum sPhi_k B^km) w = (1 + sum th_j B^j)(1 + sum sTh_k B^km) e
    """
    rng = np.random.default_rng(seed)
    total = n + burn
    e = sigma * rng.standard_normal(total)
    u = e.copy()
    for j, th in enumerate(theta, start=1):
        u[j:] += th * e[:-j]
    v = u.copy()
    for k, sth in enumerate(stheta, start=1):
        v[k * m :] += sth * u[: -k * m]
    s = np.zeros(total)
    for t in range(total):
        acc = v[t]
        for k, sph in enumerate(sphi, start=1):
            if t - k * m >= 0:
                acc += sph * s[t - k * m]
        s[t] = acc
    x = np.zeros(total)
    for t in range(total):
        acc = s[t]
        for i, ph in enumerate(phi, start=1):
            if t - i >= 0:
                acc += ph * x[t - i]
        x[t] = acc
    y = x[burn:]
    for _ in range(D):
        y = _seasonal_cumsum(y, m)
    for _ in range(d):
        y = np.cumsum(y)
    return y


def _seasonal_panel(length=140, m=12, seed=17, n_series=1):
    return generate_series(
        n_series=n_series, length=length, freq="MS", trend=0.4, seasonality=m,
        season_amp=12.0, noise=1.0, seed=seed,
    )


def _mae(model, train, test, h):
    pred = model.fit(train).predict(h)
    merged = test.merge(pred, on=["unique_id", "ds"], validate="one_to_one")
    assert len(merged) == len(test)
    return float((merged["y"] - merged["yhat"]).abs().mean())


class TestPolynomialExpansion:
    def test_ar_expansion_matches_hand_convolution(self):
        # (1 - 0.5B)(1 - 0.3B^4) = 1 - 0.5B - 0.3B^4 + 0.15B^5
        got = _expand_ar(np.array([0.5]), np.array([0.3]), 4)
        assert np.allclose(got, [0.5, 0.0, 0.0, 0.3, -0.15])

    def test_ma_expansion_matches_hand_convolution(self):
        # (1 + 0.4B)(1 + 0.2B^3) = 1 + 0.4B + 0.2B^3 + 0.08B^4
        got = _expand_ma(np.array([0.4]), np.array([0.2]), 3)
        assert np.allclose(got, [0.4, 0.0, 0.2, 0.08])

    def test_ar_expansion_two_regular_two_seasonal(self):
        # (1 - 0.6B - 0.2B^2)(1 - 0.5B^3)
        #   = 1 - 0.6B - 0.2B^2 - 0.5B^3 + 0.3B^4 + 0.1B^5
        got = _expand_ar(np.array([0.6, 0.2]), np.array([0.5]), 3)
        assert np.allclose(got, [0.6, 0.2, 0.5, -0.3, -0.1])

    def test_ar_expansion_pure_seasonal_order_two(self):
        # (1 - 0.5B^3 + 0.2B^6): sPhi = (0.5, -0.2)
        got = _expand_ar(EMPTY, np.array([0.5, -0.2]), 3)
        assert np.allclose(got, [0.0, 0.0, 0.5, 0.0, 0.0, -0.2])

    def test_ma_expansion_pure_seasonal_order_two(self):
        got = _expand_ma(EMPTY, np.array([0.5, -0.2]), 2)
        assert np.allclose(got, [0.0, 0.5, 0.0, -0.2])

    def test_expansion_without_seasonal_terms_is_identity(self):
        phi = np.array([0.3, -0.2])
        theta = np.array([0.7, 0.1])
        assert np.array_equal(_expand_ar(phi, EMPTY, 12), phi)
        assert np.array_equal(_expand_ma(theta, EMPTY, 12), theta)

    def test_expansion_of_empty_polynomials_is_empty(self):
        assert _expand_ar(EMPTY, EMPTY, 7).shape == (0,)
        assert _expand_ma(EMPTY, EMPTY, 7).shape == (0,)

    def test_m_one_collapses_onto_regular_lags(self):
        # (1 - 0.5B)(1 - 0.3B) = 1 - 0.8B + 0.15B^2
        assert np.allclose(_expand_ar(np.array([0.5]), np.array([0.3]), 1), [0.8, -0.15])
        # (1 + 0.5B)(1 + 0.3B) = 1 + 0.8B + 0.15B^2
        assert np.allclose(_expand_ma(np.array([0.5]), np.array([0.3]), 1), [0.8, 0.15])


class TestDifferencing:
    @staticmethod
    def _y(n=80, seed=1):
        rng = np.random.default_rng(seed)
        t = np.arange(n, dtype=float)
        return 20.0 + 0.3 * t + 5.0 * np.sin(2 * np.pi * t / 4) + rng.standard_normal(n)

    def test_regular_difference_matches_np_diff(self):
        y = self._y()
        w, _ = _seasonal_difference(y, 2, 0, 7)
        assert np.array_equal(w, np.diff(y, n=2))

    def test_seasonal_difference_values(self):
        y = self._y()
        w, _ = _seasonal_difference(y, 0, 1, 4)
        assert np.array_equal(w, y[4:] - y[:-4])

    def test_seasonal_then_regular_order(self):
        y = self._y()
        w, _ = _seasonal_difference(y, 1, 1, 4)
        assert np.allclose(w, np.diff(y[4:] - y[:-4]))

    def test_no_differencing_is_the_series_itself(self):
        y = self._y()
        w, stages = _seasonal_difference(y, 0, 0, 12)
        assert np.array_equal(w, y)
        assert stages == []

    @pytest.mark.parametrize(
        "d,D,m",
        [(0, 0, 4), (1, 0, 4), (2, 0, 4), (0, 1, 4), (1, 1, 4), (0, 2, 3), (2, 2, 5), (1, 1, 1)],
    )
    def test_difference_integrate_round_trip(self, d, D, m):
        y = self._y(n=80, seed=2)
        n0, h = 60, 12
        w_hist, stages = _seasonal_difference(y[:n0], d, D, m)
        assert len(w_hist) == n0 - d - D * m
        w_full, _ = _seasonal_difference(y, d, D, m)
        off = n0 - d - D * m
        # differencing is causal: the history's w is a prefix of the full w
        assert np.allclose(w_hist, w_full[:off], atol=1e-12)
        back = _seasonal_integrate(w_full[off : off + h], stages, m)
        assert back.shape == (h,)
        assert np.allclose(back, y[n0 : n0 + h], atol=1e-9)

    def test_integrating_zero_forecast_repeats_last_season(self):
        y = self._y()
        m, h = 4, 10
        _, stages = _seasonal_difference(y, 0, 1, m)
        out = _seasonal_integrate(np.zeros(h), stages, m)
        assert np.allclose(out, y[-m:][np.arange(h) % m])

    def test_integrating_zero_forecast_with_one_regular_diff_is_flat(self):
        y = self._y()
        _, stages = _seasonal_difference(y, 1, 0, 4)
        out = _seasonal_integrate(np.zeros(8), stages, 4)
        assert np.allclose(out, y[-1])

    def test_too_short_for_seasonal_difference_raises(self):
        with pytest.raises(ForecastOSError):
            _seasonal_difference(np.arange(6.0), 0, 1, 12)

    @pytest.mark.parametrize("m", [3, 4, 5, 12])
    def test_centered_ma_reproduces_a_linear_trend(self, m):
        # a properly centered MA (2xm weights for even m) is exact on a line
        y = 3.0 + 2.0 * np.arange(40.0)
        trend = _centered_ma(y, m)
        ok = ~np.isnan(trend)
        assert ok.sum() == 40 - (m if m % 2 == 0 else m - 1)
        assert np.allclose(trend[ok], y[ok])

    @pytest.mark.parametrize("m", [3, 4])
    def test_centered_ma_annihilates_a_seasonal_cycle(self, m):
        t = np.arange(40.0)
        y = 5.0 + np.cos(2 * np.pi * t / m)
        trend = _centered_ma(y, m)
        ok = ~np.isnan(trend)
        assert np.allclose(trend[ok], 5.0, atol=1e-9)

    def test_seasonal_strength_high_for_seasonal_series(self):
        t = np.arange(120, dtype=float)
        y = 10.0 * np.sin(2 * np.pi * t / 12) + 0.1 * np.random.default_rng(0).standard_normal(120)
        assert _seasonal_strength(y, 12) > 0.9

    def test_seasonal_strength_zero_for_pure_trend(self):
        y = np.arange(120, dtype=float)
        assert _seasonal_strength(y, 12) < 0.1

    def test_seasonal_strength_low_for_white_noise(self):
        y = np.random.default_rng(3).standard_normal(300)
        assert _seasonal_strength(y, 12) < 0.5


class TestPsiWeights:
    def test_ar1_psi_is_geometric(self):
        psi = _sarima_psi(np.array([0.6]), EMPTY, 8, 0, 0, 12)
        assert np.allclose(psi, 0.6 ** np.arange(8))

    def test_ma1_psi_truncates(self):
        psi = _sarima_psi(EMPTY, np.array([0.4]), 5, 0, 0, 12)
        assert np.allclose(psi, [1.0, 0.4, 0.0, 0.0, 0.0])

    def test_seasonal_ma_psi_places_weight_at_lag_m(self):
        # theta expanded for (0,0,0)(0,0,1)_4 is [0, 0, 0, 0.3]
        psi = _sarima_psi(EMPTY, np.array([0.0, 0.0, 0.0, 0.3]), 6, 0, 0, 4)
        assert np.allclose(psi, [1.0, 0.0, 0.0, 0.0, 0.3, 0.0])

    def test_one_regular_integration_gives_ones(self):
        assert np.allclose(_sarima_psi(EMPTY, EMPTY, 6, 1, 0, 12), np.ones(6))

    def test_two_regular_integrations_give_a_ramp(self):
        assert np.allclose(_sarima_psi(EMPTY, EMPTY, 6, 2, 0, 12), np.arange(1, 7))

    def test_one_seasonal_integration_is_a_lag_m_indicator(self):
        m, h = 4, 11
        psi = _sarima_psi(EMPTY, EMPTY, h, 0, 1, m)
        assert np.allclose(psi, (np.arange(h) % m == 0).astype(float))

    def test_two_seasonal_integrations_ramp_every_m(self):
        m, h = 3, 10
        psi = _sarima_psi(EMPTY, EMPTY, h, 0, 2, m)
        expected = np.where(np.arange(h) % m == 0, np.arange(h) // m + 1.0, 0.0)
        assert np.allclose(psi, expected)

    def test_mixed_integration_counts_lattice_points(self):
        # 1 / ((1 - B)(1 - B^m)): coefficient of B^j is floor(j/m) + 1
        m, h = 3, 9
        psi = _sarima_psi(EMPTY, EMPTY, h, 1, 1, m)
        assert np.allclose(psi, np.arange(h) // m + 1.0)

    def test_psi_of_an_expanded_seasonal_ma(self):
        # (1 + 0.4B)(1 + 0.5B^4) is pure MA, so psi is its coefficient vector
        theta_e = _expand_ma(np.array([0.4]), np.array([0.5]), 4)
        psi = _sarima_psi(EMPTY, theta_e, 8, 0, 0, 4)
        assert np.allclose(psi, [1.0, 0.4, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0])

    def test_psi_of_an_expanded_seasonal_ar(self):
        # (1 - 0.5B)(1 - 0.6B^2) inverted: psi_j = 0.5 psi_{j-1} + 0.6 psi_{j-2}
        #                                          - 0.3 psi_{j-3}
        phi_e = _expand_ar(np.array([0.5]), np.array([0.6]), 2)
        psi = _sarima_psi(phi_e, EMPTY, 7, 0, 0, 2)
        ref = np.zeros(7)
        ref[0] = 1.0
        for j in range(1, 7):
            ref[j] = sum(phi_e[i - 1] * ref[j - i] for i in range(1, min(j, len(phi_e)) + 1))
        assert np.allclose(psi, ref)
        # hand-checked: 1, 0.5, 0.5*0.5 + 0.6, 0.5*0.85 + 0.6*0.5 - 0.3
        assert np.allclose(psi[:4], [1.0, 0.5, 0.85, 0.425])

    def test_integration_order_does_not_matter(self):
        # cumsum and seasonal accumulation are both multiplications by a power
        # series in B, so they commute; guard against an order-dependent bug.
        m, h = 4, 20
        psi = _sarima_psi(np.array([0.5]), np.array([0.3]), h, 1, 1, m)
        ref = np.zeros(h)
        ref[0] = 1.0
        for j in range(1, h):
            ref[j] = (0.3 if j == 1 else 0.0) + 0.5 * ref[j - 1]
        for j in range(m, h):
            ref[j] += ref[j - m]
        ref = np.cumsum(ref)
        assert np.allclose(psi, ref)


class TestSARIMA:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("sarima"), SARIMA)
        assert "sarima" in set(list_models(family="statistical")["name"])

    def test_params_stored_and_clone(self):
        model = SARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 1), m=4, include_mean=False)
        assert model.get_params() == {
            "order": (1, 1, 0),
            "seasonal_order": (0, 1, 1),
            "m": 4,
            "include_mean": False,
        }
        clone = model.clone()
        assert type(clone) is SARIMA
        assert clone.get_params() == model.get_params()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"order": (1, -1, 0)},
            {"order": (1, 1)},
            {"order": ("a", 1, 1)},
            {"seasonal_order": (1, 1)},
            {"seasonal_order": (1, 1, -1)},
            {"seasonal_order": (None, 0, 0)},
            {"m": 0},
            {"m": -3},
            {"m": "yearly"},
        ],
    )
    def test_bad_args_raise_forecast_os_error(self, kwargs):
        with pytest.raises(ForecastOSError):
            SARIMA(**kwargs)

    def test_min_train_size_formula(self):
        model = SARIMA(order=(2, 1, 0), seasonal_order=(1, 1, 0), m=4)
        assert model.min_train_size == max(2 + 4, 0) + 1 + 4 + 5

    def test_series_shorter_than_min_train_size_errors_cleanly(self):
        model = SARIMA(order=(1, 0, 0), seasonal_order=(0, 1, 0), m=4)
        n = model.min_train_size - 1
        with pytest.raises(ForecastOSError, match="requires at least"):
            model.fit(to_panel(np.arange(float(n))))

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            SARIMA().predict(3)

    def test_zero_seasonal_order_equals_arima_exactly(self):
        y = _sim_sarima(200, phi=(0.6,), theta=(0.3,), m=12, seed=4)
        panel = to_panel(y)
        sar = SARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0), m=12).fit(panel)
        ari = ARIMA(order=(1, 1, 1)).fit(panel)
        s_state = sar._series_state["series-0"]
        a_state = ari._series_state["series-0"]
        assert s_state["c"] == a_state["c"]
        assert np.array_equal(s_state["phi"], a_state["phi"])
        assert np.array_equal(s_state["theta"], a_state["theta"])
        np.testing.assert_array_equal(s_state["fitted"], a_state["fitted"])
        np.testing.assert_array_equal(
            sar.predict(H)["yhat"].to_numpy(), ari.predict(H)["yhat"].to_numpy()
        )

    def test_m_one_degenerates_to_plain_arima(self):
        # with m = 1 a seasonal difference *is* a regular difference, so
        # SARIMA(1,0,1)(0,1,0)_1 == ARIMA(1,1,1)
        y = _sim_sarima(180, phi=(0.5,), theta=(0.2,), m=12, d=1, seed=9)
        panel = to_panel(y)
        sar = SARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 0), m=1).fit(panel)
        ari = ARIMA(order=(1, 1, 1)).fit(panel)
        s_state = sar._series_state["series-0"]
        a_state = ari._series_state["series-0"]
        assert np.allclose(s_state["phi"], a_state["phi"], atol=1e-12)
        assert np.allclose(s_state["theta"], a_state["theta"], atol=1e-12)
        assert np.allclose(
            sar.predict(H)["yhat"].to_numpy(), ari.predict(H)["yhat"].to_numpy(), atol=1e-8
        )

    def test_pure_seasonal_difference_is_seasonal_naive(self):
        # SARIMA(0,0,0)(0,1,0)_m with no constant reproduces SeasonalNaive
        rng = np.random.default_rng(11)
        y = np.cumsum(rng.standard_normal(60)) + 5.0 * np.sin(np.arange(60) * np.pi / 2)
        panel = to_panel(y)
        pred = SARIMA(order=(0, 0, 0), seasonal_order=(0, 1, 0), m=4).fit(panel).predict(H)
        expected = SeasonalNaive(season_length=4).fit(panel).predict(H)
        assert np.allclose(pred["yhat"].to_numpy(), expected["yhat"].to_numpy())

    def test_include_mean_ignored_when_seasonally_differenced(self):
        y = _sim_sarima(120, sphi=(0.5,), m=4, D=1, seed=13)
        model = SARIMA(
            order=(1, 0, 0), seasonal_order=(0, 1, 0), m=4, include_mean=True
        ).fit(to_panel(y))
        assert model._series_state["series-0"]["c"] == 0.0

    def test_fitted_values_warmup(self):
        y = _sim_sarima(150, phi=(0.4,), sphi=(0.3,), m=4, d=1, seed=6)
        model = SARIMA(order=(1, 1, 1), seasonal_order=(1, 0, 1), m=4).fit(to_panel(y))
        fitted = model.fitted_values()["fitted"].to_numpy()
        warm = (1 + 0 * 4) + max(1 + 1 * 4, 1 + 1 * 4)  # d + D*m + max(p+P*m, q+Q*m)
        assert np.isnan(fitted[:warm]).all()
        assert np.isfinite(fitted[warm:]).all()

    def test_recovers_seasonal_ar_coefficient(self):
        y = _sim_sarima(700, sphi=(0.7,), m=12, seed=2)
        model = SARIMA(order=(0, 0, 0), seasonal_order=(1, 0, 0), m=12).fit(to_panel(y))
        state = model._series_state["series-0"]
        assert abs(state["seasonal_phi"][0] - 0.7) <= 0.1

    def test_recovers_both_ar_coefficients(self):
        y = _sim_sarima(900, phi=(0.5,), sphi=(0.6,), m=12, seed=5)
        model = SARIMA(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12).fit(to_panel(y))
        state = model._series_state["series-0"]
        assert abs(state["phi_raw"][0] - 0.5) <= 0.1
        assert abs(state["seasonal_phi"][0] - 0.6) <= 0.1
        # the stored recursion coefficients are the expanded polynomial
        assert len(state["phi"]) == 1 + 12

    def test_recovers_seasonal_ma_coefficient(self):
        # guards the '+' sign convention on the seasonal MA factor
        y = _sim_sarima(900, stheta=(0.6,), m=12, seed=3)
        model = SARIMA(order=(0, 0, 0), seasonal_order=(0, 0, 1), m=12).fit(to_panel(y))
        state = model._series_state["series-0"]
        assert abs(state["seasonal_theta"][0] - 0.6) <= 0.1

    def test_recovers_multiplicative_ma_pair(self):
        y = _sim_sarima(900, theta=(0.5,), stheta=(0.4,), m=4, seed=8)
        model = SARIMA(order=(0, 0, 1), seasonal_order=(0, 0, 1), m=4).fit(to_panel(y))
        state = model._series_state["series-0"]
        assert abs(state["theta_raw"][0] - 0.5) <= 0.12
        assert abs(state["seasonal_theta"][0] - 0.4) <= 0.12
        # expanded theta carries the cross term theta * Theta at lag m + 1
        assert np.allclose(
            state["theta"][-1], state["theta_raw"][0] * state["seasonal_theta"][0]
        )

    @pytest.mark.parametrize(
        "order,seasonal_order", [((1, 0, 0), (1, 0, 0)), ((1, 1, 0), (1, 1, 0))]
    )
    def test_forecast_mean_and_sigma_match_monte_carlo(self, order, seasonal_order):
        """Independent check of the integration and psi-weight sigma.

        Future paths of the DIFFERENCED series are simulated from the fitted
        recursion here (not via ``_forecast_css``) and pushed back through the
        integration stages; the analytic mean/sigma must match the empirical
        ones to Monte-Carlo accuracy.
        """
        m, h, npaths = 4, 8, 8000
        rng = np.random.default_rng(101)
        t = np.arange(200.0)
        y = 40.0 + 0.15 * t + 6.0 * np.sin(2 * np.pi * t / m) + rng.standard_normal(200)
        model = SARIMA(order=order, seasonal_order=seasonal_order, m=m).fit(to_panel(y))
        state = model._series_state["series-0"]
        c, phi, theta = state["c"], state["phi"], state["theta"]
        w, e, sigma = state["w"], state["e"], state["_sigma"]
        n, p, q = len(w), len(phi), len(theta)

        draws = sigma * np.random.default_rng(202).standard_normal((npaths, h))
        wext = np.zeros((npaths, n + h))
        wext[:, :n] = w
        eext = np.zeros((npaths, n + h))
        eext[:, :n] = e
        eext[:, n:] = draws
        for k in range(h):
            step = n + k
            acc = np.full(npaths, c) + eext[:, step]
            for i in range(1, p + 1):
                acc += phi[i - 1] * wext[:, step - i]
            for j in range(1, q + 1):
                acc += theta[j - 1] * eext[:, step - j]
            wext[:, step] = acc
        paths = np.array(
            [_seasonal_integrate(wext[b, n:], state["stages"], m) for b in range(npaths)]
        )

        analytic_mean = model.predict(h)["yhat"].to_numpy()
        analytic_sigma = model._predict_sigma(state, h)
        mc_sigma = paths.std(axis=0)
        # MC standard error of the mean is mc_sigma / sqrt(npaths)
        z = (analytic_mean - paths.mean(axis=0)) / (mc_sigma / np.sqrt(npaths))
        assert np.max(np.abs(z)) < 4.0
        assert np.allclose(analytic_sigma, mc_sigma, rtol=0.06)

    def test_beats_seasonal_naive_on_simulated_sarima(self):
        y = _sim_sarima(320, phi=(0.5,), sphi=(0.8,), m=12, sigma=1.0, seed=7)
        panel = to_panel(y)
        h = 24
        train = panel.iloc[:-h].copy()
        test = panel.iloc[-h:].copy()
        sar_mae = _mae(SARIMA(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12), train, test, h)
        snaive_mae = _mae(SeasonalNaive(season_length=12), train, test, h)
        assert sar_mae < snaive_mae

    def test_beats_seasonal_naive_on_seasonal_panel(self):
        panel = _seasonal_panel(length=132, m=12, seed=17, n_series=2)
        h = 12
        train = panel.groupby("unique_id").head(120)
        test = panel.groupby("unique_id").tail(h)
        sar_mae = _mae(SARIMA(order=(1, 0, 0), seasonal_order=(1, 1, 0), m=12), train, test, h)
        snaive_mae = _mae(SeasonalNaive(season_length=12), train, test, h)
        assert sar_mae < snaive_mae

    def test_seasonal_random_walk_sigma_steps_once_per_cycle(self):
        rng = np.random.default_rng(21)
        y = _seasonal_cumsum(rng.standard_normal(120), 4)
        model = SARIMA(order=(0, 0, 0), seasonal_order=(0, 1, 0), m=4).fit(to_panel(y))
        pred = model.predict(12, level=[80])
        width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
        assert np.allclose(width / width[0], np.sqrt(np.arange(12) // 4 + 1), rtol=1e-8)

    def test_regular_random_walk_sigma_grows_like_sqrt_k(self):
        rng = np.random.default_rng(22)
        y = np.cumsum(rng.standard_normal(150))
        model = SARIMA(order=(0, 1, 0), seasonal_order=(0, 0, 0), m=12).fit(to_panel(y))
        pred = model.predict(10, level=[80])
        width = (pred["hi-80"] - pred["lo-80"]).to_numpy()
        assert np.allclose(width / width[0], np.sqrt(np.arange(1, 11)), rtol=1e-8)

    def test_panel_contract_with_levels(self):
        df = _seasonal_panel(length=120, m=12, seed=23, n_series=3)
        model = SARIMA(order=(1, 0, 0), seasonal_order=(1, 1, 0), m=12).fit(df)
        pred = model.predict(H, level=[80])
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["yhat"]).all()
        assert (pred["yhat"] <= pred["hi-80"]).all()
        for _, g in pred.groupby("unique_id"):
            width = (g["hi-80"] - g["lo-80"]).to_numpy()
            assert np.all(np.diff(width) >= -1e-9)

    def test_state_records_orders(self):
        y = _seasonal_panel(length=120, m=4, seed=25)
        state = (
            SARIMA(order=(1, 0, 1), seasonal_order=(1, 1, 0), m=4)
            .fit(y)
            ._series_state["series-0"]
        )
        assert state["order_"] == (1, 0, 1)
        assert state["seasonal_order_"] == (1, 1, 0)


class TestOrderSelectionHelpers:
    def test_aicc_matches_the_pinned_formula(self):
        n, sse, k = 100, 250.0, 4
        expected = n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        assert _aicc(n, sse, k) == pytest.approx(expected)

    def test_aicc_floors_a_zero_residual_variance(self):
        assert np.isfinite(_aicc(50, 0.0, 3))
        assert _aicc(50, 0.0, 3) == pytest.approx(50 * np.log(1e-12) + 6 + 2 * 3 * 4 / 46)

    def test_aicc_penalizes_extra_parameters_at_equal_fit(self):
        assert _aicc(100, 250.0, 5) > _aicc(100, 250.0, 4)

    def test_seasonal_caps_disabled_for_m_one(self):
        assert _seasonal_caps(1, 500, 2, 2) == (0, 0)

    def test_seasonal_caps_disabled_for_short_series(self):
        # fewer than two full seasons (plus slack) left after differencing
        assert _seasonal_caps(12, 12 * 2 + 4, 1, 1) == (0, 0)
        assert _seasonal_caps(12, 12 * 2 + 5, 1, 1) == (1, 1)

    def test_seasonal_caps_pass_through_when_identifiable(self):
        assert _seasonal_caps(4, 200, 2, 3) == (2, 3)


class TestAutoSARIMA:
    def test_registered_as_statistical(self):
        assert isinstance(get_model("auto_sarima"), AutoSARIMA)
        assert "auto_sarima" in set(list_models(family="statistical")["name"])

    def test_params_stored_and_clone(self):
        model = AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=0, max_d=1, max_D=1)
        assert model.get_params() == {
            "m": 4,
            "max_p": 1,
            "max_q": 1,
            "max_P": 1,
            "max_Q": 0,
            "max_d": 1,
            "max_D": 1,
        }
        assert model.clone().get_params() == model.get_params()

    @pytest.mark.parametrize(
        "kwargs", [{"m": 0}, {"m": "a"}, {"max_p": -1}, {"max_D": -1}, {"max_Q": -2}]
    )
    def test_bad_args_raise_forecast_os_error(self, kwargs):
        with pytest.raises(ForecastOSError):
            AutoSARIMA(**kwargs)

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            AutoSARIMA().predict(3)

    def test_min_train_size_formula(self):
        model = AutoSARIMA(m=6, max_p=2, max_q=1, max_P=1, max_Q=0, max_d=1, max_D=1)
        assert model.min_train_size == max(2 + 1 * 6, 1 + 0 * 6) + 1 + 1 * 6 + 5

    def test_series_shorter_than_min_train_size_errors_cleanly(self):
        model = AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1, max_d=1, max_D=1)
        n = model.min_train_size - 1
        with pytest.raises(ForecastOSError, match="requires at least"):
            model.fit(to_panel(np.arange(float(n))))

    def test_selects_seasonal_difference_on_seasonal_series(self):
        df = _seasonal_panel(length=120, m=4, seed=31)
        model = AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1).fit(df)
        state = model._series_state["series-0"]
        assert state["seasonal_order_"][1] == 1

    def test_no_seasonal_difference_on_trend_only_series(self):
        rng = np.random.default_rng(33)
        y = 10.0 + 0.5 * np.arange(120.0) + rng.standard_normal(120)
        model = AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1).fit(to_panel(y))
        assert model._series_state["series-0"]["seasonal_order_"][1] == 0

    def test_stores_selected_orders(self):
        df = _seasonal_panel(length=120, m=4, seed=35)
        state = (
            AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1)
            .fit(df)
            ._series_state["series-0"]
        )
        assert len(state["order_"]) == 3
        assert len(state["seasonal_order_"]) == 3
        assert all(v >= 0 for v in state["order_"] + state["seasonal_order_"])

    def test_d_is_selected_on_the_seasonally_differenced_series(self):
        # trend + strong seasonality: the seasonal difference already removes
        # the trend, so no regular difference should be taken afterwards, even
        # though the raw series on its own would be differenced.
        rng = np.random.default_rng(43)
        t = np.arange(120.0)
        y = 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 4) + 0.5 * rng.standard_normal(120)
        model = AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1)
        state = model.fit(to_panel(y))._series_state["series-0"]
        assert state["seasonal_order_"][1] == 1
        assert state["order_"][1] == 0
        assert model._select_d(y) == 1, "the raw series alone would be differenced"

    def test_grid_search_picks_the_aicc_argmin(self):
        """Independently recompute the whole grid, including the parameter count."""
        df = _seasonal_panel(length=120, m=4, seed=45)
        y = df["y"].to_numpy()
        m, d, D = 4, 0, 1
        model = AutoSARIMA(m=m, max_p=1, max_q=1, max_P=1, max_Q=1)
        got = model._search_orders(y, d, D)

        n_after = len(y) - d - D * m
        best, best_aicc = None, np.inf
        for p in range(2):
            for q in range(2):
                for P in range(2):
                    for Q in range(2):
                        if n_after - max(p + P * m, q + Q * m) < 5:
                            continue
                        st = _fit_sarima_css(y, (p, d, q), (P, D, Q), m, True)
                        if not st["converged"]:
                            continue
                        n, k = st["n_eff"], p + q + P + Q + 1
                        if n - k - 1 <= 0:
                            continue
                        aicc = _aicc(n, st["sse"], k)
                        if aicc < best_aicc:
                            best, best_aicc = ((p, d, q), (P, D, Q)), aicc
        assert best is not None
        assert (got["order_"], got["seasonal_order_"]) == best
        # the winner must actually use the seasonal machinery on this series
        assert best[1][0] + best[1][2] > 0

    def test_m_one_disables_the_seasonal_grid(self):
        rng = np.random.default_rng(37)
        y = 5.0 + np.cumsum(rng.standard_normal(90))
        state = (
            AutoSARIMA(m=1, max_p=1, max_q=1, max_P=1, max_Q=1)
            .fit(to_panel(y))
            ._series_state["series-0"]
        )
        assert state["seasonal_order_"] == (0, 0, 0)

    def test_panel_contract_with_levels(self):
        df = _seasonal_panel(length=100, m=4, seed=39, n_series=2)
        pred = AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1).fit(df).predict(H, level=[80])
        assert (pred.groupby("unique_id").size() == H).all()
        for col in ("yhat", "lo-80", "hi-80"):
            assert np.isfinite(pred[col]).all()
        assert (pred["lo-80"] <= pred["hi-80"]).all()

    def test_beats_seasonal_naive_on_seasonal_panel(self):
        panel = _seasonal_panel(length=112, m=4, seed=41, n_series=2)
        h = 8
        train = panel.groupby("unique_id").head(104)
        test = panel.groupby("unique_id").tail(h)
        auto_mae = _mae(AutoSARIMA(m=4, max_p=1, max_q=1, max_P=1, max_Q=1), train, test, h)
        snaive_mae = _mae(SeasonalNaive(season_length=4), train, test, h)
        assert auto_mae < snaive_mae


class TestAdversarialRegressions:
    """Regressions for defects found by the v0.8.0 adversarial verification wave."""

    def test_order_selection_is_invariant_to_the_units_of_y(self):
        """AICc must score every candidate on the same observations.

        Each candidate's own warm-up ``max(p + P*m, q + Q*m)`` differs by up to
        ~2*m across the grid, and ``n*log(sse/n)`` shifts by ``2*n*log(lambda)``
        under a rescaling of y — so a per-candidate ``n`` made AICc differences
        depend on the units of y. Selling in dollars vs thousands of dollars
        picked a different model (observed: (0,0,0)(0,1,0) at 1e-3 through
        (2,0,1)(1,1,1) at 1e4 on one fixed series).
        """
        t = np.arange(120)
        rng = np.random.default_rng(11)
        y = 20 + 3 * np.sin(2 * np.pi * t / 12) + 0.05 * t + rng.normal(0, 1.0, 120)

        chosen = set()
        for lam in (1e-3, 1e-1, 1.0, 10.0, 1e4):
            state = AutoSARIMA(m=12).fit(to_panel(y * lam))._series_state["series-0"]
            chosen.add((state["order_"], state["seasonal_order_"]))
        assert len(chosen) == 1, f"order selection changed with the units of y: {chosen}"

    def test_seasonal_strength_survives_a_single_outlier(self):
        """One spike must not veto a seasonal difference on a strongly seasonal series.

        Means and variances have a 0% breakdown point: a lone spike is only
        1/n_k absorbed by its cycle mean, so it lands in the remainder and drives
        Fs toward 0. Observed before the fix: Fs 0.963 -> 0.600 for a single
        +40 spike on a 6-amplitude seasonal, flipping D from 1 to 0 and tripling
        holdout error.
        """
        t = np.arange(120)
        rng = np.random.default_rng(0)
        clean = 100 + 6 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 120)
        model = AutoSARIMA(m=12)

        assert _seasonal_strength(clean, 12) > 0.9
        assert model._select_D(clean) == 1
        for spike in (10.0, 40.0, 160.0):
            dirty = clean.copy()
            dirty[60] += spike
            assert _seasonal_strength(dirty, 12) > 0.9, f"spike {spike} destroyed Fs"
            assert model._select_D(dirty) == 1, f"spike {spike} flipped D"

    def test_seasonal_strength_still_says_no_on_non_seasonal_series(self):
        """The robust statistic must not become trigger-happy: D=0 on non-seasonal data."""
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, 120)
        trend = np.linspace(0, 20, 120) + rng.normal(0, 1, 120)
        model = AutoSARIMA(m=12)
        assert model._select_D(noise) == 0
        assert model._select_D(trend) == 0

    def test_css_fit_is_equivariant_in_the_units_of_y(self):
        """L-BFGS-B's gtol is absolute but the CSS gradient scales as the units squared.

        A series measured in millionths therefore stopped at the all-zero
        starting point and reported ``converged=True``. Observed before the fix
        at lambda=1e-4: phi=theta=Phi=0.0 exactly, with sse/lambda**2 of 492.4
        against 299.2 at every other scale. Magnitudes like 1e-4 are ordinary —
        defect rates, ppm concentrations, small probabilities.
        """
        rng = np.random.default_rng(0)
        innov = rng.normal(size=300)
        y = np.zeros(300)
        for i in range(300):
            y[i] = 0.6 * (y[i - 1] if i else 0.0) + innov[i]

        ref = _fit_sarima_css(y, (1, 0, 1), (1, 0, 0), 12, True)
        for lam in (1e-6, 1e-4, 1e-2, 1e2, 1e6):
            got = _fit_sarima_css(y * lam, (1, 0, 1), (1, 0, 0), 12, True)
            for key in ("phi_raw", "theta_raw", "seasonal_phi"):
                assert np.allclose(got[key], ref[key], atol=1e-4), (
                    f"{key} moved at lambda={lam}: {got[key]} vs {ref[key]}"
                )
            # The intercept and the residual sum of squares carry the units.
            assert np.isclose(got["c"], ref["c"] * lam, rtol=1e-4, atol=1e-12 * lam)
            assert np.isclose(got["sse"] / lam**2, ref["sse"], rtol=1e-4)
