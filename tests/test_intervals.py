"""Interval calibration pins: exact Gaussian widths and empirical coverage.

Pins the residual-sigma convention used by ``PerSeriesForecaster`` for
Gaussian prediction intervals: sigma is the UNCENTERED residual rms with an
n-1 denominator, ``sigma = sqrt(sum(resid**2) / (n_resid - 1))``. The pin is
coordinated by formula (not by importing private helpers) so it fails loudly
if the convention drifts.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from forecast_os.core.types import to_panel
from forecast_os.models.baselines import Naive
from forecast_os.models.ets import SES


def test_naive_interval_width_is_z_sigma_sqrt_h_exact():
    """hi-80 - yhat == norm.ppf(0.9) * sigma * sqrt([1, 2, 3]) to 1e-9.

    For Naive the residuals are diff(y), so under the uncentered convention
    sigma = sqrt(sum(diff(y)**2) / (len(diff(y)) - 1)). The series is chosen
    with mean(diff(y)) != 0 so a centered (np.std-style) sigma would fail.
    """
    y = np.array([1.0, 3.0, 2.0, 6.0, 4.0, 5.0])
    d = np.diff(y)
    assert abs(d.mean()) > 0.1  # guard: centered and uncentered sigma differ
    sigma = np.sqrt(np.sum(d**2) / (d.size - 1))

    pred = Naive().fit(to_panel(y)).predict(3, level=[80])
    z = stats.norm.ppf(0.9)
    expected_width = z * sigma * np.sqrt(np.array([1.0, 2.0, 3.0]))

    hi_width = pred["hi-80"].to_numpy() - pred["yhat"].to_numpy()
    lo_width = pred["yhat"].to_numpy() - pred["lo-80"].to_numpy()
    np.testing.assert_allclose(hi_width, expected_width, rtol=0, atol=1e-9)
    np.testing.assert_allclose(lo_width, expected_width, rtol=0, atol=1e-9)


def test_ses_one_step_80_interval_empirical_coverage():
    """One-step-ahead 80% SES intervals cover the next iid N(10, 1) draw ~80%."""
    n_seeds = 30
    covered = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y = 10.0 + rng.standard_normal(201)
        train, next_true = y[:200], y[200]
        pred = SES().fit(to_panel(train)).predict(1, level=[80])
        lo = float(pred["lo-80"].iloc[0])
        hi = float(pred["hi-80"].iloc[0])
        covered += int(lo <= next_true <= hi)
    coverage = covered / n_seeds
    assert 0.7 <= coverage <= 0.9, f"empirical 80% coverage {coverage:.3f} not in [0.7, 0.9]"
