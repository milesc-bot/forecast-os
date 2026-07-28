"""Monte Carlo price simulation under geometric Brownian motion.

Each step multiplies the price by ``exp((mu - sigma^2/2) + sigma*Z)`` with
``Z ~ N(0, 1)``. The ``-sigma^2/2`` Ito correction means ``mu`` is the
per-period **arithmetic** (simple-return) drift, not the log-return drift:
``E[S_t / S_{t-1}] = exp(mu)``, while the *mean log* step is ``mu - sigma^2/2``.
``sigma`` is the per-period log-return volatility, which for the small returns
this is used on is interchangeable with the simple-return volatility.

So calibrate from **simple** returns, ``diff(prices) / prices[:-1]``, matching
the convention :mod:`~forecast_os.finance.metrics` uses across the package.
Feeding log returns to :meth:`MonteCarloSimulator.from_returns` under-drifts
every path by ``sigma^2/2`` per step, a shortfall of ``1 - exp(-sigma^2 * h / 2)``
after ``h`` steps (about -1.25% on a one-year daily fan at 1% daily vol; it
takes ~2.9% daily vol to reach -10% over the same year); add ``sigma^2/2``
back to ``mu`` yourself if you only have log returns.

That match is first-order, not exact: :meth:`MonteCarloSimulator.from_returns`
sets ``mu = mean(r)``, and since ``E[S_t/S_{t-1}] = exp(mu)`` the simulated
mean gross step is ``exp(m)``, not the observed ``1 + m`` — equivalently the
mean *net* step is ``exp(m) - 1`` where the data showed ``m``. That overstates
the drift by ``(exp(m) - 1) / m - 1``, roughly ``m / 2`` in relative terms:
+0.03% on daily returns (``m = 0.0006``), but +4.1% at ``m = 0.08`` and +16.6%
at ``m = 0.30``. The exact calibration is ``mu = log1p(m)``, so pass
``mu=float(np.log1p(np.mean(r)))`` yourself when calibrating from monthly,
annual or crypto-scale per-period returns.

All randomness flows through ``np.random.default_rng(seed)``; a fresh
generator per call makes repeated simulations reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["MonteCarloSimulator"]


class MonteCarloSimulator:
    """Geometric Brownian motion simulator for price paths.

    ``mu`` is the per-period arithmetic (simple-return) drift — see the module
    docstring — and ``sigma`` the per-period volatility.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.01, seed: int = 0):
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.seed = seed

    @classmethod
    def from_returns(cls, returns, seed: int = 0) -> MonteCarloSimulator:
        """Calibrate per-period mu/sigma from an observed **simple** returns array.

        ``returns`` must be simple per-period returns
        (``np.diff(prices) / prices[:-1]``), the convention used throughout
        :mod:`forecast_os.finance`. Passing log returns biases every simulated
        path down by ``sigma^2/2`` per step, because :meth:`simulate` applies
        the Ito correction to ``mu``.

        ``mu`` is set to ``mean(returns) = m`` as-is, which reproduces the
        observed mean gross return ``1 + m`` only to first order in ``m``: the
        simulated mean gross step is ``exp(m)``, so the drift is overstated by
        ``(exp(m) - 1) / m - 1`` — 0.03% on daily returns, but +4.1% at
        ``m = 0.08``. For monthly, annual or crypto-scale returns construct the
        simulator directly with ``mu=float(np.log1p(np.mean(returns)))``, which
        is exact; see the module docstring.
        """
        r = np.asarray(returns, dtype=float).ravel()
        if r.size == 0:
            raise ValueError("empty returns")
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        return cls(mu=float(np.mean(r)), sigma=sigma, seed=seed)

    def simulate(self, s0: float, h: int, n_paths: int = 1000) -> np.ndarray:
        """Simulate ``n_paths`` price paths of length ``h`` starting from ``s0``.

        Returns an array of shape ``(n_paths, h)``; column ``k`` is the price
        ``k + 1`` steps ahead. ``E[S_{k+1} / S_k] = exp(mu)``.
        """
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError(f"s0 must be a positive number, got {s0!r}")
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
            raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((int(n_paths), int(h)))
        steps = (self.mu - 0.5 * self.sigma**2) + self.sigma * z
        return float(s0) * np.exp(np.cumsum(steps, axis=1))

    def summary(
        self,
        s0: float,
        h: int,
        n_paths: int = 1000,
        levels: tuple[int, ...] = (5, 25, 50, 75, 95),
    ) -> pd.DataFrame:
        """Quantiles across simulated paths per step: columns ``step, q{level:02d}``."""
        paths = self.simulate(s0, h, n_paths)
        data: dict[str, np.ndarray] = {"step": np.arange(1, int(h) + 1)}
        for lvl in levels:
            if not 0 < lvl < 100:
                raise ValueError(f"levels must be in (0, 100), got {lvl}")
            data[f"q{int(lvl):02d}"] = np.quantile(paths, lvl / 100.0, axis=0)
        return pd.DataFrame(data)
