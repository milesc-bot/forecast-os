"""Monte Carlo price simulation under geometric Brownian motion.

Each step multiplies the price by ``exp((mu - sigma^2/2) + sigma*Z)`` with
``Z ~ N(0, 1)``, so ``mu``/``sigma`` are per-period log-return drift and
volatility. All randomness flows through ``np.random.default_rng(seed)``; a
fresh generator per call makes repeated simulations reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["MonteCarloSimulator"]


class MonteCarloSimulator:
    """Geometric Brownian motion simulator for price paths."""

    def __init__(self, mu: float = 0.0, sigma: float = 0.01, seed: int = 0):
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.seed = seed

    @classmethod
    def from_returns(cls, returns, seed: int = 0) -> MonteCarloSimulator:
        """Calibrate per-period mu/sigma from an observed returns array."""
        r = np.asarray(returns, dtype=float).ravel()
        if r.size == 0:
            raise ValueError("empty returns")
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        return cls(mu=float(np.mean(r)), sigma=sigma, seed=seed)

    def simulate(self, s0: float, h: int, n_paths: int = 1000) -> np.ndarray:
        """Simulate ``n_paths`` price paths of length ``h`` starting from ``s0``.

        Returns an array of shape ``(n_paths, h)``; column ``k`` is the price
        ``k + 1`` steps ahead.
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
