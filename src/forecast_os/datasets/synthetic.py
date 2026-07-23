"""Seeded synthetic data generators used in tests, examples, and benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.types import ID_COL, TARGET_COL, TIME_COL

__all__ = ["generate_series", "generate_returns"]


def generate_series(
    n_series: int = 3,
    length: int = 200,
    freq: str = "D",
    start: str = "2024-01-01",
    level: float = 100.0,
    trend: float = 0.5,
    seasonality: int | None = 7,
    season_amp: float = 10.0,
    noise: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Panel of series with linear trend + sinusoidal seasonality + Gaussian noise.

    Each series gets slightly perturbed level/trend/amplitude so the panel is
    heterogeneous but reproducible for a given ``seed``.
    """
    if n_series < 1 or length < 1:
        raise ValueError("n_series and length must be positive")
    rng = np.random.default_rng(seed)
    ds = pd.date_range(start=start, periods=length, freq=freq)
    t = np.arange(length, dtype=float)
    frames = []
    for i in range(n_series):
        lvl = level * (1 + 0.1 * rng.standard_normal())
        tr = trend * (1 + 0.1 * rng.standard_normal())
        y = lvl + tr * t
        if seasonality and seasonality > 1:
            amp = season_amp * (1 + 0.1 * rng.standard_normal())
            phase = rng.uniform(0, 2 * np.pi)
            y = y + amp * np.sin(2 * np.pi * t / seasonality + phase)
        y = y + noise * rng.standard_normal(length)
        frames.append(pd.DataFrame({ID_COL: f"series-{i}", TIME_COL: ds, TARGET_COL: y}))
    return pd.concat(frames, ignore_index=True)


def generate_returns(
    n_series: int = 1,
    length: int = 500,
    mu: float = 0.0005,
    sigma: float = 0.01,
    garch: tuple[float, float, float] | None = None,
    start: str = "2024-01-01",
    freq: str = "B",
    seed: int = 0,
) -> pd.DataFrame:
    """Panel of simulated financial returns.

    With ``garch=(omega, alpha, beta)`` the returns follow a GARCH(1,1)
    volatility process (``alpha + beta < 1`` required); otherwise they are
    i.i.d. Gaussian with mean ``mu`` and std ``sigma``.
    """
    if n_series < 1 or length < 1:
        raise ValueError("n_series and length must be positive")
    rng = np.random.default_rng(seed)
    ds = pd.date_range(start=start, periods=length, freq=freq)
    frames = []
    for i in range(n_series):
        if garch is None:
            y = mu + sigma * rng.standard_normal(length)
        else:
            omega, alpha, beta = garch
            if min(omega, alpha, beta) < 0 or alpha + beta >= 1:
                raise ValueError(
                    f"invalid GARCH params {garch}; need omega, alpha, beta >= 0 "
                    f"and alpha + beta < 1"
                )
            var = omega / (1 - alpha - beta)
            y = np.empty(length)
            for t in range(length):
                eps = np.sqrt(var) * rng.standard_normal()
                y[t] = mu + eps
                var = omega + alpha * eps**2 + beta * var
        frames.append(pd.DataFrame({ID_COL: f"asset-{i}", TIME_COL: ds, TARGET_COL: y}))
    return pd.concat(frames, ignore_index=True)
