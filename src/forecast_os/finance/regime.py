"""Markov regime switching: a Gaussian HMM over returns fitted by EM.

Each state is a Gaussian (mean, std) and states evolve by a first-order
Markov chain. The E-step is a scaled forward-backward pass (numerically
stable log-likelihood via the scaling constants); the M-step is the standard
Baum-Welch update. After fitting, states are reordered by mean ascending so
state 0 is always the "bear" (lowest-mean) regime.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import NotFittedError

__all__ = ["MarkovRegimeSwitching"]

_TINY = 1e-300


class MarkovRegimeSwitching:
    """K-state Gaussian hidden Markov model for return regimes."""

    def __init__(self, n_states: int = 2, max_iter: int = 200, tol: float = 1e-6, seed: int = 0):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    # -- EM internals --------------------------------------------------------

    @staticmethod
    def _e_step(r, pi, trans, means, stds):
        """Scaled forward-backward: smoothed probs, transition counts, loglik."""
        n, k = r.size, means.size
        b = np.exp(-0.5 * ((r[:, None] - means) / stds) ** 2) / (stds * np.sqrt(2 * np.pi))
        b = np.maximum(b, _TINY)

        alpha = np.empty((n, k))
        c = np.empty(n)
        a = pi * b[0]
        c[0] = max(a.sum(), _TINY)
        alpha[0] = a / c[0]
        for t in range(1, n):
            a = (alpha[t - 1] @ trans) * b[t]
            c[t] = max(a.sum(), _TINY)
            alpha[t] = a / c[t]

        beta = np.empty((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (trans @ (b[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _TINY)
        # xi_sum[i, j] = sum_t alpha[t, i] * P[i, j] * b[t+1, j] * beta[t+1, j] / c[t+1]
        xi_sum = trans * (alpha[:-1].T @ (b[1:] * beta[1:] / c[1:, None]))
        return gamma, xi_sum, float(np.sum(np.log(c)))

    def fit(self, returns) -> MarkovRegimeSwitching:
        """Fit the HMM on a 1-D returns array via EM."""
        r = np.asarray(returns, dtype=float).ravel()
        if r.size == 0:
            raise ValueError("empty returns")
        k = self.n_states
        if r.size < 5 * k:
            raise ValueError(f"need at least {5 * k} observations for {k} states, got {r.size}")
        if not np.isfinite(r).all():
            raise ValueError("returns contain NaN or infinite values")

        rng = np.random.default_rng(self.seed)
        overall_std = max(float(np.std(r)), 1e-8)
        means = np.quantile(r, (np.arange(k) + 1) / (k + 1))
        means = means + 1e-3 * overall_std * rng.standard_normal(k)
        stds = overall_std * np.linspace(0.5, 1.5, k)  # encourage variance separation
        trans = np.full((k, k), 0.05 / max(k - 1, 1))
        np.fill_diagonal(trans, 0.95)
        pi = np.full(k, 1.0 / k)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            gamma, xi_sum, ll = self._e_step(r, pi, trans, means, stds)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
            pi = gamma[0]
            trans = xi_sum / np.maximum(gamma[:-1].sum(axis=0), _TINY)[:, None]
            trans /= trans.sum(axis=1, keepdims=True)
            w = np.maximum(gamma.sum(axis=0), _TINY)
            means = (gamma * r[:, None]).sum(axis=0) / w
            stds = np.sqrt((gamma * (r[:, None] - means) ** 2).sum(axis=0) / w)
            stds = np.maximum(stds, 1e-8)
        gamma, _, _ = self._e_step(r, pi, trans, means, stds)

        order = np.argsort(means)  # state 0 = lowest mean ("bear")
        self.means_ = means[order]
        self.stds_ = stds[order]
        self.transition_ = trans[np.ix_(order, order)]
        self.smoothed_probs_ = gamma[:, order]
        self.regimes_ = np.argmax(self.smoothed_probs_, axis=1)
        self._is_fitted = True
        return self

    # -- inference -----------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("MarkovRegimeSwitching is not fitted; call fit() first")

    def predict_proba(self, h: int) -> np.ndarray:
        """State probabilities ``h`` steps ahead: ``p_T @ P^h``, shape (K,)."""
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        p = self.smoothed_probs_[-1] @ np.linalg.matrix_power(self.transition_, int(h))
        return p / p.sum()

    def expected_return(self, h: int) -> float:
        """Probability-weighted mean return ``h`` steps ahead."""
        return float(self.predict_proba(h) @ self.means_)
