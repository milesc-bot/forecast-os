"""Cohort retention: panel building and the shifted-beta-geometric model.

:func:`cohort_panel` turns an already-aggregated cohort table (cohort id,
integer period age, retained count or rate) into the ``(unique_id, ds, y)``
contract with ``ds`` as integer age and ``y`` as the retention fraction.

:class:`ShiftedBetaGeometric` is the Fader-Hardie sBG model ("How to
project customer retention", 2007): each customer churns geometrically
with a probability drawn from a Beta(alpha, beta), giving the churn
recursion ``p_1 = alpha / (alpha + beta)``,
``p_t = (beta + t - 2) / (alpha + beta + t - 1) * p_{t-1}`` and survival
``S(t) = S(t-1) - p_t``. Parameters are estimated per cohort by maximum
likelihood on the observed retention curve, with an empirical-Bayes-lite
fallback: cohorts too short for a stable MLE borrow POOLED parameters
fitted on the cohorts' scale-free (age-0 normalized) curves, combined
position-wise through their per-period retention ratios so a ragged cohort
triangle still pools into a monotone survival curve.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..core.base import PerSeriesForecaster
from ..core.exceptions import DataContractError, ForecastOSError
from ..core.registry import register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

__all__ = ["cohort_panel", "ShiftedBetaGeometric"]

_EPS = 1e-9


def cohort_panel(
    records: pd.DataFrame,
    cohort_col: str,
    period_col: str,
    retained_col: str,
    n_customers_col: str | None = None,
) -> pd.DataFrame:
    """Build a retention panel from an aggregated cohort table.

    Parameters
    ----------
    records : one row per (cohort, period age) with the retained count or rate.
    cohort_col : cohort identifier column (becomes ``unique_id``).
    period_col : integer period age column (becomes ``ds``).
    retained_col : retained count (with ``n_customers_col``) or retention
        fraction (without).
    n_customers_col : cohort size column; when given, ``y`` is
        ``retained / n_customers``.

    Returns
    -------
    A ``(unique_id, ds, y)`` panel with integer ``ds`` and ``y`` in [0, 1].
    Monotone-nonincreasing curves are NOT required (measurement noise
    happens), but out-of-range fractions raise :class:`DataContractError`.

    ``period_col`` is passed through as the cohort AGE, and no age handling
    is applied here: :class:`ShiftedBetaGeometric` additionally requires each
    cohort's ages to be consecutive integers starting at 0 or 1, so index the
    periods relative to each cohort's own acquisition period before calling.
    """
    if not isinstance(records, pd.DataFrame):
        raise DataContractError(
            f"expected a pandas DataFrame of cohort records, got {type(records).__name__}"
        )
    needed = [cohort_col, period_col, retained_col] + (
        [n_customers_col] if n_customers_col is not None else []
    )
    missing = [c for c in needed if c not in records.columns]
    if missing:
        raise DataContractError(f"missing column(s) {missing} in cohort records")

    y = pd.to_numeric(records[retained_col]).astype(float)
    if n_customers_col is not None:
        y = y / pd.to_numeric(records[n_customers_col]).astype(float)
    if ((y < -_EPS) | (y > 1.0 + _EPS)).any():
        raise DataContractError(
            f"column {retained_col!r} yields retention fractions outside [0, 1]; "
            f"pass n_customers_col for raw counts"
        )
    panel = pd.DataFrame(
        {
            ID_COL: records[cohort_col].astype(str).to_numpy(),
            TIME_COL: pd.to_numeric(records[period_col]).astype(int).to_numpy(),
            TARGET_COL: y.to_numpy(),
        }
    )
    return validate_panel(panel)


def _sbg_survival(alpha: float, beta: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Churn probabilities ``p[1..T]`` and survival ``S[0..T]`` for the sBG model."""
    p = np.zeros(horizon + 1)
    s = np.ones(horizon + 1)
    if horizon >= 1:
        p[1] = alpha / (alpha + beta)
        s[1] = 1.0 - p[1]
        for t in range(2, horizon + 1):
            p[t] = (beta + t - 2.0) / (alpha + beta + t - 1.0) * p[t - 1]
            s[t] = s[t - 1] - p[t]
    return p, s


def _survival_curve(y: np.ndarray) -> np.ndarray:
    """Rescale an age-anchored cohort curve to model survival ``S(1..T)``.

    ``y[0]`` is the age-0 observation — the cohort's own size at acquisition,
    which is 1.0 only when every acquired customer activates. It is the SCALE
    the survival curve is measured against, not an ``S(1)`` observation, so
    the remaining values are divided by it. Callers guarantee that ``y`` is
    anchored at age 0 (see :meth:`ShiftedBetaGeometric.fit`) but NOT that
    ``y[0]`` is positive or that the result lands in [0, 1]: a cohort that
    nobody activated into divides by zero, and one whose retention rises out
    of age 0 gives ratios above 1. Both callers screen for those cases
    themselves rather than pretending this function cannot produce them.
    """
    y = np.asarray(y, dtype=float)
    return y[1:] / y[0]


def _pooled_curve(curves: list[np.ndarray]) -> np.ndarray:
    """Pool ragged scale-free cohort curves into one survival curve.

    Every real cohort triangle is RAGGED — recent cohorts are short — so the
    pooling is done position-wise on per-period retention *ratios*
    (``c[t] / c[t-1]``), then cumulated, rather than on the survival levels
    themselves. Averaging the levels lets the composition of the panel change
    from one position to the next: where the short cohorts run out, the mean
    jumps to whatever the surviving (typically more mature, higher-retention)
    cohorts sit at, and the "curve" goes UP. That is not a survival curve —
    its churn weights and ``S(T)`` no longer sum to 1, so the clipping in
    :func:`_sbg_mle`, which exists to absorb measurement noise, silently
    swallows a systematic composition artifact instead.

    Cumulating mean ratios is monotone nonincreasing by construction (each
    mean ratio is clipped into [0, 1], the same noisy-up-tick convention
    :func:`_sbg_mle` uses), uses every cohort at every position it actually
    observes, and returns the identical curve when all cohorts share one
    curve. Position 0 is measured against the age-0 anchor of 1.0 that
    :func:`_survival_curve` already divided out; a position no usable cohort
    reaches carries forward flat.
    """
    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        mat[i, : len(c)] = c
    prev = np.concatenate([np.ones((len(curves), 1)), mat[:, :-1]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_prev = np.where(prev > _EPS, prev, 1.0)
        ratio = np.where(prev > _EPS, mat / safe_prev, np.nan)
    ratio = np.clip(ratio, 0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN position
        mean_ratio = np.nanmean(ratio, axis=0)
    return np.cumprod(np.where(np.isnan(mean_ratio), 1.0, mean_ratio))


def _sbg_mle(s_obs: np.ndarray) -> tuple[float, float, bool]:
    """MLE of (alpha, beta) on one retention curve; third element is success.

    Maximizes the standard sBG likelihood with fractional weights: the churn
    fraction in period ``t`` weights ``ln p_t`` and the final survivors
    weight ``ln S(T)``. Noisy up-ticks (negative churn increments) are
    clipped to zero weight. Optimized over log-parameters (Nelder-Mead) so
    alpha, beta stay positive; runaway or non-finite solutions are reported
    as failures so callers can fall back to pooled parameters.
    """
    sobs_full = np.concatenate([[1.0], np.asarray(s_obs, dtype=float)])
    weights = np.clip(-np.diff(sobs_full), 0.0, None)
    horizon = len(s_obs)
    s_last = float(sobs_full[-1])

    def nll(log_params: np.ndarray) -> float:
        if np.any(np.abs(log_params) > 12.0):
            return 1e12
        a, b = np.exp(log_params)
        p, s = _sbg_survival(a, b, horizon)
        ll = float(np.sum(weights * np.log(np.maximum(p[1:], 1e-300))))
        ll += s_last * float(np.log(max(s[horizon], 1e-300)))
        return -ll

    try:
        res = minimize(
            nll,
            x0=np.zeros(2),
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-11, "maxiter": 2000},
        )
    except (ValueError, FloatingPointError):
        return 1.0, 1.0, False
    if (
        not np.all(np.isfinite(res.x))
        or not np.isfinite(res.fun)
        or np.any(np.abs(res.x) > 10.0)
    ):
        return 1.0, 1.0, False
    a, b = np.exp(res.x)
    return float(a), float(b), True


@register("retention_sbg", family="gtm")
class ShiftedBetaGeometric(PerSeriesForecaster):
    """Fader-Hardie shifted-beta-geometric cohort retention model.

    Fits (alpha, beta) per cohort by MLE on the observed retention curve and
    forecasts future retention by projecting the model survival curve
    forward from the last observed value:
    ``yhat_k = y_last * S(T + k) / S(T)`` — monotone nonincreasing and in
    [0, 1] by construction.

    Input convention: each series is a retention curve at consecutive
    integer ages, and ``ds`` — never the value at ``y[0]`` — is what says
    which age an observation belongs to. A curve starts either at age 0,
    whose value anchors it (1.0 when every acquired customer activates, less
    when some churn before their first renewal; the model treats it as a
    scale and fits ``y[1:] / y[0]`` as ``S(1..T)``), or at age 1, which
    implies an age-0 anchor of 1.0. Any other first age raises at ``fit``:
    the curve's position on the age axis would be a guess. Values must be
    fractions in [0, 1] — anything else raises at ``fit`` (this model is
    retention-only by design and refuses generic panels).

    Empirical-Bayes-lite pooling: ``fit`` first estimates POOLED parameters on
    the cohorts' scale-free curves, pooled position-wise through their
    per-period retention ratios (see :func:`_pooled_curve` — a plain mean of
    the survival LEVELS is non-monotone on the ragged triangle every real
    panel is), then fits each cohort; a cohort with fewer than
    ``pooled_threshold`` observations,
    or whose MLE fails, or with no age-0 retention to measure survival
    against, uses the pooled parameters instead (so even a 2-point or dead
    cohort forecasts sensibly). Both ``pooled_threshold`` and
    ``min_train_size`` count the rows the CALLER supplied — an age-0 anchor
    this class supplies for an age-1 curve is bookkeeping, so identical data
    is shrunk identically whichever age it is written from. At least one
    series must still have ``min_train_size`` observations for the pooled fit
    to mean anything.
    """

    min_train_size = 3

    def __init__(self, pooled_threshold: int = 5):
        if pooled_threshold < 2:
            raise ValueError(f"pooled_threshold must be >= 2, got {pooled_threshold}")
        self.pooled_threshold = pooled_threshold

    def fit(self, df: pd.DataFrame) -> ShiftedBetaGeometric:
        """Pooled pre-pass, then the standard per-series path."""
        df = validate_panel(df)
        y_all = df[TARGET_COL].to_numpy(dtype=float)
        if np.any((y_all < -_EPS) | (y_all > 1.0 + _EPS)):
            raise DataContractError(
                f"{self.name} models retention fractions in [0, 1]; got values in "
                f"[{y_all.min():.3g}, {y_all.max():.3g}]. Build the panel with "
                f"forecast_os.gtm.cohort_panel."
            )
        # The sBG recursion indexes survival by integer age, so each cohort's
        # ds must be consecutive integers starting at the acquisition anchor:
        # a gapped or floating curve would silently map observations to the
        # wrong ages.
        implied_anchor = []
        for uid, g in df.groupby(ID_COL, sort=True):
            ages = g[TIME_COL].to_numpy()
            if not (np.issubdtype(ages.dtype, np.number) and np.all(np.diff(ages) == 1)):
                raise DataContractError(
                    f"{self.name} requires each cohort's 'ds' to be consecutive "
                    f"integer ages (0, 1, 2, ...); cohort {uid!r} violates this. "
                    f"Build the panel with forecast_os.gtm.cohort_panel and fill "
                    f"missing ages first."
                )
            first_age = ages[0]
            if first_age != 0 and first_age != 1:
                raise DataContractError(
                    f"{self.name} reads 'ds' as the cohort age, so each curve must "
                    f"start at age 0 (the acquisition anchor) or age 1 (which implies "
                    f"an age-0 anchor of 1.0); cohort {uid!r} starts at age "
                    f"{first_age}. If its ages are not measured from its own "
                    f"acquisition period, re-index them so they are; if the cohort is "
                    f"genuinely observed only from age {first_age} (earlier periods "
                    f"missing), subtract {first_age} from its ages — sBG is closed "
                    f"under left truncation, so the forecast is unchanged and only "
                    f"cohort_params() shifts, reporting beta + {first_age} for beta."
                )
            if first_age == 1:
                implied_anchor.append(uid)
        # Both thresholds below count the rows the CALLER supplied, so a curve
        # written from age 1 is treated exactly like the same curve written
        # from age 0: the anchor materialized next is bookkeeping, not an
        # observation.
        n_obs = df.groupby(ID_COL, sort=True).size().to_dict()
        min_needed = self.min_train_size
        if max(n_obs.values()) < min_needed:
            raise ForecastOSError(
                f"{self.name} requires at least {min_needed} observations in at "
                f"least one series to fit pooled parameters; longest series has "
                f"{max(n_obs.values())}"
            )
        df = self._materialize_implied_anchors(df, implied_anchor)
        self.pooled_params_ = self._fit_pooled(df)
        # The base-class loop hands _fit_series the y array alone, so each
        # cohort's caller-row count rides alongside it, lined up here against
        # the same groupby(ID_COL, sort=True) over the same frame that the
        # loop iterates — looked up by cohort, so materializing an anchor
        # cannot re-order or re-pair the counts.
        self._n_obs_ = iter(
            [int(n_obs[uid]) for uid, _ in df.groupby(ID_COL, sort=True)]
        )
        # Short cohorts are legal here because the pooled fallback covers them;
        # relax the per-series floor for the base-class loop, then restore it.
        self.min_train_size = 1
        try:
            super().fit(df)
        finally:
            self.min_train_size = min_needed
            self._n_obs_ = None
        for uid in implied_anchor:
            # Drop the synthetic anchor again so fitted_values() mirrors the
            # panel the caller passed. Sigma is unaffected: the anchor row's
            # fitted entry is the NaN warm-up slot, which the residual
            # calculation already skips.
            state = self._series_state[uid]
            for key in ("fitted", "_y", "_ds"):
                state[key] = state[key][1:]
        return self

    def cohort_params(self) -> pd.DataFrame:
        """Fitted per-cohort parameters as (unique_id, alpha, beta, pooled)."""
        self._check_is_fitted()
        rows = [
            {
                ID_COL: uid,
                "alpha": state["alpha_"],
                "beta": state["beta_"],
                "pooled": state["pooled_"],
            }
            for uid, state in self._series_state.items()
        ]
        return pd.DataFrame(rows, columns=[ID_COL, "alpha", "beta", "pooled"])

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _materialize_implied_anchors(df: pd.DataFrame, uids: list) -> pd.DataFrame:
        """Give age-1-starting cohorts their implied age-0 anchor row of 1.0.

        A curve whose ``ds`` starts at 1 has no acquisition observation, so
        full activation is implied. Materializing that anchor puts every
        cohort on one code path — ``y[0]`` is always the age-0 scale — instead
        of leaving the alignment to be guessed from the values, which is what
        shifted whole curves in the first place. The synthetic rows are
        stripped back out of the fitted state at the end of ``fit``.
        """
        if not uids:
            return df
        anchors = pd.DataFrame({ID_COL: uids, TIME_COL: 0, TARGET_COL: 1.0})
        return validate_panel(pd.concat([df, anchors], ignore_index=True))

    def _fit_pooled(self, df: pd.DataFrame) -> tuple[float, float]:
        """Pooled (alpha, beta) from the cohorts' mean per-period retention.

        Normalizing by each cohort's age-0 anchor keeps a cohort with 60%
        activation from dragging the pooled curve down as though it churned
        faster. The ratio is unbounded above, though, so cohorts whose
        retention rises out of age 0 (trial conversion, a late first payment,
        an age-0 snapshot taken before the period closed) contribute values
        above 1 that are not survival probabilities and would dominate the
        mean — one such cohort in twelve was measured moving the pooled
        parameters from (1.0, 4.0) to (4.3, 39.7). Their age-0 value is not
        the cohort's scale, so they cannot inform a survival prior and are
        left out, as are cohorts with no age-0 retention to divide by; the
        pooled curve is then in [0, 1] by construction. If that leaves nothing
        usable the pooled fit has failed and (1.0, 1.0) is returned, the same
        neutral fallback a failed MLE gets.

        The surviving curves are ragged (recent cohorts are short), so they are
        combined by :func:`_pooled_curve` rather than by a position-wise mean
        of the levels — see that function for why the levels version is not a
        survival curve at all.
        """
        curves = []
        for _, g in df.groupby(ID_COL, sort=True):
            y = g[TARGET_COL].to_numpy(dtype=float)
            if y[0] <= _EPS:
                continue
            curve = _survival_curve(y)
            if curve.size and np.all(curve <= 1.0 + _EPS):
                curves.append(curve)
        if not curves:
            return 1.0, 1.0
        alpha, beta, ok = _sbg_mle(_pooled_curve(curves))
        return (alpha, beta) if ok else (1.0, 1.0)

    def _fit_series(self, y: np.ndarray) -> dict:
        # fit() guarantees every cohort now starts at age 0, so y[0] is the
        # anchor and T is the last age — the whole curve's alignment comes
        # from ds, never from whether y[0] happens to reach 1.0.
        horizon = len(y) - 1
        pending = getattr(self, "_n_obs_", None)
        n_obs = len(y) if pending is None else next(pending, len(y))
        # A cohort nobody activated into has no scale to measure survival
        # against, which makes its shape unlearnable but not its forecast: it
        # borrows the pooled shape and carries its own last value forward,
        # exactly like a cohort whose MLE fails.
        pooled = float(y[0]) <= _EPS or n_obs < self.pooled_threshold or horizon < 2
        if not pooled:
            alpha, beta, ok = _sbg_mle(_survival_curve(y))
            pooled = not ok
        if pooled:
            alpha, beta = self.pooled_params_
        _, s = _sbg_survival(alpha, beta, horizon)
        # one-step-ahead conditional fits in the cohort's own units (the age-0
        # scale cancels): carry last observed retention forward by the model's
        # period-over-period retention ratio. Age 0 has no predecessor.
        fitted = np.concatenate([[np.nan], y[:-1] * (s[1:] / np.maximum(s[:-1], 1e-12))])
        return {
            "fitted": fitted,
            "alpha_": alpha,
            "beta_": beta,
            "pooled_": pooled,
            "T_": horizon,
            "s_last_": float(y[-1]),
        }

    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        horizon = state["T_"]
        _, s = _sbg_survival(state["alpha_"], state["beta_"], horizon + h)
        base = max(s[horizon], 1e-12)
        return state["s_last_"] * s[horizon + 1 :] / base
