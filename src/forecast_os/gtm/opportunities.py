"""Deal-level win-probability scoring and the probabilistic pipeline forecast.

The go-to-market layer speaks two grains. Most of forecast-os lives on the
``(unique_id, ds, y)`` time-series panel; this module lives on the OTHER
grain — the deal/opportunity table, one row per open or closed deal with at
least ``opp_id``, ``amount``, and ``stage`` columns (plus arbitrary
feature/segment columns). Because that shape is not a panel, the classes here
are standalone estimators (like :class:`~forecast_os.finance.garch.GARCH11`),
NOT registry ``BaseForecaster`` models.

:class:`DealScorer` is a calibrated logistic win-probability model. It fits an
L2-regularized logistic regression on standardized numeric features by
maximum penalized likelihood (L-BFGS-B with an analytic gradient — no
scikit-learn), then optionally Platt-scales the scores on a held-out split so
the predicted probabilities are reliable (a monotone scale-and-shift
calibration).

:func:`weighted_pipeline` turns those per-deal win probabilities into the
number a revenue leader actually asks for: the probabilistic pipeline
forecast. Each open deal ``i`` is treated as an independent
``Bernoulli(p_i)`` worth ``amount_i``, so a segment's booked total has
``expected = sum(p_i * amount_i)`` and
``variance = sum(p_i (1 - p_i) amount_i^2)`` exactly, with an optional
Normal-approximation interval. This is the calibrated, deal-level pipeline
number that a flat "stage times a fixed win-rate" weighting cannot give.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from ..core.exceptions import DataContractError, ForecastOSError, NotFittedError

__all__ = ["DealScorer", "weighted_pipeline"]

_PROB_EPS = 1e-12
_EXCLUDED_AUTO = ("opp_id", "amount")


def _neg_log_likelihood(params: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float):
    """Penalized logistic NLL and its analytic gradient at ``params``.

    ``params`` packs the ``k`` feature weights followed by the intercept.
    The per-sample loss is ``softplus(z) - y z`` with ``z = X w + b`` (the
    numerically stable logistic cross-entropy); the L2 penalty
    ``0.5 l2 ||w||^2`` covers the weights only, never the intercept, so a
    class-imbalanced base rate is never shrunk toward 0.5.
    """
    w = params[:-1]
    b = params[-1]
    z = X @ w + b
    p = expit(z)
    # softplus(z) - y z == -y log p - (1 - y) log(1 - p), evaluated stably
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z)) + 0.5 * l2 * float(w @ w)
    resid = p - y
    grad_w = X.T @ resid + l2 * w
    grad_b = float(np.sum(resid))
    return nll, np.concatenate([grad_w, [grad_b]])


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """L2-regularized logistic MLE via L-BFGS-B; returns ``(weights, intercept)``."""
    k = X.shape[1]
    res = minimize(
        _neg_log_likelihood,
        x0=np.zeros(k + 1),
        args=(X, y, l2),
        jac=True,
        method="L-BFGS-B",
    )
    return res.x[:-1], float(res.x[-1])


def _fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a monotone Platt calibrator ``p = sigmoid(a z + b)`` on held-out scores.

    Minimizes the calibration NLL over ``(a, b)`` with ``a`` bounded strictly
    positive so the mapping stays monotone increasing (reliability-improving
    rather than rank-altering). A non-finite or degenerate solution falls back
    to the identity ``(1, 0)``.
    """

    def obj(params: np.ndarray):
        a, b = params
        zz = a * z + b
        nll = float(np.sum(np.logaddexp(0.0, zz) - y * zz))
        resid = expit(zz) - y
        return nll, np.array([float(resid @ z), float(np.sum(resid))])

    res = minimize(
        obj, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
        bounds=[(1e-6, None), (None, None)],
    )
    a, b = float(res.x[0]), float(res.x[1])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0:
        return 1.0, 0.0
    return a, b


class DealScorer:
    """Calibrated logistic win-probability model for closed/open deals.

    Parameters
    ----------
    features : the numeric feature columns to score on. When ``None`` (the
        default) they are auto-detected at :meth:`fit` as every numeric
        column except ``opp_id``, ``amount``, and the target label. A
        ``features`` argument passed to :meth:`fit` overrides this one.
    l2 : L2 penalty strength on the standardized weights (the intercept is
        never penalized). Larger values shrink coefficients toward zero.
    calibrate : when True, hold out a random split of the training deals,
        fit the logistic on the remainder, and Platt-scale its scores on the
        held-out split so predicted probabilities are calibrated. When False
        the logistic is fit on all deals with no post-hoc calibration.
    seed : seed for the calibration split (``np.random.default_rng``).

    Attributes set by :meth:`fit`
    -----------------------------
    coef_ : pandas Series of the fitted weights indexed by feature name, on
        the STANDARDIZED scale (so magnitudes are directly comparable as
        feature importances); the sign is the direction of the effect on the
        win log-odds.
    intercept_ : float intercept on the standardized scale.
    feature_names_ : the ordered feature columns used.
    calibrated_ : whether Platt calibration was actually applied (it falls
        back to the identity when the sample is too small or a split is
        single-class).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        l2: float = 1.0,
        calibrate: bool = True,
        seed: int = 0,
    ):
        if l2 < 0:
            raise ValueError(f"l2 must be non-negative, got {l2}")
        self.features = features
        self.l2 = l2
        self.calibrate = calibrate
        self.seed = seed

    # -- public API ----------------------------------------------------------

    def fit(
        self,
        deals: pd.DataFrame,
        target: str = "won",
        features: list[str] | None = None,
    ) -> DealScorer:
        """Fit the win-probability model on historical CLOSED deals.

        ``deals`` must carry the boolean (or 0/1) ``target`` label and the
        numeric feature columns. Features are the ``features`` argument, else
        the constructor ``features``, else auto-detected numeric columns
        (excluding ``opp_id``, ``amount``, and ``target``). Raises
        :class:`DataContractError` when the target is not boolean/0-1 or
        has a single class, and when no usable numeric feature survives.
        """
        if not isinstance(deals, pd.DataFrame):
            raise DataContractError(
                f"expected a pandas DataFrame of deals, got {type(deals).__name__}"
            )
        if len(deals) == 0:
            raise DataContractError("deals frame is empty")
        y = self._extract_target(deals, target)
        feats = self._resolve_features(deals, target, features)

        X_raw = deals[feats].to_numpy(dtype=float)
        self._check_finite(X_raw, feats)
        self.feature_names_ = list(feats)

        # Standardize on the full training frame for a stable, reusable scale.
        self.feature_means_ = X_raw.mean(axis=0)
        stds = X_raw.std(axis=0)
        stds[stds < 1e-12] = 1.0  # constant column -> standardized to 0
        self.feature_stds_ = stds
        X = (X_raw - self.feature_means_) / self.feature_stds_

        w, b, cal_a, cal_b, calibrated = self._fit_core(X, y)
        self.coef_ = pd.Series(w, index=self.feature_names_, name="coef")
        self.intercept_ = float(b)
        self._calib_a = float(cal_a)
        self._calib_b = float(cal_b)
        self.calibrated_ = bool(calibrated)
        self._is_fitted = True
        return self

    def predict_proba(self, open_deals: pd.DataFrame) -> pd.Series:
        """Win probability ``P(win)`` for each open deal, in the open (0, 1).

        Applies the stored standardization and coefficients, then the Platt
        calibration, returning a pandas Series aligned to ``open_deals``'
        index. ``open_deals`` need not carry the target label, but must carry
        every fitted feature column with finite values.
        """
        self._check_is_fitted()
        if not isinstance(open_deals, pd.DataFrame):
            raise DataContractError(
                f"expected a pandas DataFrame of deals, got {type(open_deals).__name__}"
            )
        missing = [c for c in self.feature_names_ if c not in open_deals.columns]
        if missing:
            raise DataContractError(
                f"open_deals is missing fitted feature column(s) {missing}"
            )
        X_raw = open_deals[self.feature_names_].to_numpy(dtype=float)
        self._check_finite(X_raw, self.feature_names_)
        X = (X_raw - self.feature_means_) / self.feature_stds_
        z = X @ self.coef_.to_numpy() + self.intercept_
        p = expit(self._calib_a * z + self._calib_b)
        p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
        return pd.Series(p, index=open_deals.index, name="win_proba")

    # -- internals -----------------------------------------------------------

    def _fit_core(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float, float, float, bool]:
        """Fit the logistic and (optionally) a held-out Platt calibrator.

        With calibration the logistic is fit on a random ~75% split and the
        calibrator on the held-out ~25%, so the calibrator never sees scores
        from deals it was trained on. When the sample is too small or either
        split is single-class, calibration is skipped and the logistic is fit
        on all deals with an identity calibrator.
        """
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        if self.calibrate and n >= 20:
            perm = rng.permutation(n)
            n_hold = max(int(round(0.25 * n)), 1)
            hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
            both = y[train_idx]
            held = y[hold_idx]
            if (
                len(np.unique(both)) == 2
                and len(np.unique(held)) == 2
                and len(hold_idx) >= 4
            ):
                w, b = _fit_logistic(X[train_idx], y[train_idx], self.l2)
                z_hold = X[hold_idx] @ w + b
                cal_a, cal_b = _fit_platt(z_hold, y[hold_idx])
                return w, b, cal_a, cal_b, True
        w, b = _fit_logistic(X, y, self.l2)
        return w, b, 1.0, 0.0, False

    def _extract_target(self, deals: pd.DataFrame, target: str) -> np.ndarray:
        if target not in deals.columns:
            raise DataContractError(
                f"target column {target!r} not found; historical closed deals "
                f"must carry a boolean win label"
            )
        col = deals[target]
        if pd.api.types.is_bool_dtype(col):
            y = col.to_numpy(dtype=float)
        elif pd.api.types.is_numeric_dtype(col):
            y = col.to_numpy(dtype=float)
            if not np.isfinite(y).all():
                raise DataContractError(
                    f"target column {target!r} contains NaN/inf; every closed "
                    f"deal must have a definite win/loss label"
                )
            uniq = set(np.unique(y).tolist())
            if not uniq.issubset({0.0, 1.0}):
                raise DataContractError(
                    f"target column {target!r} must be boolean or 0/1; got "
                    f"values {sorted(uniq)[:5]}"
                )
        else:
            raise DataContractError(
                f"target column {target!r} must be boolean or 0/1, got dtype "
                f"{col.dtype}"
            )
        if len(np.unique(y)) < 2:
            raise DataContractError(
                f"target column {target!r} has a single class; a scorer needs "
                f"both won and lost deals to train"
            )
        return y

    def _resolve_features(
        self, deals: pd.DataFrame, target: str, features: list[str] | None
    ) -> list[str]:
        feats = features if features is not None else self.features
        if feats is not None:
            feats = list(feats)
            missing = [c for c in feats if c not in deals.columns]
            if missing:
                raise DataContractError(
                    f"feature column(s) {missing} not found in deals"
                )
            bad = [c for c in feats if not pd.api.types.is_numeric_dtype(deals[c])]
            if bad:
                raise DataContractError(
                    f"feature column(s) {bad} are not numeric; the scorer needs "
                    f"numeric features (encode categoricals first)"
                )
            if not feats:
                raise DataContractError("no feature columns provided")
            return feats
        excluded = {*_EXCLUDED_AUTO, target}
        auto = [
            c
            for c in deals.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(deals[c])
        ]
        if not auto:
            raise DataContractError(
                "no usable numeric feature columns found (auto-detect excludes "
                f"{sorted(excluded)}); pass features=[...] with numeric columns"
            )
        return auto

    @staticmethod
    def _check_finite(X: np.ndarray, feats: list[str]) -> None:
        col_ok = np.isfinite(X).all(axis=0)
        if not col_ok.all():
            bad = [f for f, ok in zip(feats, col_ok) if not ok]
            raise DataContractError(
                f"feature column(s) {bad} contain non-finite value(s) (NaN/inf); "
                f"impute them first (see forecast_os.preprocessing) or drop them"
            )

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("DealScorer is not fitted; call fit() first")


def weighted_pipeline(
    open_deals: pd.DataFrame,
    scorer: DealScorer | None = None,
    proba: np.ndarray | pd.Series | None = None,
    by: str | list[str] | None = None,
    amount_col: str = "amount",
    level: int | None = None,
) -> pd.DataFrame:
    """Probabilistic pipeline forecast from per-deal win probabilities.

    Each open deal ``i`` is an independent ``Bernoulli(p_i)`` worth
    ``amount_i``. For each group (a single total when ``by`` is None) the
    booked amount has, exactly,

    - ``expected = sum(p_i * amount_i)``
    - ``variance = sum(p_i (1 - p_i) amount_i^2)``

    and, when ``level`` is given, a Normal-approximation interval
    ``expected +/- z_level * sqrt(variance)`` clamped to the attainable
    support ``[0, sum(amount)]`` (a group's realized won-$ cannot be negative
    or exceed the sum of its deal amounts). ``expected`` and ``variance`` are
    exact for independent deals; the interval is the only approximation.

    The Normal approximation is well-calibrated when a group holds many
    comparably sized deals (the sum-of-Bernoullis tends to Normal). For a
    *small or lumpy* group — a handful of deals, or one deal dominating the
    variance — the true distribution of won-$ is multimodal and the
    Normal-approx interval's coverage can be well off nominal; treat the
    interval as indicative there and prefer segmenting into larger groups.

    Parameters
    ----------
    open_deals : one row per open opportunity, with the ``amount_col`` value
        and (when ``by`` is given) the grouping columns.
    scorer : a fitted :class:`DealScorer`; its ``predict_proba(open_deals)``
        supplies ``p``. Ignored when ``proba`` is passed.
    proba : an array/Series of win probabilities aligned to ``open_deals``
        (a Series is aligned by index, an array positionally). Takes
        precedence over ``scorer``. Exactly one of ``scorer``/``proba`` is
        required.
    by : segment column(s) to group by; ``None`` returns a single total row.
    amount_col : the deal-value column (default ``"amount"``).
    level : confidence level in (0, 100) for the interval columns
        ``lo-{level}``/``hi-{level}``; ``None`` omits them.

    Returns
    -------
    A DataFrame with the ``by`` column(s) (if any), ``expected``, the
    ``lo-{level}``/``hi-{level}`` columns (if ``level``), and ``n_deals``.
    """
    if not isinstance(open_deals, pd.DataFrame):
        raise ForecastOSError(
            f"open_deals must be a pandas DataFrame, got {type(open_deals).__name__}"
        )
    if len(open_deals) == 0:
        raise ForecastOSError("open_deals is empty")
    if amount_col not in open_deals.columns:
        raise ForecastOSError(
            f"open_deals is missing amount column {amount_col!r}; pass amount_col"
        )

    p = _resolve_proba(open_deals, scorer, proba)
    amounts = open_deals[amount_col].to_numpy(dtype=float)
    if not np.isfinite(amounts).all():
        raise ForecastOSError(
            f"amount column {amount_col!r} contains non-finite value(s) (NaN/inf)"
        )

    levels = _check_pipeline_level(level)
    work = pd.DataFrame(
        {"_pa": p * amounts, "_var": p * (1.0 - p) * amounts**2, "_amt": amounts},
        index=open_deals.index,
    )

    if by is None:
        expected = np.array([work["_pa"].sum()])
        variance = np.array([work["_var"].sum()])
        total_amt = np.array([work["_amt"].sum()])
        n_deals = np.array([len(work)])
        out = pd.DataFrame({"expected": expected})
        keys: list[str] = []
    else:
        keys = [by] if isinstance(by, str) else list(by)
        missing = [c for c in keys if c not in open_deals.columns]
        if missing:
            raise ForecastOSError(
                f"open_deals is missing grouping column(s) {missing}"
            )
        for c in keys:
            work[c] = open_deals[c].to_numpy()
        grouped = work.groupby(keys, sort=True)
        agg = grouped.agg(
            expected=("_pa", "sum"), _variance=("_var", "sum"), _total=("_amt", "sum")
        )
        n_deals = grouped.size().to_numpy()
        variance = agg["_variance"].to_numpy()
        expected = agg["expected"].to_numpy()
        total_amt = agg["_total"].to_numpy()
        out = agg[["expected"]].reset_index()

    if levels is not None:
        z = float(stats.norm.ppf(0.5 + levels / 200.0))
        sd = np.sqrt(variance)
        # The realized won-$ of a group lives in [0, sum(amount)] — every deal
        # contributes either 0 or its full amount. Clamp the Normal-approx band
        # to that support so a lumpy segment never reports more (or less) than
        # is physically attainable.
        out[f"lo-{levels}"] = np.maximum(expected - z * sd, 0.0)
        out[f"hi-{levels}"] = np.minimum(expected + z * sd, total_amt)
    out["n_deals"] = np.asarray(n_deals, dtype=int)
    return out.reset_index(drop=True)


def _resolve_proba(
    open_deals: pd.DataFrame,
    scorer: DealScorer | None,
    proba: np.ndarray | pd.Series | None,
) -> np.ndarray:
    """Resolve the per-deal win probabilities, validating range and length."""
    if proba is None and scorer is None:
        raise ForecastOSError(
            "provide a fitted scorer or a proba array; weighted_pipeline needs "
            "per-deal win probabilities"
        )
    if proba is not None:
        source: object = proba
    else:
        source = scorer.predict_proba(open_deals)
    if isinstance(source, pd.Series):
        p = source.reindex(open_deals.index).to_numpy(dtype=float)
    else:
        p = np.asarray(source, dtype=float).ravel()
    if p.shape[0] != len(open_deals):
        raise ForecastOSError(
            f"proba has length {p.shape[0]} but open_deals has "
            f"{len(open_deals)} row(s)"
        )
    if not np.isfinite(p).all():
        raise ForecastOSError(
            "win probabilities contain non-finite value(s) (NaN/inf); a Series "
            "proba must be aligned to the open_deals index"
        )
    if (p < -1e-9).any() or (p > 1.0 + 1e-9).any():
        raise ForecastOSError(
            "win probabilities must lie in [0, 1]; got values outside that range"
        )
    return np.clip(p, 0.0, 1.0)


def _check_pipeline_level(level: int | None) -> int | None:
    if level is None:
        return None
    if not (0 < level < 100):
        raise ValueError(f"level must be in (0, 100), got {level}")
    return int(level)
