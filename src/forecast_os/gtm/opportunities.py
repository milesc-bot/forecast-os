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
scikit-learn), then optionally Platt-scales the scores against cross-fitted
out-of-fold predictions so the predicted probabilities are reliable (a
monotone scale-and-shift calibration).

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

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from ..core.exceptions import DataContractError, ForecastOSError, NotFittedError

__all__ = ["DealScorer", "weighted_pipeline"]

_PROB_EPS = 1e-12
_EXCLUDED_AUTO = ("opp_id", "amount")

# A Platt map has two free parameters fitted on out-of-fold scores, so it needs
# a workable number of BOTH outcomes: with fewer than this many of the rarer
# class the two parameters are estimated on a handful of points and the map is
# not worth its variance, so calibration is skipped and the identity is used
# instead. See _fit_core.
#
# There is deliberately no minimum on the total deal count. An earlier n >= 100
# floor was measured (120 seeds per cell, out-of-sample log loss against a
# 4000-deal test frame) and did not carry its weight: at the default l2 = 1 it
# bought 0.5-1.2% of mean log loss at n = 60..99, while at l2 >= 5 — a
# documented knob whose whole effect is to over-shrink and under-confidence the
# logistic, which is exactly what post-hoc Platt scaling repairs — it COST
# 4-30%. It was also a cliff: fitting on 99 vs 100 of the same deals moved
# individual win probabilities by up to 0.26. The bounded (smoothed-target)
# objective and cross-fitting are what removed the runaway slope this constant
# was introduced alongside; the ungated worst case over that sweep is 0.69 nats
# against a 0.68 base rate, not the 6.4 the floor was written to prevent.
_CALIB_MIN_PER_CLASS = 20
_CALIB_FOLDS = 5


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


def _platt_targets(y: np.ndarray) -> np.ndarray:
    """Platt (1999) smoothed calibration targets for a 0/1 label vector.

    Wins map to ``(N+ + 1) / (N+ + 2)`` and losses to ``1 / (N- + 2)`` rather
    than to a hard 1 and 0. This is what keeps the calibration objective
    BOUNDED: against hard 0/1 targets a separable score vector has no finite
    maximum-likelihood ``(a, b)`` — the slope runs away until the optimizer's
    own convergence test stops it, pinning every probability to 0 or 1. Against
    targets strictly inside (0, 1) the loss diverges as ``a -> inf``, so the
    optimum is finite and the fitted map stays a mild rescaling.
    """
    n_pos = float(np.count_nonzero(y > 0.5))
    n_neg = float(y.shape[0] - n_pos)
    return np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))


def _fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a monotone Platt calibrator ``p = sigmoid(a z + b)`` on out-of-sample scores.

    Minimizes the calibration NLL against :func:`_platt_targets`' smoothed
    labels over ``(a, b)``, with ``a`` bounded strictly positive so the mapping
    stays monotone increasing (reliability-improving rather than rank-altering)
    and no upper bound needed — the smoothed targets already make the objective
    coercive in ``a``. A non-finite or degenerate solution falls back to the
    identity ``(1, 0)``.
    """
    t = _platt_targets(y)

    def obj(params: np.ndarray):
        a, b = params
        zz = a * z + b
        nll = float(np.sum(np.logaddexp(0.0, zz) - t * zz))
        resid = expit(zz) - t
        return nll, np.array([float(resid @ z), float(np.sum(resid))])

    res = minimize(
        obj, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
        bounds=[(1e-6, None), (None, None)],
    )
    a, b = float(res.x[0]), float(res.x[1])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0:
        return 1.0, 0.0
    return a, b


def _out_of_fold_scores(
    X: np.ndarray, y: np.ndarray, l2: float, rng: np.random.Generator, n_folds: int
) -> np.ndarray | None:
    """Cross-fitted logistic scores: every deal's coefficients come from a model
    that never saw it.

    Splits the deals into ``n_folds`` random folds, fits the logistic on the
    other folds, and scores the held-out one — so all ``n`` deals contribute an
    honest out-of-sample score to the calibrator instead of a single 25% split
    contributing ``n / 4``. Returns ``None`` when any fold's training subset is
    single-class, in which case the caller skips calibration.

    The COEFFICIENTS are strictly out-of-fold; the standardization is not. ``X``
    arrives already centred and scaled on all ``n`` rows (see :meth:`fit`), so
    each fold model sees features whose scale was computed with the held-out
    rows included. That is a mild, well-known leak, unchanged from the earlier
    single-split calibrator, and negligible for O(n) statistics — but it is the
    reason this says "coefficients" rather than "nothing".
    """
    n = y.shape[0]
    z = np.empty(n, dtype=float)
    for fold in np.array_split(rng.permutation(n), n_folds):
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        if len(np.unique(y[mask])) < 2:
            return None
        w, b = _fit_logistic(X[mask], y[mask], l2)
        z[fold] = X[fold] @ w + b
    return z


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
    calibrate : when True, score every training deal out-of-fold (5-fold
        cross-fitting) and Platt-scale those honest scores so predicted
        probabilities are calibrated. The reported logistic is fit on all
        deals either way — calibration costs no training data — and is
        skipped, with a warning, when either outcome is too rare to estimate
        a two-parameter map (see ``calibrated_``). When False no post-hoc
        calibration is applied at all.
    seed : seed for the cross-fitting folds (``np.random.default_rng``).

    Attributes set by :meth:`fit`
    -----------------------------
    coef_ : pandas Series of the fitted weights indexed by feature name, on
        the STANDARDIZED scale (so magnitudes are directly comparable as
        feature importances); the sign is the direction of the effect on the
        win log-odds.
    intercept_ : float intercept on the standardized scale.
    feature_names_ : the ordered feature columns used.
    calibrated_ : whether Platt calibration was actually applied. It falls
        back to the identity map — leaving the logistic's own probabilities
        untouched — when there are fewer than 20 of either outcome, or when a
        cross-fitting fold is single-class. Both fallbacks warn when
        ``calibrate`` is True; there is no minimum on the total deal count.
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
        """Fit the logistic on all deals and (optionally) a cross-fitted Platt map.

        The reported logistic is ALWAYS fit on every deal — calibration costs
        no training data. The calibrator is fit on cross-fitted (out-of-fold)
        scores, so it sees an honest out-of-sample score for every deal without
        any of them being withheld from the final model.

        Calibration is skipped — identity map, ``calibrated_ = False`` — when
        either outcome has fewer than ``_CALIB_MIN_PER_CLASS`` deals, or if any
        fold's training subset turns out single-class. Both skips warn when the
        caller asked for calibration, since ``calibrate=True`` is the default
        and a silently uncalibrated scorer is indistinguishable from a
        calibrated one at the call site.
        """
        rng = np.random.default_rng(self.seed)
        w, b = _fit_logistic(X, y, self.l2)
        if not self.calibrate:
            return w, b, 1.0, 0.0, False
        n = int(y.shape[0])
        n_pos = int(np.count_nonzero(y > 0.5))
        n_neg = n - n_pos
        if min(n_pos, n_neg) < _CALIB_MIN_PER_CLASS:
            warnings.warn(
                f"ignoring calibrate=True: {n} closed deal(s) ({n_pos} won / "
                f"{n_neg} lost) is too few to fit a Platt map, which needs at "
                f"least {_CALIB_MIN_PER_CLASS} of each outcome; the logistic's "
                f"own probabilities are returned (calibrated_ = False)",
                UserWarning,
                stacklevel=3,
            )
            return w, b, 1.0, 0.0, False
        z_oof = _out_of_fold_scores(X, y, self.l2, rng, _CALIB_FOLDS)
        if z_oof is None:
            warnings.warn(
                "ignoring calibrate=True: a cross-fitting fold's training "
                "subset was single-class, so no out-of-fold scores could be "
                "produced; the logistic's own probabilities are returned "
                "(calibrated_ = False)",
                UserWarning,
                stacklevel=3,
            )
            return w, b, 1.0, 0.0, False
        cal_a, cal_b = _fit_platt(z_oof, y)
        return w, b, cal_a, cal_b, True

    def _extract_target(self, deals: pd.DataFrame, target: str) -> np.ndarray:
        if target not in deals.columns:
            raise DataContractError(
                f"target column {target!r} not found; historical closed deals "
                f"must carry a boolean win label"
            )
        col = deals[target]
        is_bool = pd.api.types.is_bool_dtype(col)
        if not (is_bool or pd.api.types.is_numeric_dtype(col)):
            raise DataContractError(
                f"target column {target!r} must be boolean or 0/1, got dtype "
                f"{col.dtype}"
            )
        # Missingness is checked BEFORE the dtype dispatch and before the
        # float conversion: pandas' nullable dtypes ("boolean", "Int64",
        # "Float64") are bool/numeric by is_*_dtype but carry pd.NA, which
        # converts to a silent NaN. An unlabelled deal must never train the
        # model — a NaN label poisons every likelihood term and yields an
        # all-zero "null model" that scores every deal at p = 0.5.
        n_null = int(col.isna().sum())
        if n_null:
            raise DataContractError(
                f"target column {target!r} contains {n_null} NaN/NA value(s); "
                f"every closed deal must have a definite win/loss label"
            )
        y = col.to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise DataContractError(
                f"target column {target!r} contains NaN/inf; every closed "
                f"deal must have a definite win/loss label"
            )
        if not is_bool:
            uniq = set(np.unique(y).tolist())
            if not uniq.issubset({0.0, 1.0}):
                raise DataContractError(
                    f"target column {target!r} must be boolean or 0/1; got "
                    f"values {sorted(uniq)[:5]}"
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
        The returned segments always partition ``open_deals`` exactly — their
        ``expected`` and ``n_deals`` sum to the ungrouped totals. Deals with a
        null segment label are therefore kept, as their own NaN-keyed row, and
        warned about; they are never silently dropped. Relabel them beforehand
        if you want them named (on a ``category`` dtype that means
        ``add_categories`` first, or ``.astype(object)``).
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
        # pandas' groupby drops null keys by default, which would delete those
        # deals from the output entirely: the segments would no longer sum to
        # the ungrouped total, silently and by an unbounded amount. An
        # unassigned region/owner/territory is an everyday CRM state, though, so
        # the fix is to KEEP those deals (dropna=False below) as their own
        # null-keyed segment — which restores the partition exactly — and warn,
        # rather than to refuse the frame.
        for c in keys:
            null_mask = open_deals[c].isna().to_numpy()
            n_null = int(null_mask.sum())
            if n_null:
                lost = float(work["_pa"].to_numpy()[null_mask].sum())
                warnings.warn(
                    f"grouping column {c!r} has {n_null} null segment label(s) "
                    f"covering {lost:,.2f} of expected pipeline; they are "
                    f"reported as one unlabelled (NaN-keyed) segment so the "
                    f"segments still sum to the ungrouped total",
                    UserWarning,
                    stacklevel=2,
                )
        for c in keys:
            work[c] = open_deals[c].to_numpy()
        grouped = work.groupby(keys, sort=True, dropna=False)
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
