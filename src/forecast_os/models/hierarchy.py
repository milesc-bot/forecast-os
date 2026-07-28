"""Hierarchical aggregation and forecast reconciliation.

The hierarchy is encoded in ``unique_id`` paths with a separator: leaves
``"west/alice"`` and ``"west/bob"`` roll up into the interior node ``"west"``,
and every top-level node rolls up into the grand total (``total_id``). A flat
panel whose ids contain no separator is a legal one-level hierarchy: its
series are the leaves and the only aggregate is the total.

:func:`aggregate_panel` materializes every ancestor level of a leaf panel;
:class:`HierarchicalReconciler` wraps any registered forecaster and returns
forecasts for ALL nodes whose POINT forecasts are coherent — for every
parent, the children's ``yhat`` sums exactly to the parent's. Interval
bounds are a separate per-column projection and are not guaranteed to be
additive; see :class:`HierarchicalReconciler` for what they do guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster, _check_level
from ..core.exceptions import DataContractError
from ..core.registry import get_model, register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

__all__ = ["HierarchicalReconciler", "aggregate_panel"]

_METHODS = ("bottom_up", "top_down", "mint_ols")


def _split_path(uid, sep: str) -> list:
    return uid.split(sep) if isinstance(uid, str) else [uid]


def _build_tree(leaf_ids, sep: str, total_id) -> tuple[list, list, np.ndarray]:
    """Tree implied by leaf id paths: (all nodes, leaves, summing matrix S).

    ``nodes`` is ordered ``[total, *sorted(interior), *sorted(leaves)]`` and
    ``S`` has shape ``(len(nodes), len(leaves))`` with ``S[i, j] = 1`` iff
    leaf ``j`` is the node ``i`` itself or one of its descendants.
    """
    leaf_ids = list(leaf_ids)
    if any(isinstance(uid, str) for uid in leaf_ids) and not all(
        isinstance(uid, str) for uid in leaf_ids
    ):
        raise DataContractError(
            "unique_id values must be uniformly typed strings for hierarchy "
            f"paths; got mixed types {sorted({type(uid).__name__ for uid in leaf_ids})}"
        )
    leaves = sorted(leaf_ids)
    interior: set = set()
    ancestors: dict = {}
    for leaf in leaves:
        parts = _split_path(leaf, sep)
        if isinstance(leaf, str) and any(p == "" for p in parts):
            raise DataContractError(
                f"series id {leaf!r} contains an empty path segment (sep={sep!r})"
            )
        prefixes = [sep.join(parts[:i]) for i in range(1, len(parts))]
        interior.update(prefixes)
        ancestors[leaf] = prefixes
    clash = interior.intersection(leaves)
    if clash:
        raise DataContractError(
            f"ambiguous hierarchy: id(s) {sorted(clash)!r} appear both as a "
            f"leaf series and as an ancestor of another leaf"
        )
    if total_id in interior or total_id in leaves:
        raise DataContractError(
            f"total_id {total_id!r} collides with a series id; pass a different total_id"
        )
    nodes = [total_id, *sorted(interior), *leaves]
    index = {node: i for i, node in enumerate(nodes)}
    S = np.zeros((len(nodes), len(leaves)))
    for j, leaf in enumerate(leaves):
        S[0, j] = 1.0
        S[index[leaf], j] = 1.0
        for anc in ancestors[leaf]:
            S[index[anc], j] = 1.0
    return nodes, leaves, S


def aggregate_panel(df: pd.DataFrame, sep: str = "/", total_id="total") -> pd.DataFrame:
    """Materialize every ancestor level of a path-encoded leaf panel.

    Returns a panel containing the input leaf series PLUS one summed series
    per ancestor path prefix, plus the grand total under ``total_id``. All
    series under a common parent (hence, via the total, all leaves) must
    share an identical ``ds`` index; otherwise :class:`DataContractError` is
    raised. Only the numeric ``y`` column is aggregated — any extra columns
    in ``df`` are dropped from the result.
    """
    df = validate_panel(df)
    nodes, leaves, S = _build_tree(df[ID_COL].unique(), sep, total_id)
    groups = {uid: g for uid, g in df.groupby(ID_COL, sort=True)}
    ref_ds = groups[leaves[0]][TIME_COL].to_numpy()
    for leaf in leaves[1:]:
        if not np.array_equal(groups[leaf][TIME_COL].to_numpy(), ref_ds):
            raise DataContractError(
                f"series under a common parent must share an aligned 'ds' index "
                f"(re-bucket with gtm.to_panel(span='panel') or "
                f"preprocessing.fill_gaps); {leaves[0]!r} and {leaf!r} differ"
            )
    values = S @ np.vstack([groups[leaf][TARGET_COL].to_numpy(dtype=float) for leaf in leaves])
    out = pd.concat(
        [
            pd.DataFrame({ID_COL: node, TIME_COL: ref_ds, TARGET_COL: values[i]})
            for i, node in enumerate(nodes)
        ],
        ignore_index=True,
    )
    return out.sort_values([ID_COL, TIME_COL], kind="stable").reset_index(drop=True)


@register("reconciled", family="ensemble")
class HierarchicalReconciler(BaseForecaster):
    """Coherent forecasts for every node of a path-encoded hierarchy.

    ``fit(df)`` takes the LEAF panel (ids may contain ``sep``; a flat panel
    with no separator is a single-level hierarchy of leaves + total). The
    member ``model`` may be a forecaster instance or a registry-name string;
    strings are resolved with :func:`get_model` lazily inside :meth:`fit`.
    ``predict`` returns forecasts for ALL nodes (leaves, interior, total).
    The POINT forecasts are coherent to floating-point accuracy: children's
    ``yhat`` sums to their parent's. That guarantee covers ``yhat`` only —
    ``lo``/``hi`` are projected per column and then repaired for ordering,
    which can leave them non-additive (see "Intervals" below).

    Methods:

    - ``bottom_up``: fit/forecast the leaves only; ancestors are sums.
    - ``top_down``: fit/forecast the total plus independent leaf forecasts;
      per horizon step, the leaf forecasts define proportions and each leaf
      becomes ``total * proportion`` (interior nodes are sums of those).
      Signed leaf forecasts that nearly cancel make top-down proportions
      ill-conditioned (a tiny denominator explodes every share), so a step
      is treated as degenerate — falling back to equal shares — when the
      leaf sum is negligible RELATIVE to the summed leaf magnitude:
      ``|sum(yhat)| < 1e-8 * sum(|yhat|)`` (absolute floor ``1e-12`` for
      the all-zero case).
    - ``mint_ols``: fit/forecast every node independently, then project onto
      the coherent subspace with OLS weights: with summing matrix ``S``,
      ``reconciled = S (S'S)^{-1} S' yhat_all`` (pure numpy least squares).

    Intervals: ``lo``/``hi`` columns are pushed through the SAME linear
    mapping as ``yhat`` (top-down proportions are frozen at their
    point-forecast values, keeping the map linear). This is a
    linear-projection approximation, NOT a variance-aware reconciliation:
    the member models' interval widths are combined as if perfectly
    correlated, which is inexact for every method — even ``bottom_up``
    would need the cross-series covariance for exact bounds. A final guard
    re-orders any projected bounds so that ``lo <= yhat <= hi`` always holds,
    then enforces monotone nesting across levels — each higher confidence
    band contains every lower one — because the per-column projection can
    otherwise leave a 50% bound outside the 90% band.

    So the bounds guarantee ordering and nesting, NOT additivity: under
    ``mint_ols`` the projection can invert a leaf band (``lo`` above ``hi``),
    and once the guard repairs that row no per-row fix can restore both
    ordering and additivity. Ordering is the invariant kept, deliberately —
    a lower bound above its upper bound is unusable output, whereas additive
    quantiles are not a statistical property anyway (the quantile of a sum is
    not the sum of the quantiles). Do not sum children's ``lo``/``hi`` and
    expect the parent's.

    ``bottom_up`` is the one mapping where they do happen to line up: its leaf
    columns are the member's own bands pushed through the same summing matrix
    as ``yhat``, so when those bands are ordered the guard has nothing to
    repair. That is a property of the mapping, not part of the contract — only
    ``yhat`` is. It does **not** extend to ``top_down``, whose leaf bands are
    ``prop * total`` rather than the member's columns, so a proportion derived
    from ``yhat`` can leave them non-additive even when every member band is
    ordered.
    """

    def __init__(self, model="auto_ets", method="mint_ols", sep="/", total_id="total"):
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")
        if not isinstance(sep, str) or not sep:
            raise ValueError(f"sep must be a non-empty string, got {sep!r}")
        self.model = model
        self.method = method
        self.sep = sep
        self.total_id = total_id

    def clone(self) -> HierarchicalReconciler:
        """Deep-clone an instance member; registry-name strings pass through."""
        params = self.get_params()
        if not isinstance(params["model"], str):
            params["model"] = params["model"].clone()
        return type(self)(**params)

    def fit(self, df: pd.DataFrame) -> HierarchicalReconciler:
        df = validate_panel(df)
        self._nodes_, self._leaves_, self._S_ = _build_tree(
            df[ID_COL].unique(), self.sep, self.total_id
        )
        full = aggregate_panel(df, sep=self.sep, total_id=self.total_id)
        if self.method == "bottom_up":
            train_ids = list(self._leaves_)
        elif self.method == "top_down":
            train_ids = [self.total_id, *self._leaves_]
        else:  # mint_ols
            train_ids = list(self._nodes_)
        train = full[full[ID_COL].isin(train_ids)].reset_index(drop=True)
        member = get_model(self.model) if isinstance(self.model, str) else self.model.clone()
        self._member_ = member.fit(train)
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        levels = _check_level(level)
        pred = (
            self._member_.predict(int(h), level=level)
            .sort_values([ID_COL, TIME_COL], kind="stable")
            .reset_index(drop=True)
        )
        value_cols = [c for c in pred.columns if c not in (ID_COL, TIME_COL)]
        series = {uid: g for uid, g in pred.groupby(ID_COL, sort=False)}

        def base(ids, col):
            return np.vstack([series[uid][col].to_numpy(dtype=float) for uid in ids])

        if self.method == "top_down":
            leaf_yhat = base(self._leaves_, "yhat")
            colsum = leaf_yhat.sum(axis=0)
            # Scale-relative degeneracy test: signed leaf forecasts that
            # nearly cancel leave a denominator that is tiny relative to the
            # leaves' magnitude, so dividing by it would explode the shares.
            # Such steps (and truly all-zero steps, via the 1e-12 floor)
            # fall back to equal shares instead.
            scale = np.abs(leaf_yhat).sum(axis=0)
            degenerate = np.abs(colsum) < np.maximum(1e-8 * scale, 1e-12)
            prop = leaf_yhat / np.where(degenerate, 1.0, colsum)
            prop[:, degenerate] = 1.0 / len(self._leaves_)

        full: dict[str, np.ndarray] = {}
        for col in value_cols:
            if self.method == "bottom_up":
                leaf_mat = base(self._leaves_, col)
            elif self.method == "top_down":
                leaf_mat = prop * series[self.total_id][col].to_numpy(dtype=float)
            else:  # mint_ols: leaf coordinates of the OLS projection
                leaf_mat = np.linalg.lstsq(self._S_, base(self._nodes_, col), rcond=None)[0]
            full[col] = self._S_ @ leaf_mat

        for lvl in levels:
            lo = np.minimum(full[f"lo-{lvl}"], full[f"hi-{lvl}"])
            hi = np.maximum(full[f"lo-{lvl}"], full[f"hi-{lvl}"])
            full[f"lo-{lvl}"] = np.minimum(lo, full["yhat"])
            full[f"hi-{lvl}"] = np.maximum(hi, full["yhat"])
        # Each column is projected independently, so the per-level guard above
        # cannot prevent bands from crossing BETWEEN levels (a 50% bound can
        # land outside the 90% band). Enforce monotone nesting: walk the
        # ascending-sorted levels and widen each band to contain the previous
        # one; the running min/max accumulates every lower level.
        for prev, lvl in zip(levels[:-1], levels[1:]):
            full[f"lo-{lvl}"] = np.minimum(full[f"lo-{lvl}"], full[f"lo-{prev}"])
            full[f"hi-{lvl}"] = np.maximum(full[f"hi-{lvl}"], full[f"hi-{prev}"])

        ds = series[self._leaves_[0]][TIME_COL].to_numpy()
        out = pd.concat(
            [
                pd.DataFrame(
                    {ID_COL: node, TIME_COL: ds, **{c: full[c][i] for c in value_cols}}
                )
                for i, node in enumerate(self._nodes_)
            ],
            ignore_index=True,
        )
        return out.sort_values([ID_COL, TIME_COL], kind="stable").reset_index(drop=True)
