"""Pipeline waterfall (bridge) from two point-in-time opportunity snapshots.

Two deal snapshots — a ``before`` and an ``after`` opportunity table, each
one row per deal on the DEAL-GRAIN ``(opp_id, amount, stage[, close_date])``
contract (typically ``store.load(as_of=t0, kind="deals")`` and ``t1``) — are
reconciled into a classic pipeline bridge: which deals were created, advanced,
expanded/contracted, pushed, won, lost, or removed between the two dates, and
how each movement contributed to the open-pipeline dollar total.

Categorization is a strict PARTITION of the deal universe (the union of
``opp_id`` across the two snapshots): every deal lands in exactly one category,
so no deal is dropped or double-counted. Each deal is classified from its
``after`` state, by this priority:

1. ``removed``    — in ``before`` but absent from ``after`` (dropped/deleted).
2. ``won``        — ``after`` stage equals ``won_stage``.
3. ``lost``       — ``after`` stage equals ``lost_stage``.
4. ``created``    — appears in ``after`` open, and was not open before (new or
   re-entered the pipeline).
5. for a deal open in BOTH snapshots, the amount change wins first, then the
   stage move, then the close-date move:

   - ``expanded``   — ``amount`` increased.
   - ``contracted`` — ``amount`` decreased.
   - ``advanced``   — same ``amount``, later ``stage`` (higher index in
     ``stages``).
   - ``regressed``  — same ``amount``, earlier ``stage``.
   - ``pushed``     — same ``amount``/``stage``, ``close_date`` moved later
     (needs ``close_col``).
   - ``pulled_in``  — same ``amount``/``stage``, ``close_date`` moved earlier.
   - ``unchanged``  — no material change.

Sign convention (the ``amount`` column is the SIGNED contribution to the OPEN
pipeline total, i.e. the sum of ``amount`` over deals whose stage is not
``won_stage``/``lost_stage``):

- ``created``    = ``+amount`` in ``after`` (enters open pipeline).
- ``expanded``   = ``+(amount_after - amount_before)`` (> 0).
- ``contracted`` = ``amount_after - amount_before`` (< 0).
- ``won``/``lost``/``removed`` = ``-amount_before`` (the value that LEFT the
  open pipeline; ``0`` for a deal created-and-closed in the same period, which
  never entered the open pipeline).
- ``advanced``/``regressed``/``pushed``/``pulled_in``/``unchanged`` = ``0``
  (status-only moves that do not change the open-pipeline dollar total).

With these signs the categories reconcile the bridge exactly::

    opening_open_pipeline + sum(amount) == closing_open_pipeline

so ``won``/``lost`` here show the pipeline REDUCTION (their ``before`` value),
not the booked revenue. :func:`waterfall_summary` wraps the categories with
opening/closing anchor rows and a running total, ready to render as a bridge.

Because that identity is the point of the bridge, a null (``NaN``/``pd.NA``)
``amount`` on a deal that is OPEN in either snapshot is a
:class:`DataContractError`, not a zero: an amount that is merely missing has no
defensible signed contribution, and imputing one would close the bridge on a
number nobody supplied. Fill or drop those deals before bridging.

A null on a deal that is already CLOSED in the snapshot it appears in is fine —
the signs above never read it (won/lost/removed contribute the BEFORE value, and
only for deals that were open before; the open-pipeline totals sum open stages
only), so the bridge is still exactly computable. That covers the common CRM
habit of blanking the amount when a deal is marked lost.

The module is pure pandas — it touches neither disk nor pyarrow — so it is
testable on hand-built frames independent of the :class:`SnapshotStore`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from ..core.exceptions import DataContractError

__all__ = ["pipeline_waterfall", "waterfall_summary"]

#: Canonical display order for the waterfall categories (positive movements,
#: then status-only zero moves, then the negative reductions).
_CATEGORY_ORDER = (
    "created",
    "expanded",
    "advanced",
    "regressed",
    "pushed",
    "pulled_in",
    "unchanged",
    "contracted",
    "won",
    "lost",
    "removed",
)


def _require_deals(
    frame: pd.DataFrame,
    name: str,
    id_col: str,
    amount_col: str,
    stage_col: str,
    close_col: str | None,
    closed: set[Any],
) -> None:
    """Validate an opportunity snapshot, raising :class:`DataContractError`.

    Requires ``frame`` to be a DataFrame carrying the ``id_col``, ``amount_col``
    and ``stage_col`` columns (plus ``close_col`` when supplied), a numeric
    amount column that is non-null on every OPEN deal (stage not in ``closed``),
    and ``opp_id`` values unique within the snapshot. Empty frames are allowed
    (a first-period ``before`` may hold no deals).

    Null amounts on open deals are rejected rather than imputed: the bridge is a
    reconciliation, and a missing amount cannot be netted against its
    counterpart without silently breaking the identity it exists to prove. A
    null on a deal already CLOSED in this snapshot is left alone — the sign
    convention never reads it (won/lost/removed contribute the deal's BEFORE
    value, and only when it was open before, while the opening/closing anchors
    sum open stages only), so rejecting it would refuse a bridge that is exactly
    computable.
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataContractError(
            f"{name} opportunity snapshot must be a pandas DataFrame, got "
            f"{type(frame).__name__}"
        )
    required = [id_col, amount_col, stage_col]
    if close_col is not None:
        required.append(close_col)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataContractError(
            f"{name} opportunity snapshot is missing column(s) {missing}; the "
            f"deal contract is (opp_id, amount, stage) with an optional close date"
        )
    try:
        amounts = pd.to_numeric(frame[amount_col])
    except (ValueError, TypeError) as exc:
        raise DataContractError(
            f"{name} opportunity snapshot column {amount_col!r} must be numeric: {exc}"
        ) from exc
    # only an OPEN deal's amount feeds the bridge; a null on a deal already
    # closed in this snapshot is never read, so it is not an error here
    null_amount = amounts.isna() & ~frame[stage_col].isin(closed)
    if null_amount.any():
        first = frame.loc[null_amount, id_col].iloc[0]
        raise DataContractError(
            f"{name} opportunity snapshot has a null amount for {id_col} {first!r} "
            f"({int(null_amount.sum())} of {len(frame)} rows); that deal is open in "
            f"this snapshot, so its amount is part of the open-pipeline total and "
            f"the waterfall cannot reconcile a missing one — fill or drop those "
            f"deals before bridging"
        )
    dup = frame[id_col].duplicated()
    if dup.any():
        first = frame.loc[dup, id_col].iloc[0]
        raise DataContractError(
            f"{name} opportunity snapshot has duplicate {id_col} {first!r}; "
            f"deal ids must be unique within a snapshot"
        )


def _index_deals(
    frame: pd.DataFrame,
    id_col: str,
    amount_col: str,
    stage_col: str,
    close_col: str | None,
) -> dict[Any, tuple[float, Any, pd.Timestamp | None]]:
    """Index a snapshot by ``opp_id`` into ``(amount, stage, close_date)`` tuples."""
    ids = frame[id_col].to_numpy()
    amounts = pd.to_numeric(frame[amount_col]).to_numpy(dtype=float)
    stages = frame[stage_col].to_numpy()
    if close_col is not None:
        parsed = pd.to_datetime(frame[close_col], errors="coerce")
        closes = [c if pd.notna(c) else None for c in parsed]
    else:
        closes = [None] * len(frame)
    out: dict[Any, tuple[float, Any, pd.Timestamp | None]] = {}
    for oid, amt, stage, close in zip(ids, amounts, stages, closes):
        out[oid] = (float(amt), stage, None if close is None else pd.Timestamp(close))
    return out


def _classify(
    oid: Any,
    before: dict[Any, tuple[float, Any, pd.Timestamp | None]],
    after: dict[Any, tuple[float, Any, pd.Timestamp | None]],
    stage_index: dict[Any, int],
    closed: set[Any],
    won_stage: Any,
    lost_stage: Any,
    use_close: bool,
) -> tuple[str, float]:
    """Classify one ``opp_id`` into ``(category, signed_open_pipeline_amount)``."""
    in_before = oid in before
    in_after = oid in after
    amt_b, st_b, cl_b = before.get(oid, (0.0, None, None))
    open_b = in_before and st_b not in closed

    if not in_after:  # present in before, gone from after
        return "removed", -(amt_b if open_b else 0.0)

    amt_a, st_a, cl_a = after[oid]
    if st_a in closed:  # closed in the after snapshot
        category = "won" if st_a == won_stage else "lost"
        return category, -(amt_b if open_b else 0.0)

    # open in after
    if not open_b:  # new deal, or one that re-entered the open pipeline
        return "created", amt_a

    delta = amt_a - amt_b
    if delta > 0:
        return "expanded", delta
    if delta < 0:
        return "contracted", delta

    si_a = stage_index.get(st_a)
    si_b = stage_index.get(st_b)
    if si_a is not None and si_b is not None:
        if si_a > si_b:
            return "advanced", 0.0
        if si_a < si_b:
            return "regressed", 0.0
    if use_close and cl_a is not None and cl_b is not None:
        if cl_a > cl_b:
            return "pushed", 0.0
        if cl_a < cl_b:
            return "pulled_in", 0.0
    return "unchanged", 0.0


def pipeline_waterfall(
    before: pd.DataFrame,
    after: pd.DataFrame,
    stages: list[Any],
    id_col: str = "opp_id",
    amount_col: str = "amount",
    stage_col: str = "stage",
    close_col: str | None = None,
    won_stage: Any = None,
    lost_stage: Any = None,
) -> pd.DataFrame:
    """Categorize every deal across two opportunity snapshots into a bridge.

    ``before`` and ``after`` are deal-grain opportunity tables (one row per
    ``opp_id``, with ``amount`` and ``stage`` columns and an optional
    ``close_col`` close date); their ``amount`` columns must be numeric, and
    non-null on every deal whose stage is open (a null on a deal already closed
    in that snapshot is never read, so it is allowed). ``stages`` is the ordered
    list of OPEN funnel
    stages (index = pipeline progression); ``won_stage``/``lost_stage`` are the
    closed-stage names — a deal whose ``after`` stage equals one of them is
    categorized won/lost. Neither may appear inside ``stages``.

    Every deal in the union of ``opp_id`` values is placed in exactly one
    category (see the module docstring for the priority rules and the sign
    convention). Returns ``[category, n_deals, amount]`` with one row per
    NON-EMPTY category, in the canonical bridge order, where ``amount`` is the
    signed contribution to the open-pipeline total. The categories partition
    the deal universe, so ``n_deals`` sums to the number of distinct deals and
    ``amount`` sums to ``closing_open_pipeline - opening_open_pipeline``.
    """
    closed = {s for s in (won_stage, lost_stage) if s is not None}
    _require_deals(before, "before", id_col, amount_col, stage_col, close_col, closed)
    _require_deals(after, "after", id_col, amount_col, stage_col, close_col, closed)
    if stages is None or len(stages) == 0:
        raise DataContractError(
            "stages must be a non-empty ordered sequence of open-stage names"
        )
    stages = list(stages)
    overlap = [s for s in stages if s in closed]
    if overlap:
        raise DataContractError(
            f"won_stage/lost_stage {sorted(closed)} must not appear in the open "
            f"'stages' list; found {overlap}"
        )
    stage_index = {s: i for i, s in enumerate(stages)}

    b = _index_deals(before, id_col, amount_col, stage_col, close_col)
    a = _index_deals(after, id_col, amount_col, stage_col, close_col)
    use_close = close_col is not None

    counts: dict[str, int] = defaultdict(int)
    amounts: dict[str, float] = defaultdict(float)
    for oid in {*b, *a}:
        category, amount = _classify(
            oid, b, a, stage_index, closed, won_stage, lost_stage, use_close
        )
        counts[category] += 1
        amounts[category] += amount

    rows = [
        {"category": cat, "n_deals": int(counts[cat]), "amount": float(amounts[cat])}
        for cat in _CATEGORY_ORDER
        if counts[cat]
    ]
    return pd.DataFrame(rows, columns=["category", "n_deals", "amount"])


def _open_total(
    frame: pd.DataFrame, amount_col: str, stage_col: str, closed: set[Any]
) -> tuple[float, int]:
    """The open-pipeline dollar total and open-deal count of a snapshot."""
    if len(frame) == 0:
        return 0.0, 0
    is_open = ~frame[stage_col].isin(closed)
    total = float(pd.to_numeric(frame.loc[is_open, amount_col]).sum())
    return total, int(is_open.sum())


def waterfall_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
    stages: list[Any],
    id_col: str = "opp_id",
    amount_col: str = "amount",
    stage_col: str = "stage",
    close_col: str | None = None,
    won_stage: Any = None,
    lost_stage: Any = None,
) -> pd.DataFrame:
    """Wrap :func:`pipeline_waterfall` as a render-ready bridge with anchors.

    Returns ``[category, n_deals, amount, running_total]`` framed by an
    ``opening_pipeline`` row (the open-pipeline total of ``before``) and a
    closing ``closing_pipeline`` row (that of ``after``), with each movement
    category in between accumulating ``running_total``. Because the categories
    reconcile the bridge exactly, the running total after the final movement
    equals the ``closing_pipeline`` amount — feed the rows straight into a
    classic waterfall chart.
    """
    wf = pipeline_waterfall(
        before,
        after,
        stages,
        id_col=id_col,
        amount_col=amount_col,
        stage_col=stage_col,
        close_col=close_col,
        won_stage=won_stage,
        lost_stage=lost_stage,
    )
    closed = {s for s in (won_stage, lost_stage) if s is not None}
    open_before, n_before = _open_total(before, amount_col, stage_col, closed)
    open_after, n_after = _open_total(after, amount_col, stage_col, closed)

    rows = [
        {
            "category": "opening_pipeline",
            "n_deals": n_before,
            "amount": open_before,
            "running_total": open_before,
        }
    ]
    running = open_before
    for r in wf.itertuples(index=False):
        running += float(r.amount)
        rows.append(
            {
                "category": r.category,
                "n_deals": int(r.n_deals),
                "amount": float(r.amount),
                "running_total": running,
            }
        )
    rows.append(
        {
            "category": "closing_pipeline",
            "n_deals": n_after,
            "amount": open_after,
            "running_total": open_after,
        }
    )
    return pd.DataFrame(
        rows, columns=["category", "n_deals", "amount", "running_total"]
    )
