"""The forecast-os MCP server: the engine as a set of agent-callable tools.

Point any MCP client (Claude Desktop, Claude Code, ...) at the
``forecast-os-mcp`` console script and it gets six tools: discover models
and schema mappings, preview how raw records shape into the
``(unique_id, ds, y)`` panel contract, then forecast, compare models, or
score quota attainment.

The module separates pure logic from transport: the ``*_tool`` functions
are plain Python — JSON-able arguments in, JSON-able rows out — so they are
testable and reusable without a running server; :func:`main` wires them
into a FastMCP stdio server. The ``mcp`` dependency is optional: importing
this module always succeeds, and :func:`main` raises an ``ImportError``
with an install hint when the extra is missing.
"""

from __future__ import annotations

import functools
import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..connectors.base import apply_mapping, list_mappings
from ..core.exceptions import ForecastOSError
from ..core.registry import get_model, list_models
from ..core.types import ID_COL, TIME_COL, validate_panel
from ..engine import ForecastEngine
from ..gtm.quota import attainment_probability

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:  # tests exercise this path by monkeypatching the flag
    FastMCP = None
    _HAS_MCP = False

__all__ = [
    "list_models_tool",
    "list_mappings_tool",
    "preview_panel",
    "forecast_tool",
    "compare_tool",
    "quota_tool",
    "main",
]

_MCP_HINT = (
    "the MCP server needs the optional 'mcp' dependency; "
    'install it with: pip install "forecast-os[mcp]"'
)

#: Default leaderboard pool: a baseline plus two strong statistical models.
_DEFAULT_COMPARE_MODELS = ("naive", "theta", "auto_ets")


def _tool_errors(fn):
    """Re-raise engine errors as plain ``ValueError`` — what MCP clients surface."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ForecastOSError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

    return wrapper


def _not_a_boolean(value: Any, name: str) -> Any:
    """Return ``value`` unless it is a boolean, which is never a number here.

    ``bool`` is an ``int`` subclass, so ``int(True) == 1``: unguarded, ``h=true``
    is a silent one-step forecast, ``seasonality=true`` swaps the seasonal-naive
    MASE denominator for a naive one, ``level=true`` reports a 1% interval, and
    ``quota=true`` scores against a target of 1.0. Every other coercion these
    tools accept (``"2"``, ``2.0``) is kept — only the boolean one is nonsense.

    Note which surfaces this actually covers. It runs in the tool body, so it
    protects direct Python callers; the REST layer gets the same protection
    earlier, from ``serve.app._StrictInt``. It does **not** protect the MCP
    wire: FastMCP validates against the annotation and coerces a JSON boolean
    to ``int`` before the body runs, so an MCP client sending ``h=true`` still
    gets the one-step forecast. Closing that would mean annotating the tool
    parameters themselves, not checking inside.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    return value


def _a_number(value: Any, name: str) -> Any:
    """``_not_a_boolean`` plus a real-number check, for values passed on untouched.

    ``level`` reaches ``predict`` uncoerced so the model layer's own whole-number
    check stays visible on this surface. That left a string level to fail as a raw
    ``TypeError`` from a comparison deep inside ``_check_level`` — which escapes
    the ``ValueError``-only error decorator and surfaces as a 500 traceback rather
    than the documented 400.
    """
    _not_a_boolean(value, name)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    return value


def _records_out(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as plain JSON-able row dicts (datetimes become ISO strings)."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _load_records(
    csv_path: str | None = None, records: list[dict[str, Any]] | None = None
) -> pd.DataFrame:
    """Raw records from exactly one of a CSV path or an inline list of row dicts."""
    if (csv_path is None) == (records is None):
        raise ValueError(
            "provide exactly one of csv_path (a CSV file of records) or "
            "records (an inline list of row dicts)"
        )
    if csv_path is not None:
        try:
            return pd.read_csv(csv_path)
        except OSError as exc:
            raise ValueError(f"cannot read csv_path {csv_path!r}: {exc}") from exc
    return pd.DataFrame(records)


def _build_panel(
    csv_path: str | None,
    records: list[dict[str, Any]] | None,
    mapping: str | None,
    **overrides,
) -> pd.DataFrame:
    """Shape raw records into a validated ``(unique_id, ds, y)`` panel.

    With ``mapping`` the records go through the named (or given) recipe;
    without one they must already carry the contract columns, and ``ds`` is
    parsed to datetimes when needed.

    Shaping runs on caller-supplied values, so the arithmetic and type errors
    it raises are almost always client errors, not server faults: they are
    re-raised as ``ValueError`` (which the REST layer renders as a 400) rather
    than escaping as e.g. ``ZeroDivisionError`` from ``freq="0D"`` or
    ``TypeError`` from a series id that is a nested object.

    The catch selects on exception type, not on origin, so it is broader than
    those cases: a ``TypeError`` from a genuine bug inside ``apply_mapping``,
    ``to_panel``, or ``validate_panel`` is relabelled the same way and reported
    to the caller as a 400 blaming their records, instead of surfacing as the
    500 an internal fault deserves.
    """
    frame = _load_records(csv_path=csv_path, records=records)
    try:
        if mapping is not None:
            return apply_mapping(frame, mapping, **overrides)
        if overrides:
            raise ValueError(
                f"panel override(s) {sorted(overrides)} need a mapping; records "
                f"passed without one must already be (unique_id, ds, y) rows"
            )
        if TIME_COL in frame.columns and not (
            pd.api.types.is_numeric_dtype(frame[TIME_COL])
            or pd.api.types.is_datetime64_any_dtype(frame[TIME_COL])
        ):
            frame = frame.assign(**{TIME_COL: pd.to_datetime(frame[TIME_COL])})
        return validate_panel(frame)
    except (ArithmeticError, TypeError) as exc:
        raise ValueError(f"cannot shape these records into a panel: {exc}") from exc


def _panel_overrides(freq: str | None, agg: str | None) -> dict[str, str]:
    """The panel-shaping overrides actually set (None means "use the mapping's")."""
    return {k: v for k, v in (("freq", freq), ("agg", agg)) if v is not None}


def _compare_failures(
    requested: list[Any], board: pd.DataFrame, caught: list[warnings.WarningMessage]
) -> list[str]:
    """One message per requested model that is missing from the leaderboard.

    The board is the source of truth: a requested model that did not survive
    the backtest is a failure, whether or not its warning was captured.
    :meth:`ForecastEngine.compare` explains each drop in a
    ``"model X failed: ..."`` warning, which is used here for its reason text
    only — reading the warning list *as* the failure list was wrong twice
    over. Warning capture mutates process-global state, so an overlapping
    request's ``catch_warnings`` block could swallow the explanation and leave
    a model silently absent; and every other warning raised during the
    backtest (numpy overflow, deprecations) was reported as a model failure
    even when the whole field ran.
    """
    survived = set(board.index)
    messages = [str(w.message) for w in caught]
    failures: list[str] = []
    seen: set[str] = set()
    for name in requested:
        # Only registry names map onto the board index. ForecastEngine also
        # accepts model instances and (name, params) specs; those are outside
        # this tool's documented list[str] contract and cannot be matched to a
        # board row, so they are skipped rather than reported as failures.
        if not isinstance(name, str) or name in survived or name in seen:
            continue
        seen.add(name)
        prefix = f"model {name} failed: "
        explained = next((m for m in messages if m.startswith(prefix)), None)
        failures.append(explained or f"model {name} failed: reason unavailable")
    return failures


# -- tool functions (pure logic; registered with FastMCP in main) --------------


def list_models_tool() -> list[dict[str, Any]]:
    """List every forecasting model this engine can run.

    Call this first when choosing a ``model`` for the forecast/compare/quota
    tools. Returns one entry per registered model: ``name`` (the value to
    pass as ``model``), ``family`` (baseline, statistical, ml, ensemble,
    financial, ...), and a one-line ``description``. Good defaults:
    ``auto_select`` picks the best model per series, ``theta`` and
    ``auto_ets`` are strong statistical choices, and ``reconciled``
    forecasts hierarchies (path-style ids like ``"west/alice"``) coherently.
    """
    return _records_out(list_models()[["name", "family", "description"]])


def list_mappings_tool() -> list[dict[str, Any]]:
    """List the registered schema mappings (platform-export-to-panel recipes).

    A mapping shapes raw platform records (CRM deals, invoices, product
    events) into the ``(unique_id, ds, y)`` panel contract the engine
    speaks. Call this to find a ``mapping`` name for the preview, forecast,
    compare, and quota tools whenever the rows are raw exports rather than
    an already-shaped panel. Returns each recipe's ``name``,
    ``description``, and default ``freq``/``agg``.
    """
    return _records_out(list_mappings())


@_tool_errors
def preview_panel(
    csv_path: str | None = None,
    records: list[dict[str, Any]] | None = None,
    mapping: str | None = None,
    **overrides,
) -> dict[str, Any]:
    """Preview how records shape into the ``(unique_id, ds, y)`` forecast panel.

    Call this before forecasting new data — it is a cheap dry run showing
    exactly what the engine will see. Pass exactly one of ``csv_path`` (path
    to a CSV of records) or ``records`` (inline list of row dicts). When the
    rows are raw platform records, pass ``mapping`` (a name from
    ``list_mappings``) plus any recipe overrides (e.g. ``freq="W"``,
    ``agg="mean"``); when the rows already carry ``unique_id`` (series id),
    ``ds`` (date), and ``y`` (numeric value), omit ``mapping``. Returns
    ``rows`` (panel length), ``series`` (the distinct series ids), and
    ``head`` (the first 10 panel rows). If the shape looks wrong, fix the
    mapping or the data before forecasting.
    """
    panel = _build_panel(csv_path, records, mapping, **overrides)
    return {
        "rows": len(panel),
        "series": panel[ID_COL].drop_duplicates().tolist(),
        "head": _records_out(panel.head(10)),
    }


@_tool_errors
def forecast_tool(
    csv_path: str | None = None,
    records: list[dict[str, Any]] | None = None,
    mapping: str | None = None,
    model: str = "auto_select",
    model_params: dict[str, Any] | None = None,
    h: int = 12,
    level: list[int] | None = None,
    freq: str | None = None,
    agg: str | None = None,
) -> list[dict[str, Any]]:
    """Forecast future values for one or more time series.

    Use this after ``preview_panel`` confirms the data shapes correctly.
    Data input works the same everywhere: exactly one of ``csv_path`` or
    ``records``, plus ``mapping`` (a name from ``list_mappings``) when the
    rows are raw platform records — omit ``mapping`` when they already form
    the ``(unique_id, ds, y)`` panel contract. ``freq`` (pandas alias, e.g.
    "MS", "W") and ``agg`` ("sum", "mean", "count") override the mapping's
    defaults exactly as in ``preview_panel``, so a previewed panel and the
    forecast operate on the same shape. ``model`` is any name from
    ``list_models`` (default ``auto_select``, which picks the best model per
    series); ``model_params`` are constructor overrides for it. Returns one
    row per series per future period: ``unique_id``, ``ds``, ``yhat`` (the
    point forecast), and ``lo-L``/``hi-L`` interval bounds for each
    confidence level in ``level`` (default ``[80]``).
    """
    panel = _build_panel(csv_path, records, mapping, **_panel_overrides(freq, agg))
    try:
        forecaster = get_model(model, **(model_params or {}))
    except TypeError as exc:
        raise ValueError(f"invalid model_params for model {model!r}: {exc}") from exc
    # levels reach predict() uncoerced: int() here truncated 99.9 to 99 and
    # served a ~22% narrower interval under a label nobody asked for, hiding
    # the model layer's own whole-number check from this surface entirely.
    levels = [80] if level is None else [_a_number(lvl, "level") for lvl in level]
    try:
        pred = forecaster.fit(panel).predict(int(_not_a_boolean(h, "h")), level=levels or None)
    except (TypeError, AttributeError) as exc:
        # Constructor overrides that survive get_model() can still be the wrong
        # TYPE and only blow up deep inside fit()/predict() (season_length="x",
        # lags=true, models=[1]). Those are caller mistakes, and the documented
        # contract for this surface is a 400 with a message — never a 500 with a
        # traceback. With no model_params there is nothing caller-supplied to
        # blame, so a genuine internal fault still propagates untouched.
        if not model_params:
            raise
        raise ValueError(f"invalid model_params for model {model!r}: {exc}") from exc
    return _records_out(pred)


@_tool_errors
def compare_tool(
    csv_path: str | None = None,
    records: list[dict[str, Any]] | None = None,
    mapping: str | None = None,
    models: list[str] | None = None,
    h: int = 12,
    n_windows: int = 3,
    metrics: list[str] | None = None,
    level: list[int] | None = None,
    seasonality: int = 1,
    freq: str | None = None,
    agg: str | None = None,
) -> dict[str, Any]:
    """Backtest several models on the data and rank them on a leaderboard.

    Use this when unsure which model to trust: each candidate is
    walk-forward cross-validated (``n_windows`` folds of horizon ``h``) and
    scored on held-out data. Data input works like ``forecast`` (one of
    ``csv_path``/``records``, optional ``mapping``, optional ``freq``/``agg``
    panel-shaping overrides). ``models`` is a list of names from
    ``list_models`` (default: naive, theta, auto_ets). The default metric
    ``mase`` is scale-free — below 1 beats a naive forecaster; pass
    ``seasonality`` (e.g. 12 for monthly data) to measure against a
    seasonal-naive yardstick instead. Interval metrics such as
    ``coverage``/``winkler`` also need ``level``. Returns ``leaderboard``
    (one row per surviving model, best first — feed the winner's ``model``
    name back into ``forecast``) and ``failures`` (one message per requested
    model that failed backtesting and is therefore absent from the
    leaderboard; empty when all models ran).
    """
    panel = _build_panel(csv_path, records, mapping, **_panel_overrides(freq, agg))
    season = int(_not_a_boolean(seasonality, "seasonality"))
    if season < 1:
        raise ValueError(
            f"seasonality must be a positive integer (the seasonal period, "
            f"e.g. 12 for monthly data), got {season}"
        )
    requested = list(models) if models else list(_DEFAULT_COMPARE_MODELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        board = ForecastEngine().compare(
            panel,
            h=int(_not_a_boolean(h, "h")),
            n_windows=int(_not_a_boolean(n_windows, "n_windows")),
            metrics=list(metrics) if metrics else ["mase"],
            seasonality=season,
            models=requested,
            level=None if level is None else [_not_a_boolean(lvl, "level") for lvl in level],
        )
    return {
        "leaderboard": _records_out(board.reset_index()),
        "failures": _compare_failures(requested, board, caught),
    }


@_tool_errors
def quota_tool(
    csv_path: str | None = None,
    records: list[dict[str, Any]] | None = None,
    mapping: str | None = None,
    model: str = "reconciled",
    h: int = 12,
    *,
    quota: float | dict[str, float],
    level: int = 80,
    freq: str | None = None,
    agg: str | None = None,
) -> dict[str, Any]:
    """Probability that each series' forecast total meets its quota/target.

    Use this for questions like "will we hit the number this quarter?".
    Data input works like ``forecast`` (one of ``csv_path``/``records``,
    optional ``mapping``, optional ``freq``/``agg`` panel-shaping
    overrides). Forecasts ``h`` periods ahead with ``model`` — the default
    ``reconciled`` rolls up path-style ids (``"west/alice"``) coherently and
    always adds a grand ``total`` series — then converts the ``level``%
    prediction interval into the probability that each series' summed
    horizon meets ``quota``. ``quota`` is either a single number applied to
    every series, or a ``{unique_id: quota}`` object, in which case only the
    named series are scored (use ``{"total": N}`` for the company-level
    number). Returns ``rows`` (one per scored series: ``expected`` — the
    forecast total — ``quota``, and ``p_attain`` in [0, 1]) and
    ``unmatched_quota_keys`` (quota keys naming no forecast series, e.g.
    typos; empty when every key matched). Errors only when NO quota key
    matches any series or a quota is not a number (a non-numeric value, or
    NaN). Infinities are numbers and are scored, not rejected: ``+inf``
    gives ``p_attain`` 0.0 and ``-inf`` gives 1.0, though the echoed
    ``quota`` field comes back as JSON ``null`` (infinity has no JSON
    spelling).
    """
    raw_targets = list(quota.values()) if isinstance(quota, dict) else [quota]
    for target in raw_targets:
        _not_a_boolean(target, "quota")
    try:
        # coerce BEFORE testing: math.isnan() on the caller's raw value raised
        # an uncaught TypeError for quota="100", a string every other layer
        # (attainment_probability, pydantic) accepts and floats happily.
        targets = [float(target) for target in raw_targets]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"quota must be a number, got {quota!r}") from exc
    if any(math.isnan(target) for target in targets):
        # NaN scores p_attain as NaN -- serialised as JSON null, which is not
        # the probability in [0, 1] this tool promises. An infinite quota is
        # answerable (0.0 / 1.0), so it is left alone.
        raise ValueError(f"quota must be a number, got {quota!r}")
    panel = _build_panel(csv_path, records, mapping, **_panel_overrides(freq, agg))
    # level reaches predict() uncoerced so a fractional level is rejected there
    # rather than truncated here; the interval column labels then follow the
    # same rounding predict() used to name them. Unlike forecast_tool this
    # keeps the boolean-only guard, so a non-numeric level (level="80") is not
    # rejected here and instead fails inside _check_level's comparison as a
    # raw TypeError, which _tool_errors does not convert to a ValueError. Both
    # wire surfaces declare level as an int and coerce "80" before the call, so
    # that gap is reachable only by calling this function directly in Python.
    _not_a_boolean(level, "level")
    pred = get_model(model).fit(panel).predict(int(_not_a_boolean(h, "h")), level=[level])
    label = int(round(float(level)))
    unmatched: list[str] = []
    if isinstance(quota, dict):
        scored = pred[pred[ID_COL].isin(quota)]
        if len(scored) == 0:
            raise ValueError(
                f"quota keys {sorted(quota)} match no forecast series; "
                f"forecast has {sorted(pred[ID_COL].unique())}"
            )
        unmatched = sorted(set(quota) - set(scored[ID_COL]))
        pred = scored
    return {
        "rows": _records_out(attainment_probability(pred, quota, level=label)),
        "unmatched_quota_keys": unmatched,
    }


# -- FastMCP wiring ------------------------------------------------------------


def _preview_panel_mcp(
    csv_path: str | None = None,
    records: list[dict[str, Any]] | None = None,
    mapping: str | None = None,
    freq: str | None = None,
    agg: str | None = None,
) -> dict[str, Any]:
    """Preview how records shape into the ``(unique_id, ds, y)`` forecast panel.

    Call this before forecasting new data — it is a cheap dry run showing
    exactly what the engine will see. Pass exactly one of ``csv_path`` (path
    to a CSV of records) or ``records`` (inline list of row dicts). When the
    rows are raw platform records, pass ``mapping`` (a name from
    ``list_mappings``) and optionally override its ``freq`` (pandas alias,
    e.g. "MS", "W") or ``agg`` ("sum", "mean", "count"); when the rows
    already carry ``unique_id`` (series id), ``ds`` (date), and ``y``
    (numeric value), omit ``mapping``. Returns ``rows`` (panel length),
    ``series`` (the distinct series ids), and ``head`` (the first 10 panel
    rows). If the shape looks wrong, fix the mapping or the data before
    forecasting.
    """
    return preview_panel(
        csv_path=csv_path, records=records, mapping=mapping, **_panel_overrides(freq, agg)
    )


def _build_server():
    """Construct the FastMCP server with the six tools registered."""
    if not _HAS_MCP:
        raise ImportError(_MCP_HINT)
    server = FastMCP("forecast-os")
    server.add_tool(list_models_tool, name="list_models")
    server.add_tool(list_mappings_tool, name="list_mappings")
    server.add_tool(_preview_panel_mcp, name="preview_panel")
    server.add_tool(forecast_tool, name="forecast")
    server.add_tool(compare_tool, name="compare")
    server.add_tool(quota_tool, name="quota")
    return server


def main() -> None:
    """Console entry point: run the forecast-os MCP server over stdio.

    Raises ``ImportError`` with the ``forecast-os[mcp]`` install hint when
    the optional ``mcp`` dependency is missing.
    """
    _build_server().run(transport="stdio")
