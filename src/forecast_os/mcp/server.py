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
import warnings
from typing import Any

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
    """
    frame = _load_records(csv_path=csv_path, records=records)
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


def _panel_overrides(freq: str | None, agg: str | None) -> dict[str, str]:
    """The panel-shaping overrides actually set (None means "use the mapping's")."""
    return {k: v for k, v in (("freq", freq), ("agg", agg)) if v is not None}


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
    levels = [80] if level is None else [int(lvl) for lvl in level]
    pred = forecaster.fit(panel).predict(int(h), level=levels or None)
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
    name back into ``forecast``) and ``failures`` (warning messages
    explaining every model that failed backtesting and is therefore absent
    from the leaderboard; empty when all models ran).
    """
    panel = _build_panel(csv_path, records, mapping, **_panel_overrides(freq, agg))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        board = ForecastEngine().compare(
            panel,
            h=int(h),
            n_windows=int(n_windows),
            metrics=list(metrics) if metrics else ["mase"],
            seasonality=int(seasonality),
            models=list(models) if models else list(_DEFAULT_COMPARE_MODELS),
            level=level,
        )
    return {
        "leaderboard": _records_out(board.reset_index()),
        "failures": [str(w.message) for w in caught],
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
    matches any series.
    """
    panel = _build_panel(csv_path, records, mapping, **_panel_overrides(freq, agg))
    pred = get_model(model).fit(panel).predict(int(h), level=[int(level)])
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
        "rows": _records_out(attainment_probability(pred, quota, level=int(level))),
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
