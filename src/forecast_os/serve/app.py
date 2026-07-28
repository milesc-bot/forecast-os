"""The forecast-os REST server: ``forecast-os-serve``.

A thin JSON HTTP surface over the engine, built on the optional ``fastapi``
and ``uvicorn`` dependencies (``pip install "forecast-os[serve]"``). Every
route reuses a pure MCP tool function verbatim — the serving layer only parses
the request body, calls the tool, and returns its JSON-able result — so the
HTTP and MCP surfaces stay behaviourally identical.

Routes:

- ``GET  /health``   -> ``{"status": "ok", "version": ...}``
- ``GET  /models``   -> the model catalogue (:func:`list_models_tool`)
- ``GET  /mappings`` -> the schema-mapping catalogue (:func:`list_mappings_tool`)
- ``POST /preview``  -> panel preview (:func:`preview_panel`)
- ``POST /forecast`` -> ``{"forecast": [...]}`` (:func:`forecast_tool`)
- ``POST /compare``  -> a backtest leaderboard (:func:`compare_tool`)
- ``POST /quota``    -> quota-attainment probabilities (:func:`quota_tool`)

Engine errors (bad data, unknown models, contract violations) surface as HTTP
400 with a ``{"error": message}`` body — never a 500 traceback — via a
registered exception handler.

The ``fastapi``/``uvicorn`` dependencies are optional: importing this module
always succeeds, and :func:`create_app` / :func:`main` raise ``ImportError``
with the ``forecast-os[serve]`` install hint when the extra is missing.
"""

from __future__ import annotations

import argparse
from typing import Annotated, Any

import forecast_os

from ..core.exceptions import ForecastOSError
from ..mcp.server import (
    compare_tool,
    forecast_tool,
    list_mappings_tool,
    list_models_tool,
    preview_panel,
    quota_tool,
)

_SERVE_HINT = (
    "the REST server needs the optional fastapi and uvicorn dependencies; "
    'install them with: pip install "forecast-os[serve]"'
)

try:
    from fastapi import FastAPI
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, BeforeValidator, ConfigDict

    _HAS_FASTAPI = True
except ImportError:  # tests exercise this path by monkeypatching the flag
    _HAS_FASTAPI = False

try:
    import uvicorn

    _HAS_UVICORN = True
except ImportError:  # tests exercise this path by monkeypatching the flag
    _HAS_UVICORN = False

__all__ = ["create_app", "main"]


def _preview_overrides(freq: str | None, agg: str | None) -> dict[str, str]:
    """The panel-shaping overrides actually set (None means "use the mapping's").

    ``preview_panel`` forwards ``**overrides`` straight into the mapping, so a
    ``None`` would blank the mapping's own ``freq``/``agg`` default; only the
    values the caller actually provided are passed through.
    """
    return {k: v for k, v in (("freq", freq), ("agg", agg)) if v is not None}


if _HAS_FASTAPI:

    def _not_a_boolean(value: Any) -> Any:
        """Reject a JSON boolean where a number belongs.

        ``bool`` is an ``int`` subclass, so pydantic's lax mode happily reads
        ``"h": true`` as ``h=1`` and returns a 200 with one forecast row —
        silently answering a question the caller did not ask. The same hole
        is worse on the siblings: ``"level": true`` reports attainment
        against a 1% interval instead of 80%, ``"seasonality": true`` swaps
        the seasonal-naive MASE denominator for a naive one, and
        ``"quota": true`` scores against a target of 1.0. Other lax
        coercions (``"2"``, ``2.0``) stay: only the boolean one is nonsense.
        """
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value

    #: An integer field that does not quietly accept ``true``/``false``.
    _StrictInt = Annotated[int, BeforeValidator(_not_a_boolean)]

    #: A float field that does not quietly accept ``true``/``false``.
    _StrictFloat = Annotated[float, BeforeValidator(_not_a_boolean)]

    class PreviewRequest(BaseModel):
        """Body for ``POST /preview`` — mirrors :func:`preview_panel`.

        Rows are supplied inline via ``records``. The underlying
        :func:`preview_panel` also accepts a server-side ``csv_path``, but that
        parameter is deliberately **not** exposed over HTTP: it was handed
        straight to ``pandas.read_csv``, which accepts URLs as readily as paths,
        making an unauthenticated request able to read any file the server
        process can reach, drive outbound requests to arbitrary hosts with the
        response reflected back to the caller, and allocate unbounded memory
        decompressing a remote archive before any validation ran. The MCP
        surface, which is local and trusted, keeps the affordance.
        """

        records: list[dict[str, Any]] | None = None
        mapping: str | None = None
        freq: str | None = None
        agg: str | None = None

    class ForecastRequest(BaseModel):
        """Body for ``POST /forecast`` — mirrors :func:`forecast_tool`.

        ``model_params`` is free-form constructor passthrough: it is forwarded
        to the chosen model's constructor as-is, and the strict-number guards
        on ``h``/``level`` do not reach inside it. Whatever the constructor
        does not check for itself is accepted, so with the default
        ``auto_select`` (and with ``auto_ets`` or ``theta``)
        ``{"season_length": true}`` is read as ``1`` and
        ``{"season_length": -3}`` is taken at face value — both answer with a
        200. A model that validates the value itself still rejects it:
        ``seasonal_naive`` 400s on ``-3``, ``holt_winters`` on both.

        Params that are the wrong *type* still fail as
        a 400 rather than a 500, because :func:`forecast_tool` blames
        ``model_params`` for a ``TypeError``/``AttributeError`` out of
        ``get_model``/``fit``/``predict`` whenever any were supplied.
        """

        model_config = ConfigDict(protected_namespaces=())

        records: list[dict[str, Any]]
        mapping: str | None = None
        model: str = "auto_select"
        model_params: dict[str, Any] | None = None
        h: _StrictInt = 12
        level: list[_StrictInt] | None = None
        freq: str | None = None
        agg: str | None = None

    class CompareRequest(BaseModel):
        """Body for ``POST /compare`` — mirrors :func:`compare_tool`."""

        records: list[dict[str, Any]]
        mapping: str | None = None
        models: list[str] | None = None
        h: _StrictInt = 12
        n_windows: _StrictInt = 3
        metrics: list[str] | None = None
        level: list[_StrictInt] | None = None
        seasonality: _StrictInt = 1
        freq: str | None = None
        agg: str | None = None

    class QuotaRequest(BaseModel):
        """Body for ``POST /quota`` — mirrors :func:`quota_tool`."""

        model_config = ConfigDict(protected_namespaces=())

        records: list[dict[str, Any]]
        mapping: str | None = None
        model: str = "reconciled"
        h: _StrictInt = 12
        quota: _StrictFloat | dict[str, _StrictFloat]
        level: _StrictInt = 80
        freq: str | None = None
        agg: str | None = None


def create_app() -> FastAPI:
    """Build the forecast-os FastAPI application.

    Every route reuses a pure MCP tool function; engine errors
    (:class:`ValueError`/:class:`ForecastOSError`) are turned into HTTP 400
    ``{"error": message}`` responses by a registered handler. Raises
    ``ImportError`` with the ``forecast-os[serve]`` install hint when the
    optional ``fastapi`` dependency is missing.
    """
    if not _HAS_FASTAPI:
        raise ImportError(_SERVE_HINT)

    app = FastAPI(title="forecast-os", version=forecast_os.__version__)

    async def _bad_request(request, exc):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    async def _invalid_body(request, exc):
        # A malformed request body is still a client error; return the same
        # {"error": ...} envelope (not FastAPI's default 422 {"detail": ...}).
        detail = jsonable_encoder(exc.errors()) if hasattr(exc, "errors") else str(exc)
        return JSONResponse(
            status_code=400,
            content={"error": "invalid request body", "detail": detail},
        )

    # Engine errors are user errors (bad data/model/quota), not server faults:
    # surface them as 400 with a clean message instead of a 500 traceback.
    app.add_exception_handler(ValueError, _bad_request)
    app.add_exception_handler(ForecastOSError, _bad_request)
    app.add_exception_handler(RequestValidationError, _invalid_body)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": forecast_os.__version__}

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        return list_models_tool()

    @app.get("/mappings")
    def mappings() -> list[dict[str, Any]]:
        return list_mappings_tool()

    @app.post("/preview")
    def preview(body: PreviewRequest) -> dict[str, Any]:
        return preview_panel(
            records=body.records,
            mapping=body.mapping,
            **_preview_overrides(body.freq, body.agg),
        )

    @app.post("/forecast")
    def forecast(body: ForecastRequest) -> dict[str, Any]:
        return {
            "forecast": forecast_tool(
                records=body.records,
                mapping=body.mapping,
                model=body.model,
                model_params=body.model_params,
                h=body.h,
                level=body.level,
                freq=body.freq,
                agg=body.agg,
            )
        }

    @app.post("/compare")
    def compare(body: CompareRequest) -> dict[str, Any]:
        return compare_tool(
            records=body.records,
            mapping=body.mapping,
            models=body.models,
            h=body.h,
            n_windows=body.n_windows,
            metrics=body.metrics,
            level=body.level,
            seasonality=body.seasonality,
            freq=body.freq,
            agg=body.agg,
        )

    @app.post("/quota")
    def quota(body: QuotaRequest) -> dict[str, Any]:
        return quota_tool(
            records=body.records,
            mapping=body.mapping,
            model=body.model,
            h=body.h,
            quota=body.quota,
            level=body.level,
            freq=body.freq,
            agg=body.agg,
        )

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-os-serve",
        description="Serve the forecast-os engine over a JSON REST API.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host interface to bind (default 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="port to bind (default 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="auto-reload on code changes (development only)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console entry point for ``forecast-os-serve``.

    Parses ``--host``/``--port``/``--reload`` and runs the app under uvicorn.
    Raises ``ImportError`` with the ``forecast-os[serve]`` install hint when
    the optional ``fastapi``/``uvicorn`` dependencies are missing.
    """
    args = _build_parser().parse_args(argv)
    if not (_HAS_FASTAPI and _HAS_UVICORN):
        raise ImportError(_SERVE_HINT)
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)
