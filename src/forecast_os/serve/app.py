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
from typing import Any

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
    from pydantic import BaseModel, ConfigDict

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

    class PreviewRequest(BaseModel):
        """Body for ``POST /preview`` — mirrors :func:`preview_panel`.

        Provide exactly one of ``records`` (inline row dicts) or ``csv_path``
        (a CSV path read from the **server's** filesystem, not the client's).
        """

        records: list[dict[str, Any]] | None = None
        csv_path: str | None = None
        mapping: str | None = None
        freq: str | None = None
        agg: str | None = None

    class ForecastRequest(BaseModel):
        """Body for ``POST /forecast`` — mirrors :func:`forecast_tool`."""

        model_config = ConfigDict(protected_namespaces=())

        records: list[dict[str, Any]]
        mapping: str | None = None
        model: str = "auto_select"
        model_params: dict[str, Any] | None = None
        h: int = 12
        level: list[int] | None = None
        freq: str | None = None
        agg: str | None = None

    class CompareRequest(BaseModel):
        """Body for ``POST /compare`` — mirrors :func:`compare_tool`."""

        records: list[dict[str, Any]]
        mapping: str | None = None
        models: list[str] | None = None
        h: int = 12
        n_windows: int = 3
        metrics: list[str] | None = None
        level: list[int] | None = None
        seasonality: int = 1
        freq: str | None = None
        agg: str | None = None

    class QuotaRequest(BaseModel):
        """Body for ``POST /quota`` — mirrors :func:`quota_tool`."""

        model_config = ConfigDict(protected_namespaces=())

        records: list[dict[str, Any]]
        mapping: str | None = None
        model: str = "reconciled"
        h: int = 12
        quota: float | dict[str, float]
        level: int = 80
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
            csv_path=body.csv_path,
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
