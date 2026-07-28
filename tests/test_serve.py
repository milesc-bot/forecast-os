"""Tests for the REST serving layer (forecast_os.serve.app).

The HTTP surface reuses the MCP pure tool functions verbatim, so these tests
drive the app through ``fastapi.testclient.TestClient`` (in-process, no real
server, no network) and assert the JSON contract of every route. The import
guard and the ``main`` console entry point get their own tests, exercising the
``[serve]`` install hint by monkeypatching the dependency-present flags.
"""

import pandas as pd
import pytest

import forecast_os
from forecast_os.connectors.base import SchemaMapping, register_mapping
from forecast_os.serve import app as serve_app

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


register_mapping(
    SchemaMapping(
        name="serve_test_deals",
        description="test recipe: inline deal records to a monthly per-rep panel",
        date_col="close_date",
        id_cols=("rep",),
        value_col="amount",
        freq="MS",
    )
)


def _deal_records():
    """24 months x 2 reps of deterministic deal rows (one deal per rep-month)."""
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    rows = []
    for i, month in enumerate(months):
        day = (month + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        rows.append(
            {"rep": "alice", "close_date": day, "amount": 100.0 + 5.0 * i + 10.0 * (i % 3)}
        )
        rows.append(
            {"rep": "bob", "close_date": day, "amount": 80.0 + 3.0 * i + 5.0 * (i % 4)}
        )
    return rows


def _panel_records():
    """An already-shaped (unique_id, ds, y) panel as inline row dicts."""
    return [
        {"unique_id": "s", "ds": f"2024-{m:02d}-01", "y": float(m) + (m % 3)}
        for m in range(1, 13)
    ]


@pytest.fixture
def client():
    return TestClient(serve_app.create_app())


class TestHealthAndDiscovery:
    def test_health_reports_version(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == forecast_os.__version__

    def test_models_lists_registered_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) >= 17
        names = {m["name"] for m in models}
        assert {"auto_select", "theta", "reconciled"} <= names
        assert set(models[0]) == {"name", "family", "description"}

    def test_mappings_lists_registered_recipes(self, client):
        resp = client.get("/mappings")
        assert resp.status_code == 200
        mappings = resp.json()
        ours = [m for m in mappings if m["name"] == "serve_test_deals"]
        assert len(ours) == 1
        assert ours[0]["freq"] == "MS"


class TestPreview:
    def test_preview_with_mapping(self, client):
        resp = client.post(
            "/preview", json={"records": _deal_records(), "mapping": "serve_test_deals"}
        )
        assert resp.status_code == 200
        out = resp.json()
        assert out["rows"] == 48
        assert out["series"] == ["alice", "bob"]
        assert len(out["head"]) == 10
        assert set(out["head"][0]) == {"unique_id", "ds", "y"}

    def test_preview_agg_override_forwarded(self, client):
        resp = client.post(
            "/preview",
            json={"records": _deal_records(), "mapping": "serve_test_deals", "agg": "count"},
        )
        assert resp.status_code == 200
        assert all(row["y"] == 1.0 for row in resp.json()["head"])

    def test_preview_already_shaped_panel_no_mapping(self, client):
        resp = client.post("/preview", json={"records": _panel_records()})
        assert resp.status_code == 200
        out = resp.json()
        assert out["rows"] == 12
        assert out["series"] == ["s"]


class TestForecast:
    def test_inline_panel_records_return_h_rows_with_yhat(self, client):
        # inline already-shaped panel, no mapping; default model
        resp = client.post("/forecast", json={"records": _panel_records(), "h": 4})
        assert resp.status_code == 200
        rows = resp.json()["forecast"]
        assert len(rows) == 4
        for row in rows:
            assert {"unique_id", "ds", "yhat", "lo-80", "hi-80"} <= set(row)
            assert row["unique_id"] == "s"
            assert row["lo-80"] <= row["yhat"] <= row["hi-80"]

    def test_with_registered_mapping_applied(self, client):
        resp = client.post(
            "/forecast",
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "model": "naive",
                "h": 6,
            },
        )
        assert resp.status_code == 200
        rows = resp.json()["forecast"]
        assert len(rows) == 12  # 6 periods x 2 series
        assert {r["unique_id"] for r in rows} == {"alice", "bob"}
        alice = [r for r in rows if r["unique_id"] == "alice"]
        assert len(alice) == 6
        assert alice[0]["ds"].startswith("2024-01-01")

    def test_model_params_and_level_forwarded(self, client):
        resp = client.post(
            "/forecast",
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "model": "window_average",
                "model_params": {"window": 3},
                "h": 3,
                "level": [90],
            },
        )
        assert resp.status_code == 200
        rows = resp.json()["forecast"]
        assert len(rows) == 6
        assert {"lo-90", "hi-90"} <= set(rows[0])
        assert "lo-80" not in rows[0]


class TestCompare:
    def test_returns_leaderboard(self, client):
        resp = client.post(
            "/compare",
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "models": ["naive", "theta"],
                "h": 4,
                "n_windows": 2,
            },
        )
        assert resp.status_code == 200
        out = resp.json()
        assert set(out) == {"leaderboard", "failures"}
        assert out["failures"] == []
        board = out["leaderboard"]
        assert len(board) == 2
        assert {row["model"] for row in board} == {"naive", "theta"}
        assert set(board[0]) == {"model", "mase"}
        assert board[0]["mase"] <= board[1]["mase"]


class TestQuota:
    def test_returns_attainment_and_unmatched_keys(self, client):
        resp = client.post(
            "/quota",
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "h": 6,
                "quota": {"total": 3000.0, "allice": 500.0},  # "allice" is a typo
            },
        )
        assert resp.status_code == 200
        out = resp.json()
        assert set(out) == {"rows", "unmatched_quota_keys"}
        assert [r["unique_id"] for r in out["rows"]] == ["total"]
        assert out["unmatched_quota_keys"] == ["allice"]
        row = out["rows"][0]
        assert 0.0 <= row["p_attain"] <= 1.0
        assert row["quota"] == 3000.0

    def test_scalar_quota_scores_every_series(self, client):
        resp = client.post(
            "/quota",
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "h": 6,
                "quota": 100.0,
            },
        )
        assert resp.status_code == 200
        out = resp.json()
        assert {r["unique_id"] for r in out["rows"]} == {"alice", "bob", "total"}
        assert out["unmatched_quota_keys"] == []


class TestErrorHandling:
    def test_contract_violation_is_400_not_500(self, client):
        # structurally valid body, but the records violate the panel contract
        # (no mapping, and no unique_id/ds/y columns) -> ValueError -> HTTP 400
        resp = client.post("/forecast", json={"records": [{"foo": 1}]})
        assert resp.status_code == 400
        body = resp.json()
        assert set(body) == {"error"}
        assert isinstance(body["error"], str) and body["error"]

    def test_unknown_model_is_400(self, client):
        resp = client.post(
            "/forecast",
            json={"records": _panel_records(), "model": "no_such_model_xyz"},
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_unknown_mapping_is_400(self, client):
        resp = client.post(
            "/preview", json={"records": _deal_records(), "mapping": "no_such_mapping_xyz"}
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    @pytest.mark.parametrize(
        ("route", "extra"),
        [
            ("/preview", {}),
            ("/forecast", {"h": 2}),
            ("/compare", {"h": 2, "n_windows": 2, "models": ["naive"]}),
            ("/quota", {"h": 2, "quota": 1.0, "model": "naive"}),
        ],
    )
    def test_zero_length_freq_is_400_not_500(self, client, route, extra):
        """A caller-supplied ``freq`` of "0D" used to be a 500 on every route.

        Resampling on a zero-length frequency raises ZeroDivisionError, which
        is neither ValueError nor ForecastOSError, so it escaped the handlers
        and became a bare "Internal Server Error" — while the control case
        freq="ZZZ" correctly returned 400. Both are the same class of mistake:
        a bad value in the request body.
        """
        resp = client.post(
            route,
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "freq": "0D",
                **extra,
            },
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_unhashable_record_values_are_400_not_500(self, client):
        """Nested JSON objects where a series id / timestamp belongs -> 400.

        ``{"unique_id": {...}}`` raises TypeError("unhashable type: dict") and
        ``{"ds": {...}}`` raises TypeError from ``to_datetime``; both escaped
        as 500s though both are ordinary malformed client data.
        """
        for rows in (
            [{"unique_id": {"a": 1}, "ds": "2024-01-01", "y": 1.0}] * 3,
            [{"unique_id": "s", "ds": {"a": 1}, "y": 1.0}] * 3,
        ):
            resp = client.post("/forecast", json={"records": rows, "h": 2})
            assert resp.status_code == 400, resp.text
            assert "error" in resp.json()

    def test_nan_quota_is_400_but_infinities_are_scored(self, client):
        """A NaN quota returned 200 with ``p_attain: null`` — not a probability.

        JSON's non-standard ``NaN``/``Infinity`` literals parse fine, so a raw
        body reached the scorer. NaN is unanswerable and is a 400. An infinite
        quota is answerable — ``P(total >= +inf) = 0`` — and v0.9.0 answered
        it, so the finiteness guard over-reached and is narrowed to NaN.
        """
        import json as _json

        body = _json.dumps({"records": _panel_records(), "h": 3, "model": "naive"})

        def _post(literal):
            return client.post(
                "/quota",
                content=body[:-1] + f', "quota": {literal}}}',
                headers={"content-type": "application/json"},
            )

        resp = _post("NaN")
        assert resp.status_code == 400, resp.text
        assert "number" in resp.json()["error"]

        for literal, expected in (("Infinity", 0.0), ("-Infinity", 1.0)):
            resp = _post(literal)
            assert resp.status_code == 200, (literal, resp.text)
            assert all(row["p_attain"] == expected for row in resp.json()["rows"])

    def test_negative_seasonality_is_400(self, client):
        """seasonality=-1 returned a MASE where naive 'beat' naive 11x."""
        resp = client.post(
            "/compare",
            json={
                "records": _panel_records(),
                "models": ["naive"],
                "h": 2,
                "n_windows": 2,
                "seasonality": -1,
            },
        )
        assert resp.status_code == 400
        assert "seasonality" in resp.json()["error"]

    def test_boolean_horizon_is_400(self, client):
        """``h: true`` silently forecast 1 period; a boolean is not a horizon.

        pydantic's lax mode coerces bool -> int (bool is an int subclass), so
        ``true`` became h=1 and returned a 200 with one forecast row. Only the
        unrelated positivity guard stopped ``false`` from doing the same.
        String and integral-float horizons keep working — the coercion that is
        wrong is the boolean one.
        """
        import json as _json

        records = _json.dumps(_panel_records())
        for literal, expected in (("true", 400), ("false", 400), ('"2"', 200), ("2.0", 200)):
            resp = client.post(
                "/forecast",
                content=f'{{"records": {records}, "h": {literal}}}',
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == expected, (literal, resp.text)
            if expected == 200:
                assert len(resp.json()["forecast"]) == 2

    def test_boolean_is_rejected_on_every_numeric_field(self, client):
        """The bool guard went on ``h`` only; its siblings share the hole.

        ``{"level": true}`` returned a 200 whose ``p_attain`` came from a 1%
        interval instead of 80% — a probability answering a question nobody
        asked, strictly worse than the ``h`` case. ``{"seasonality": true}``
        silently swapped the seasonal-naive MASE denominator for a naive one
        (the ``< 1`` guard cannot see it: ``int(True) == 1``), and
        ``{"quota": true}`` scored against a target of 1.0.
        """
        import json as _json

        records = _json.dumps(_panel_records())
        for route, extra in (
            ("/quota", '"quota": 100.0, "level": true'),
            ("/quota", '"quota": true'),
            ("/compare", '"seasonality": true'),
            ("/compare", '"n_windows": true'),
            ("/compare", '"level": [true]'),
            ("/forecast", '"level": [true]'),
        ):
            resp = client.post(
                route,
                content=f'{{"records": {records}, "h": 2, {extra}}}',
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400, (route, extra, resp.text)
            assert "boolean" in resp.text

    def test_numeric_fields_keep_their_lax_coercions(self, client):
        """Backward compat: only the boolean coercion is refused.

        A JSON body has no int/float distinction and a form-encoded client
        sends numbers as strings, so ``"quota": "100"``, ``"level": 80.0`` and
        ``"seasonality": "4"`` are legitimate v0.9.0 calls and must stay 200.
        """
        import json as _json

        records = _json.dumps(_panel_records())
        for route, extra in (
            ("/quota", '"quota": "100", "level": 80.0'),
            ("/compare", '"seasonality": "4", "n_windows": 2.0'),
            ("/forecast", '"level": [90.0]'),
        ):
            resp = client.post(
                route,
                content=f'{{"records": {records}, "h": 2, {extra}}}',
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200, (route, extra, resp.text)

    def test_quota_matching_no_series_is_400(self, client):
        resp = client.post(
            "/quota",
            json={
                "records": _deal_records(),
                "mapping": "serve_test_deals",
                "h": 6,
                "quota": {"ghost": 1.0},
            },
        )
        assert resp.status_code == 400
        assert "error" in resp.json()


class TestImportGuard:
    def test_create_app_without_fastapi_raises_serve_hint(self, monkeypatch):
        monkeypatch.setattr(serve_app, "_HAS_FASTAPI", False)
        with pytest.raises(ImportError, match=r"forecast-os\[serve\]"):
            serve_app.create_app()

    def test_main_without_fastapi_raises_serve_hint(self, monkeypatch):
        monkeypatch.setattr(serve_app, "_HAS_FASTAPI", False)
        with pytest.raises(ImportError, match=r"forecast-os\[serve\]"):
            serve_app.main([])

    def test_main_without_uvicorn_raises_serve_hint(self, monkeypatch):
        monkeypatch.setattr(serve_app, "_HAS_UVICORN", False)
        with pytest.raises(ImportError, match=r"forecast-os\[serve\]"):
            serve_app.main([])


class TestMainEntryPoint:
    def test_main_runs_uvicorn_with_app_and_args(self, monkeypatch):
        captured = {}

        def fake_run(app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

        monkeypatch.setattr(serve_app.uvicorn, "run", fake_run)
        serve_app.main(["--host", "0.0.0.0", "--port", "9123", "--reload"])
        # the app instance is what create_app() builds (has the routes)
        assert captured["app"] is not None
        assert captured["kwargs"]["host"] == "0.0.0.0"
        assert captured["kwargs"]["port"] == 9123
        assert captured["kwargs"]["reload"] is True

    def test_main_defaults(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            serve_app.uvicorn, "run", lambda app, **kw: captured.update(kw)
        )
        serve_app.main([])
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 8000
        assert captured["reload"] is False


def test_malformed_body_returns_400_error_envelope():
    """Pydantic body-validation failures return 400 {"error"}, not 422 {"detail"} bare."""
    from fastapi.testclient import TestClient

    from forecast_os.serve.app import create_app

    client = TestClient(create_app())
    r = client.post("/forecast", json={"model": "naive"})  # missing required records
    assert r.status_code == 400
    body = r.json()
    assert "error" in body and "invalid request body" in body["error"]


class TestPreviewDoesNotReadServerSidePaths:
    """``csv_path`` must not be reachable over HTTP.

    It was passed straight to ``pandas.read_csv``, which treats a URL as
    readily as a path. An unauthenticated request could therefore read any
    file the server process could reach, make the server issue outbound
    requests to an arbitrary host *with the fetched body reflected back in the
    200 response*, and allocate unbounded memory decompressing a remote
    archive — all before any panel validation ran.
    """

    def test_url_is_not_fetched(self, client, monkeypatch):
        import pandas as pd

        attempted = []

        def spy(*args, **kwargs):  # pragma: no cover - must never run
            attempted.append(args[0] if args else None)
            raise AssertionError(f"read_csv reached the network/filesystem: {args!r}")

        monkeypatch.setattr(pd, "read_csv", spy)
        resp = client.post("/preview", json={"csv_path": "http://169.254.169.254/latest/meta-data"})
        assert resp.status_code == 400
        assert attempted == []

    def test_local_path_is_not_read(self, client):
        resp = client.post("/preview", json={"csv_path": "/etc/passwd"})
        assert resp.status_code == 400
        assert "passwd" not in resp.text

    def test_inline_records_still_work(self, client, _panel_records=None):
        rows = [
            {"unique_id": "a", "ds": "2024-01-01", "y": 1.0},
            {"unique_id": "a", "ds": "2024-01-02", "y": 2.0},
        ]
        resp = client.post("/preview", json={"records": rows})
        assert resp.status_code == 200
        assert resp.json()["rows"] == 2


class TestMalformedModelParamsAreNot500:
    """Bad `model_params` must yield 400 + {"error": ...}, never a 500 traceback.

    `get_model` was guarded, but constructor overrides that SURVIVE construction
    and are only the wrong type deep inside fit()/predict() (season_length="x",
    lags=true, models=[1]) escaped every handler and surfaced as a 500 with a
    server-side traceback — contradicting the "never a 500 traceback" guarantee
    in the module docstring and docs/serving.md.
    """

    @staticmethod
    def _panel():
        return [{"unique_id": "a", "ds": f"2024-{m:02d}-01", "y": float(m)} for m in range(1, 13)]

    @pytest.mark.parametrize(
        "model,params",
        [
            ("theta", {"season_length": "x"}),
            ("ridge_lag", {"lags": True}),
            ("holt", {"alpha": "x"}),
            ("ensemble", {"models": [1]}),
        ],
    )
    def test_bad_model_params_return_400(self, client, model, params):
        resp = client.post(
            "/forecast",
            json={"records": self._panel(), "model": model, "h": 2, "model_params": params},
        )
        assert resp.status_code == 400, f"{model} {params} returned {resp.status_code}"
        assert "error" in resp.json()
        assert "model_params" in resp.json()["error"]

    def test_valid_model_params_still_work(self, client):
        resp = client.post(
            "/forecast",
            json={
                "records": self._panel(),
                "model": "theta",
                "h": 2,
                "model_params": {"season_length": 4},
            },
        )
        assert resp.status_code == 200
