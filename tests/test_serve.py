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
