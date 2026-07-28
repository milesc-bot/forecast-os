"""Tests for the MCP server tool functions (forecast_os.mcp.server).

Every tool is exercised as a plain Python function — no running server, no
network. The FastMCP wiring itself gets a construction smoke test (skipped
when the optional ``mcp`` extra is missing).
"""

import asyncio
import json

import pandas as pd
import pytest

from forecast_os.connectors.base import SchemaMapping, apply_mapping, register_mapping
from forecast_os.core import registry
from forecast_os.core.base import BaseForecaster
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.mcp import server

register_mapping(
    SchemaMapping(
        name="mcp_test_deals",
        description="test recipe: inline deal records to a monthly per-rep panel",
        date_col="close_date",
        id_cols=("rep",),
        value_col="amount",
        freq="MS",
    )
)


@pytest.fixture
def failing_model():
    """Register a deliberately failing model for the duration of one test."""
    name = "_test_always_fails"

    @registry.register(name, family="baseline", description="test model that always fails")
    class _AlwaysFails(BaseForecaster):
        def fit(self, df):
            raise RuntimeError("deliberate test failure")

        def predict(self, h, level=None):  # pragma: no cover - fit always raises
            raise RuntimeError("unreachable")

    yield name
    registry._REGISTRY.pop(name, None)


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


class TestLoadRecords:
    def test_neither_input_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            server._load_records()

    def test_both_inputs_raise(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("a\n1\n")
        with pytest.raises(ValueError, match="exactly one"):
            server._load_records(csv_path=str(path), records=[{"a": 1}])

    def test_inline_records(self):
        frame = server._load_records(records=[{"a": 1}, {"a": 2}])
        assert list(frame["a"]) == [1, 2]

    def test_csv_path(self, tmp_path):
        path = tmp_path / "deals.csv"
        pd.DataFrame(_deal_records()).to_csv(path, index=False)
        frame = server._load_records(csv_path=str(path))
        assert len(frame) == 48
        assert list(frame.columns) == ["rep", "close_date", "amount"]

    def test_missing_csv_is_value_error(self):
        with pytest.raises(ValueError, match="cannot read"):
            server._load_records(csv_path="/nonexistent/nope.csv")


class TestListTools:
    def test_list_models_tool(self):
        models = server.list_models_tool()
        assert set(models[0]) == {"name", "family", "description"}
        names = [m["name"] for m in models]
        assert "auto_select" in names
        assert "theta" in names
        assert "reconciled" in names
        json.dumps(models)

    def test_list_mappings_tool(self):
        mappings = server.list_mappings_tool()
        ours = [m for m in mappings if m["name"] == "mcp_test_deals"]
        assert len(ours) == 1
        assert ours[0]["freq"] == "MS"
        assert "deal records" in ours[0]["description"]
        json.dumps(mappings)


class TestPreviewPanel:
    def test_preview_with_mapping(self):
        out = server.preview_panel(records=_deal_records(), mapping="mcp_test_deals")
        assert out["rows"] == 48
        assert out["series"] == ["alice", "bob"]
        assert len(out["head"]) == 10
        first = out["head"][0]
        assert set(first) == {"unique_id", "ds", "y"}
        assert isinstance(first["ds"], str)
        assert first["ds"].startswith("2022-01-01")
        assert first["y"] == pytest.approx(100.0)
        json.dumps(out)

    def test_preview_from_csv_path(self, tmp_path):
        path = tmp_path / "deals.csv"
        pd.DataFrame(_deal_records()).to_csv(path, index=False)
        out = server.preview_panel(csv_path=str(path), mapping="mcp_test_deals")
        assert out["rows"] == 48

    def test_preview_overrides_forwarded(self):
        out = server.preview_panel(
            records=_deal_records(), mapping="mcp_test_deals", agg="count"
        )
        assert all(row["y"] == 1.0 for row in out["head"])

    def test_already_shaped_panel_without_mapping(self):
        records = [
            {"unique_id": "s", "ds": f"2024-0{m}-01", "y": float(m)} for m in range(1, 7)
        ]
        out = server.preview_panel(records=records)
        assert out["rows"] == 6
        assert out["series"] == ["s"]

    def test_overrides_without_mapping_raise(self):
        records = [{"unique_id": "s", "ds": "2024-01-01", "y": 1.0}]
        with pytest.raises(ValueError, match="mapping"):
            server.preview_panel(records=records, freq="MS")

    def test_unknown_mapping_is_value_error(self):
        with pytest.raises(ValueError, match="unknown mapping"):
            server.preview_panel(records=_deal_records(), mapping="no_such_mapping_xyz")

    def test_contract_violation_surfaces_as_plain_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            server.preview_panel(records=[{"foo": 1}], mapping="mcp_test_deals")
        assert not isinstance(excinfo.value, ForecastOSError)

    def test_neither_input_raises_through_tool(self):
        with pytest.raises(ValueError, match="exactly one"):
            server.preview_panel(mapping="mcp_test_deals")


class TestForecastTool:
    def test_end_to_end_default_model(self):
        rows = server.forecast_tool(records=_deal_records(), mapping="mcp_test_deals", h=6)
        assert len(rows) == 12
        assert {r["unique_id"] for r in rows} == {"alice", "bob"}
        for row in rows:
            assert {"unique_id", "ds", "yhat", "lo-80", "hi-80"} <= set(row)
            assert row["lo-80"] <= row["yhat"] <= row["hi-80"]
        # forecasts continue the panel: history ends 2023-12, so h=6 starts 2024-01
        alice = [r for r in rows if r["unique_id"] == "alice"]
        assert len(alice) == 6
        assert alice[0]["ds"].startswith("2024-01-01")
        json.dumps(rows)

    def test_model_params_and_level(self):
        rows = server.forecast_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            model="window_average",
            model_params={"window": 3},
            h=3,
            level=[90],
        )
        assert len(rows) == 6
        assert {"lo-90", "hi-90"} <= set(rows[0])
        assert "lo-80" not in rows[0]

    def test_freq_override_forecasts_the_previewed_panel(self):
        # preview-with-freq='W' then forecast-with-freq='W' must see the SAME panel
        records = _deal_records()
        preview = server.preview_panel(records=records, mapping="mcp_test_deals", freq="W")
        panel = apply_mapping(pd.DataFrame(records), "mcp_test_deals", freq="W")
        assert preview["rows"] == len(panel)
        assert preview["rows"] > 48  # weekly reshape, not the mapping's monthly default

        rows = server.forecast_tool(
            records=records, mapping="mcp_test_deals", model="naive", h=4, freq="W"
        )
        assert len(rows) == 4 * len(preview["series"])
        for uid in preview["series"]:
            ds = pd.to_datetime([r["ds"] for r in rows if r["unique_id"] == uid])
            # weekly spacing, continuing exactly where the weekly panel ends
            assert (ds.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all()
            last_hist = panel.loc[panel["unique_id"] == uid, "ds"].max()
            assert ds[0] == last_hist + pd.Timedelta(weeks=1)

    def test_agg_override_reaches_panel(self):
        # count panel is 1.0 per rep-month, so the forecast stays near 1, not dollars
        rows = server.forecast_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            model="naive",
            h=3,
            agg="count",
        )
        assert all(r["yhat"] == pytest.approx(1.0) for r in rows)

    def test_unknown_model_is_value_error(self):
        with pytest.raises(ValueError, match="unknown model"):
            server.forecast_tool(
                records=_deal_records(), mapping="mcp_test_deals", model="nope"
            )

    def test_bad_model_params_is_value_error(self):
        with pytest.raises(ValueError, match="model_params"):
            server.forecast_tool(
                records=_deal_records(),
                mapping="mcp_test_deals",
                model="window_average",
                model_params={"bogus": 1},
            )

    def test_fractional_level_is_rejected_not_truncated(self):
        """Regression: ``level=[99.9]`` was ``int()``-ed to 99 right here.

        The model layer rejects a fractional level because the interval is
        labelled ``lo-{level}``/``hi-{level}``, but this tool truncated
        first, so the guard never saw the value: an MCP client asking for
        99.9% quietly received a ~22% narrower interval under a ``lo-99``
        label it never asked for. The value now reaches ``predict``
        uncoerced.
        """
        with pytest.raises(ValueError, match="whole number"):
            server.forecast_tool(
                records=_deal_records(), mapping="mcp_test_deals", h=2, level=[99.9]
            )

    def test_whole_number_float_level_still_works(self):
        """JSON has no int/float distinction: ``level=[90.0]`` must still work."""
        rows = server.forecast_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=2, level=[90.0]
        )
        assert {"lo-90", "hi-90"} <= set(rows[0])

    @pytest.mark.parametrize("kwargs", [{"h": True}, {"level": [True]}])
    def test_boolean_arguments_are_rejected(self, kwargs):
        """``bool`` is an ``int`` subclass: ``h=true`` forecast one period and
        ``level=[true]`` produced a 1% interval, both silently."""
        kwargs.setdefault("h", 2)
        with pytest.raises(ValueError, match="not a boolean"):
            server.forecast_tool(
                records=_deal_records(), mapping="mcp_test_deals", **kwargs
            )


class TestCompareTool:
    def test_leaderboard_shape(self):
        out = server.compare_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            models=["naive", "theta"],
            h=4,
            n_windows=2,
        )
        assert set(out) == {"leaderboard", "failures"}
        assert out["failures"] == []
        board = out["leaderboard"]
        assert len(board) == 2
        assert set(board[0]) == {"model", "mase"}
        assert {row["model"] for row in board} == {"naive", "theta"}
        # best model first (mase ascending)
        assert board[0]["mase"] <= board[1]["mase"]
        json.dumps(out)

    def test_explicit_metrics_and_seasonality(self):
        board = server.compare_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            models=["naive", "drift"],
            h=4,
            n_windows=2,
            metrics=["mae", "rmse"],
            seasonality=12,
        )["leaderboard"]
        assert set(board[0]) == {"model", "mae", "rmse"}
        assert board[0]["mae"] <= board[1]["mae"]

    def test_failing_model_lands_in_failures_not_silence(self, failing_model):
        out = server.compare_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            models=["naive", failing_model],
            h=4,
            n_windows=2,
        )
        assert [row["model"] for row in out["leaderboard"]] == ["naive"]
        # the agent must SEE why the model is absent from the board
        assert any(
            failing_model in msg and "deliberate test failure" in msg
            for msg in out["failures"]
        )
        json.dumps(out)

    def test_agg_override_reaches_panel(self):
        # count panel is constant 1.0, so naive backtests to exactly zero error
        out = server.compare_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            models=["naive"],
            h=4,
            n_windows=2,
            metrics=["mae"],
            agg="count",
        )
        assert out["leaderboard"][0]["mae"] == pytest.approx(0.0)

    def test_unrelated_warnings_are_not_reported_as_failures(self):
        """``failures`` explains absent models — it is not a warning dump.

        It used to be built from *every* warning raised anywhere inside the
        backtest, so a panel whose values overflow float64 during scoring
        produced dozens of identical numpy 'overflow encountered in square'
        strings alongside a complete leaderboard: no model failed, yet the
        agent was handed a pile of unattributed warnings whose length grew
        with ``n_windows``. Every model on the board means ``failures == []``.
        """
        months = pd.date_range("2022-01-01", periods=36, freq="MS")
        records = [
            {"unique_id": "s", "ds": d.strftime("%Y-%m-%d"), "y": 1e300 * (1 + i % 5)}
            for i, d in enumerate(months)
        ]
        out = server.compare_tool(records=records, models=["theta", "naive"], h=3, n_windows=5)
        assert {row["model"] for row in out["leaderboard"]} == {"theta", "naive"}
        assert out["failures"] == []

    def test_failed_model_is_reported_even_when_its_warning_is_lost(
        self, failing_model, monkeypatch
    ):
        """A model missing from the board is never silently missing.

        ``failures`` was inferred purely from ``warnings.catch_warnings``,
        which mutates process-global state: when two requests overlapped in
        FastAPI's threadpool, one request's ``__exit__`` restored the default
        showwarning under the other, and the failure explanation vanished —
        the caller got a short leaderboard with ``failures == []`` and no clue
        a model had been dropped. The list is now derived from the board
        itself (requested models minus survivors), so the warning text is an
        enrichment, not the source of truth. Losing the warning is simulated
        deterministically here by neutering the engine's ``warnings.warn``.
        """
        import types

        from forecast_os import engine as engine_module

        monkeypatch.setattr(
            engine_module, "warnings", types.SimpleNamespace(warn=lambda *a, **k: None)
        )
        out = server.compare_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            models=["naive", failing_model],
            h=4,
            n_windows=2,
        )
        assert [row["model"] for row in out["leaderboard"]] == ["naive"]
        assert len(out["failures"]) == 1
        assert failing_model in out["failures"][0]
        json.dumps(out)

    def test_model_specs_beyond_plain_names_still_work(self):
        """Direct Python callers may pass ``(name, params)`` specs.

        ``failures`` is now derived by matching requested models against the
        board index, which only works for registry names; a spec tuple must
        not crash that lookup or be mis-reported as a failed model.
        """
        out = server.compare_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            models=[("window_average", {"window": 3}), "naive"],
            h=4,
            n_windows=2,
        )
        assert {row["model"] for row in out["leaderboard"]} == {"window_average", "naive"}
        assert out["failures"] == []

    def test_non_positive_seasonality_is_rejected(self):
        """A negative ``seasonality`` produced a meaningless, inverted MASE.

        The MASE denominator is ``mean|y[m:] - y[:-m]|``; for m = -1 that
        slices ``y[-1:] - y[:1]``, i.e. the whole series range, deflating
        every score — ``naive`` scored 0.09 against its own yardstick, so an
        agent reading "below 1 beats a naive forecaster" would pick a model on
        a fabricated margin. m = 0 already failed, but with a raw numpy
        broadcast message. A seasonal period below 1 is not a period.
        """
        for bad in (0, -1, -12):
            with pytest.raises(ValueError, match="seasonality"):
                server.compare_tool(
                    records=_deal_records(),
                    mapping="mcp_test_deals",
                    models=["naive"],
                    h=2,
                    n_windows=2,
                    seasonality=bad,
                )

    @pytest.mark.parametrize(
        "kwargs", [{"h": True}, {"n_windows": True}, {"seasonality": True}, {"level": [True]}]
    )
    def test_boolean_arguments_are_rejected(self, kwargs):
        """``seasonality=true`` is 1, so the ``< 1`` guard waved it through.

        That silently swaps the seasonal-naive MASE denominator for a naive
        one — a leaderboard measured against a yardstick the caller did not
        choose. ``n_windows``/``h``/``level`` share the coercion.
        """
        kwargs.setdefault("h", 2)
        kwargs.setdefault("n_windows", 2)
        with pytest.raises(ValueError, match="not a boolean"):
            server.compare_tool(
                records=_deal_records(),
                mapping="mcp_test_deals",
                models=["naive"],
                **kwargs,
            )


class TestQuotaTool:
    def test_probabilities_sane(self):
        out = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota=100.0
        )
        assert set(out) == {"rows", "unmatched_quota_keys"}
        assert out["unmatched_quota_keys"] == []
        rows = out["rows"]
        # default model is "reconciled": leaves plus the grand total roll-up
        assert {r["unique_id"] for r in rows} == {"alice", "bob", "total"}
        for row in rows:
            assert 0.0 <= row["p_attain"] <= 1.0
            assert row["quota"] == 100.0
            assert row["expected"] > 0.0
        json.dumps(out)

    def test_easy_quota_near_one_absurd_quota_near_zero(self):
        easy = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota=1.0
        )
        assert all(r["p_attain"] > 0.99 for r in easy["rows"])
        absurd = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota=1e9
        )
        assert all(r["p_attain"] < 0.01 for r in absurd["rows"])

    def test_dict_quota_scores_only_named_series(self):
        out = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota={"total": 3000.0}
        )
        assert [r["unique_id"] for r in out["rows"]] == ["total"]
        assert out["rows"][0]["quota"] == 3000.0
        assert out["unmatched_quota_keys"] == []

    def test_typo_quota_key_is_surfaced_not_swallowed(self):
        out = server.quota_tool(
            records=_deal_records(),
            mapping="mcp_test_deals",
            h=6,
            quota={"total": 3000.0, "allice": 500.0},  # "allice" is a typo for alice
        )
        assert [r["unique_id"] for r in out["rows"]] == ["total"]
        assert out["unmatched_quota_keys"] == ["allice"]
        json.dumps(out)

    def test_agg_override_reaches_panel(self):
        # count panel is 1.0 per rep-month: a 6-period total near 6, not dollars
        out = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota=100.0, agg="count"
        )
        alice = next(r for r in out["rows"] if r["unique_id"] == "alice")
        assert alice["expected"] < 20.0

    def test_dict_quota_matching_no_series_raises(self):
        with pytest.raises(ValueError, match="match no forecast series"):
            server.quota_tool(
                records=_deal_records(), mapping="mcp_test_deals", h=6, quota={"ghost": 1.0}
            )

    def test_quota_is_required(self):
        with pytest.raises(TypeError, match="quota"):
            server.quota_tool(records=_deal_records(), mapping="mcp_test_deals", h=6)

    def test_nan_quota_is_rejected(self):
        """A NaN quota broke the documented ``p_attain in [0, 1]``.

        NaN scored every series ``p_attain=None`` (JSON null — a consumer
        doing ``0 <= p <= 1`` gets a TypeError on a field declared a
        probability). A NaN in ``y`` is already a client error at the
        boundary; a NaN target is too.
        """
        with pytest.raises(ValueError, match="quota must be a number"):
            server.quota_tool(
                records=_deal_records(), mapping="mcp_test_deals", h=6, quota=float("nan")
            )

    def test_nan_quota_in_dict_is_rejected(self):
        with pytest.raises(ValueError, match="quota must be a number"):
            server.quota_tool(
                records=_deal_records(),
                mapping="mcp_test_deals",
                h=6,
                quota={"total": float("nan")},
            )

    @pytest.mark.parametrize(("target", "expected"), [(float("inf"), 0.0), (float("-inf"), 1.0)])
    def test_infinite_quota_is_scored_not_rejected(self, target, expected):
        """An infinite quota has a correct, well-defined answer: keep it.

        The finiteness guard over-reached. ``P(total >= +inf)`` is 0.0 and
        ``P(total >= -inf)`` is 1.0 — both probabilities in [0, 1], both
        what v0.9.0 returned. Only NaN is unanswerable. The echoed ``quota``
        field is JSON null (infinity has no JSON spelling), which the
        docstring now says.
        """
        out = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota=target
        )
        assert all(row["p_attain"] == expected for row in out["rows"])

    def test_string_quota_still_works(self):
        """Regression: ``math.isfinite("100")`` raised an uncaught TypeError.

        ``quota_tool`` is exported in ``__all__``, and at v0.9.0 a direct
        Python caller could pass the string form a JSON/CSV round trip
        produces — ``attainment_probability`` floats it internally, giving
        the same answer as ``quota=100.0``. The finiteness guard tested the
        raw value before any coercion, and TypeError is not one of the
        exceptions ``_tool_errors`` normalises, so it escaped raw.
        """
        as_text = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota="100"
        )
        as_number = server.quota_tool(
            records=_deal_records(), mapping="mcp_test_deals", h=6, quota=100.0
        )
        assert as_text == as_number
        assert all(0.0 <= row["p_attain"] <= 1.0 for row in as_text["rows"])

    def test_non_numeric_quota_raises_value_error(self):
        """A quota that is not a number at all is a ValueError, not a TypeError."""
        with pytest.raises(ValueError, match="quota must be a number"):
            server.quota_tool(
                records=_deal_records(), mapping="mcp_test_deals", h=6, quota="lots"
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"quota": True},
            {"quota": 100.0, "h": True},
            {"quota": 100.0, "level": True},
        ],
    )
    def test_boolean_arguments_are_rejected(self, kwargs):
        """``bool`` is an ``int`` subclass, so every one of these was silent.

        ``quota=true`` scored against a target of 1.0, ``h=true`` forecast a
        single period, and ``level=true`` reported attainment against a 1%
        interval labelled ``lo-1``/``hi-1``. The guard the sweep added to the
        HTTP ``h`` field belongs on every numeric argument, on both surfaces.
        """
        kwargs.setdefault("h", 6)
        with pytest.raises(ValueError, match="not a boolean"):
            server.quota_tool(records=_deal_records(), mapping="mcp_test_deals", **kwargs)


class TestClientDataErrorsAreValueErrors:
    """Bad *data* must raise ``ValueError`` — the REST layer turns it into 400.

    Panel shaping runs entirely on caller-supplied values, but only
    ValueError/ForecastOSError were normalised, so ordinary bad input escaped
    as ZeroDivisionError/TypeError: over HTTP that is a bare 500 "Internal
    Server Error", contradicting the serving module's documented "never a 500"
    contract, and over MCP it is an exception type clients do not expect.
    """

    def test_zero_length_freq_is_a_value_error(self):
        with pytest.raises(ValueError):
            server.preview_panel(
                records=_deal_records(), mapping="mcp_test_deals", freq="0D"
            )

    def test_unhashable_series_id_is_a_value_error(self):
        rows = [{"unique_id": {"a": 1}, "ds": "2024-01-01", "y": 1.0}] * 3
        with pytest.raises(ValueError):
            server.preview_panel(records=rows)

    def test_undatelike_timestamp_is_a_value_error(self):
        rows = [{"unique_id": "s", "ds": {"a": 1}, "y": 1.0}] * 3
        with pytest.raises(ValueError):
            server.preview_panel(records=rows)


class TestMCPWiring:
    def test_main_without_mcp_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(server, "_HAS_MCP", False)
        with pytest.raises(ImportError, match=r"forecast-os\[mcp\]"):
            server.main()

    @pytest.mark.skipif(not server._HAS_MCP, reason="optional mcp extra not installed")
    def test_fastmcp_server_exposes_six_tools(self):
        srv = server._build_server()
        assert srv.name == "forecast-os"
        tools = asyncio.run(srv.list_tools())
        assert {t.name for t in tools} == {
            "list_models",
            "list_mappings",
            "preview_panel",
            "forecast",
            "compare",
            "quota",
        }
        # the docstrings are the agent-facing UX — every tool must carry one
        assert all(t.description for t in tools)

    @pytest.mark.skipif(not server._HAS_MCP, reason="optional mcp extra not installed")
    def test_panel_shaping_overrides_exposed_on_every_panel_tool(self):
        # preview/forecast/compare/quota must accept the SAME freq/agg overrides,
        # so an agent can preview a panel and then operate on that same panel
        tools = {t.name: t for t in asyncio.run(server._build_server().list_tools())}
        for name in ("preview_panel", "forecast", "compare", "quota"):
            props = tools[name].inputSchema["properties"]
            assert {"freq", "agg"} <= set(props), f"{name} is missing freq/agg"


class TestLevelMustBeANumber:
    """A string `level` raised a bare TypeError that escaped the error decorator.

    `level` is passed to predict() uncoerced so the model layer's own whole-number
    check stays visible on this surface. That left a string to fail as a raw
    TypeError from a comparison inside _check_level, which the ValueError-only
    decorator does not catch — surfacing as a 500 traceback instead of a clean
    client error.
    """

    @staticmethod
    def _panel():
        return [{"unique_id": "a", "ds": f"2024-{m:02d}-01", "y": float(m)} for m in range(1, 13)]

    @pytest.mark.parametrize("bad", [["80"], [True], [None]])
    def test_non_numeric_level_raises_value_error(self, bad):
        from forecast_os.mcp.server import forecast_tool

        with pytest.raises(ValueError, match="level must be a number"):
            forecast_tool(records=self._panel(), model="naive", h=2, level=bad)

    def test_numeric_levels_still_work(self):
        from forecast_os.mcp.server import forecast_tool

        rows = forecast_tool(records=self._panel(), model="naive", h=2, level=[80])
        assert len(rows) == 2
        assert "lo-80" in rows[0]
