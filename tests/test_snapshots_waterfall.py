"""Tests for the pipeline waterfall (snapshots.waterfall) and the store's
``deals`` snapshot kind.

The waterfall categorizes every opportunity across two point-in-time deal
snapshots into exactly one bridge category, so the categories partition the
deal universe and the signed ``amount`` column reconciles the open-pipeline
movement (opening pipeline + sum(amount) == closing pipeline).
"""

import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError, ForecastOSError
from forecast_os.snapshots.store import SnapshotStore
from forecast_os.snapshots.waterfall import pipeline_waterfall, waterfall_summary

STAGES = ["prospect", "qualify", "propose", "negotiate"]
WON = "closed_won"
LOST = "closed_lost"


def _before():
    """Opportunity snapshot at t0: eight open deals (D1..D8)."""
    return pd.DataFrame(
        {
            "opp_id": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
            "amount": [100.0, 200.0, 150.0, 300.0, 120.0, 80.0, 500.0, 90.0],
            "stage": [
                "prospect",  # D1 -> advances
                "negotiate",  # D2 -> won
                "qualify",  # D3 -> lost
                "propose",  # D4 -> expands
                "qualify",  # D5 -> pushed (close date slips)
                "prospect",  # D6 -> unchanged
                "propose",  # D7 -> contracts
                "prospect",  # D8 -> removed from the after snapshot
            ],
            "close_date": pd.to_datetime(
                [
                    "2024-06-30",
                    "2024-03-31",
                    "2024-03-31",
                    "2024-06-30",
                    "2024-05-31",
                    "2024-09-30",
                    "2024-06-30",
                    "2024-06-30",
                ]
            ),
            "owner": ["ann", "bob", "ann", "bob", "ann", "bob", "ann", "bob"],
        }
    )


def _after():
    """Opportunity snapshot at t1: D8 dropped, D9 created, rest evolved."""
    return pd.DataFrame(
        {
            "opp_id": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D9"],
            "amount": [100.0, 200.0, 150.0, 400.0, 120.0, 80.0, 350.0, 250.0],
            "stage": [
                "qualify",  # D1 advanced prospect -> qualify
                WON,  # D2 won
                LOST,  # D3 lost
                "propose",  # D4 same stage, amount 300 -> 400 (expanded)
                "qualify",  # D5 same stage/amount, close date later (pushed)
                "prospect",  # D6 unchanged
                "propose",  # D7 same stage, amount 500 -> 350 (contracted)
                "prospect",  # D9 created
            ],
            "close_date": pd.to_datetime(
                [
                    "2024-06-30",
                    "2024-03-31",
                    "2024-03-31",
                    "2024-06-30",
                    "2024-07-31",  # D5 slipped one month later
                    "2024-09-30",
                    "2024-06-30",
                    "2024-08-31",
                ]
            ),
            "owner": ["ann", "bob", "ann", "bob", "ann", "bob", "ann", "cid"],
        }
    )


def _cat(df, category):
    row = df[df["category"] == category]
    assert len(row) == 1, f"expected exactly one {category!r} row, got {len(row)}"
    return int(row["n_deals"].iloc[0]), float(row["amount"].iloc[0])


class TestPipelineWaterfall:
    def test_every_category_n_and_amount(self):
        wf = pipeline_waterfall(
            _before(),
            _after(),
            STAGES,
            close_col="close_date",
            won_stage=WON,
            lost_stage=LOST,
        )
        assert list(wf.columns) == ["category", "n_deals", "amount"]

        assert _cat(wf, "created") == (1, 250.0)  # D9 enters at +amount
        assert _cat(wf, "advanced") == (1, 0.0)  # D1 moved a stage, $ unchanged
        assert _cat(wf, "expanded") == (1, 100.0)  # D4 300 -> 400
        assert _cat(wf, "contracted") == (1, -150.0)  # D7 500 -> 350
        assert _cat(wf, "pushed") == (1, 0.0)  # D5 close date slipped later
        assert _cat(wf, "unchanged") == (1, 0.0)  # D6
        assert _cat(wf, "won") == (1, -200.0)  # D2 leaves open pipeline
        assert _cat(wf, "lost") == (1, -150.0)  # D3 leaves open pipeline
        assert _cat(wf, "removed") == (1, -90.0)  # D8 dropped from snapshot

    def test_categories_partition_the_deal_universe(self):
        before, after = _before(), _after()
        wf = pipeline_waterfall(
            before, after, STAGES, close_col="close_date", won_stage=WON, lost_stage=LOST
        )
        universe = set(before["opp_id"]) | set(after["opp_id"])
        # every deal counted exactly once, none dropped, none double-counted
        assert int(wf["n_deals"].sum()) == len(universe)
        # no category appears twice
        assert wf["category"].is_unique

    def test_amount_reconciles_open_pipeline_movement(self):
        before, after = _before(), _after()
        wf = pipeline_waterfall(
            before, after, STAGES, close_col="close_date", won_stage=WON, lost_stage=LOST
        )
        closed = {WON, LOST}
        o_before = before.loc[~before["stage"].isin(closed), "amount"].sum()
        o_after = after.loc[~after["stage"].isin(closed), "amount"].sum()
        assert wf["amount"].sum() == pytest.approx(o_after - o_before)

    def test_without_close_col_a_pure_date_slip_is_unchanged(self):
        # D5 only changes its close date; with no close_col it cannot be a push
        wf = pipeline_waterfall(
            _before(), _after(), STAGES, won_stage=WON, lost_stage=LOST
        )
        assert "pushed" not in set(wf["category"])
        # D5 and D6 both land in unchanged now
        assert _cat(wf, "unchanged") == (2, 0.0)

    def test_stable_canonical_category_order(self):
        wf = pipeline_waterfall(
            _before(), _after(), STAGES, close_col="close_date", won_stage=WON, lost_stage=LOST
        )
        order = list(wf["category"])
        expected_rel = [
            "created",
            "expanded",
            "advanced",
            "pushed",
            "contracted",
            "won",
            "lost",
            "removed",
        ]
        # each present category keeps the canonical relative order
        assert [c for c in order if c in expected_rel] == [
            c for c in expected_rel if c in order
        ]


class TestValidation:
    def test_won_or_lost_stage_inside_open_stages_raises(self):
        with pytest.raises(DataContractError):
            pipeline_waterfall(
                _before(), _after(), [*STAGES, WON], won_stage=WON, lost_stage=LOST
            )

    def test_missing_required_column_raises(self):
        bad = _before().drop(columns=["stage"])
        with pytest.raises(DataContractError):
            pipeline_waterfall(bad, _after(), STAGES, won_stage=WON, lost_stage=LOST)

    def test_duplicate_opp_id_raises(self):
        dup = pd.concat([_before(), _before().iloc[:1]], ignore_index=True)
        with pytest.raises(DataContractError):
            pipeline_waterfall(dup, _after(), STAGES, won_stage=WON, lost_stage=LOST)

    def test_empty_stages_raises(self):
        with pytest.raises(DataContractError):
            pipeline_waterfall(_before(), _after(), [], won_stage=WON, lost_stage=LOST)

    def test_non_dataframe_raises(self):
        with pytest.raises(DataContractError):
            pipeline_waterfall([1, 2, 3], _after(), STAGES, won_stage=WON, lost_stage=LOST)


class TestWaterfallSummary:
    def test_bridge_anchors_and_running_total(self):
        before, after = _before(), _after()
        summ = waterfall_summary(
            before, after, STAGES, close_col="close_date", won_stage=WON, lost_stage=LOST
        )
        assert list(summ.columns) == ["category", "n_deals", "amount", "running_total"]
        assert summ["category"].iloc[0] == "opening_pipeline"
        assert summ["category"].iloc[-1] == "closing_pipeline"

        closed = {WON, LOST}
        o_before = before.loc[~before["stage"].isin(closed), "amount"].sum()
        o_after = after.loc[~after["stage"].isin(closed), "amount"].sum()
        assert summ["amount"].iloc[0] == pytest.approx(o_before)
        assert summ["amount"].iloc[-1] == pytest.approx(o_after)
        assert summ["running_total"].iloc[0] == pytest.approx(o_before)
        assert summ["running_total"].iloc[-1] == pytest.approx(o_after)

        # the running total after the last movement equals the closing pipeline
        last_move = summ.iloc[-2]
        assert last_move["running_total"] == pytest.approx(o_after)

        # opening/closing deal counts are the open-deal counts of each snapshot
        assert int(summ["n_deals"].iloc[0]) == int((~before["stage"].isin(closed)).sum())
        assert int(summ["n_deals"].iloc[-1]) == int((~after["stage"].isin(closed)).sum())


class TestDealsSnapshotKind:
    def test_round_trip_and_history(self, tmp_path):
        store = SnapshotStore(tmp_path)
        d0, d1 = _before(), _after()
        e = store.snapshot(d0, as_of="2024-03-01", kind="deals")
        assert e["kind"] == "deals"
        assert e["rows"] == len(d0)
        store.snapshot(d1, as_of="2024-04-01", kind="deals")

        loaded = store.load(as_of="2024-03-01", kind="deals")
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True), d0.reset_index(drop=True), check_dtype=False
        )

        # load(None) is the latest deals snapshot
        latest = store.load(kind="deals")
        assert set(latest["opp_id"]) == set(d1["opp_id"])

        # history stacks every deals snapshot with an as_of column
        hist = store.history(kind="deals")
        assert "as_of" in hist.columns
        assert len(hist) == len(d0) + len(d1)
        assert set(pd.to_datetime(hist["as_of"].unique())) == {
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-04-01"),
        }

    def test_deals_kind_requires_opp_id_amount_stage(self, tmp_path):
        store = SnapshotStore(tmp_path)
        bad = pd.DataFrame({"opp_id": ["D1"], "amount": [100.0]})  # missing stage
        with pytest.raises(DataContractError):
            store.snapshot(bad, as_of="2024-03-01", kind="deals")

    def test_deals_kind_rejects_non_dataframe(self, tmp_path):
        store = SnapshotStore(tmp_path)
        with pytest.raises(DataContractError):
            store.snapshot([1, 2, 3], as_of="2024-03-01", kind="deals")

    def test_deals_end_to_end_with_waterfall(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.snapshot(_before(), as_of="2024-03-01", kind="deals")
        store.snapshot(_after(), as_of="2024-04-01", kind="deals")
        before = store.load(as_of="2024-03-01", kind="deals")
        after = store.load(as_of="2024-04-01", kind="deals")
        wf = pipeline_waterfall(
            before, after, STAGES, close_col="close_date", won_stage=WON, lost_stage=LOST
        )
        assert _cat(wf, "won") == (1, -200.0)
        assert _cat(wf, "created") == (1, 250.0)

    def test_panel_and_forecast_kinds_still_validate(self, tmp_path):
        store = SnapshotStore(tmp_path)
        panel = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "y": [1.0, 2.0],
            }
        )
        forecast = pd.DataFrame(
            {
                "unique_id": ["a"],
                "ds": pd.to_datetime(["2024-03-01"]),
                "yhat": [3.0],
            }
        )
        store.snapshot(panel, as_of="2024-02-15", kind="panel")
        store.snapshot(forecast, as_of="2024-02-15", kind="forecast")
        # a deals frame is not a valid panel/forecast
        with pytest.raises(DataContractError):
            store.snapshot(_before(), as_of="2024-02-15", kind="panel")
        with pytest.raises(DataContractError):
            store.snapshot(_before(), as_of="2024-02-15", kind="forecast")
        # unknown kind still raises ForecastOSError
        with pytest.raises(ForecastOSError):
            store.snapshot(_before(), as_of="2024-02-15", kind="bogus")
