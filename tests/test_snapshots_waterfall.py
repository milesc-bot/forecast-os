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


class TestNullAmounts:
    """A null amount on an OPEN deal must be rejected, never reconciled away.

    A NaN/pd.NA amount used to slip through ``_require_deals`` (which only
    checked that the column was numeric) and then split the two code paths:
    ``_open_total`` summed it as 0 via pandas' skipna, while ``_classify``
    computed ``delta = amount_after - amount_before`` = NaN, which is neither
    ``> 0`` nor ``< 0``, so the deal fell through to ``unchanged``/``advanced``
    with amount 0.0. The bridge then failed to close (opening 100 + movements 0
    != closing 300) with NO NaN anywhere in the output to signal it — the one
    outcome a reconciliation tool must never produce. A missing amount on an
    open deal is missing money: the waterfall cannot know whether it is 0 or
    300, so it raises DataContractError naming the deal instead of guessing.

    The rejection is scoped to the amounts the bridge actually reads. A CLOSED
    deal's amount in the snapshot it is closed in is never read — ``_classify``
    returns ``-(amount_before if open_before else 0)`` for won/lost/removed and
    ``_open_total`` filters closed stages out of both anchors — so rejecting it
    would refuse an input the waterfall answers exactly, which the class below
    pins.
    """

    def test_null_amount_in_before_raises(self):
        before = _before()
        before.loc[before["opp_id"] == "D4", "amount"] = float("nan")
        with pytest.raises(DataContractError, match=r"null amount.*'D4'"):
            pipeline_waterfall(
                before, _after(), STAGES, won_stage=WON, lost_stage=LOST
            )

    def test_null_amount_in_after_raises(self):
        after = _after()
        after.loc[after["opp_id"] == "D9", "amount"] = float("nan")
        with pytest.raises(DataContractError, match=r"null amount.*'D9'"):
            pipeline_waterfall(
                _before(), after, STAGES, won_stage=WON, lost_stage=LOST
            )

    def test_nullable_integer_pd_na_amount_raises(self):
        before = _before()
        before["amount"] = pd.array(
            [100, 200, 150, None, 120, 80, 500, 90], dtype="Int64"
        )
        with pytest.raises(DataContractError, match=r"null amount.*'D4'"):
            pipeline_waterfall(
                before, _after(), STAGES, won_stage=WON, lost_stage=LOST
            )

    def test_summary_also_raises_rather_than_reporting_a_broken_bridge(self):
        # the exact repro: 100 -> 300 with the before amount missing used to
        # return opening 100 / unchanged 0 / closing 300 and no NaN at all
        before = pd.DataFrame(
            {"opp_id": ["D1"], "amount": [float("nan")], "stage": ["propose"]}
        )
        after = pd.DataFrame(
            {"opp_id": ["D1"], "amount": [300.0], "stage": ["propose"]}
        )
        with pytest.raises(DataContractError, match=r"null amount"):
            waterfall_summary(before, after, STAGES, won_stage=WON, lost_stage=LOST)


class TestNullAmountsOnClosedDealsAreAnswerable:
    """A null amount the bridge never reads must not abort the bridge.

    The null-amount guard was first written as a blanket "no NaN anywhere in
    either snapshot" check, which over-rejected: the sign convention makes a
    closed deal's amount in the snapshot where it is closed provably unread
    (won/lost/removed contribute ``-amount_before``, and only when the deal was
    OPEN before; ``_open_total`` excludes closed stages from both anchors). So
    a CRM that blanks the amount when a deal is marked lost — an everyday
    behaviour, and one the module docstring already accommodates by reporting
    the BEFORE value for won/lost — made the whole bridge unrunnable even
    though every number in it was exactly computable.

    These cases must reconcile, with the same numbers a fully-populated
    snapshot would give. The genuine bug stays fixed: a null on a deal that is
    open in that snapshot (see :class:`TestNullAmounts`) still raises.
    """

    def test_deal_lost_in_after_with_a_blanked_amount_still_bridges(self):
        before = pd.DataFrame(
            {"opp_id": ["a", "b"], "amount": [100.0, 300.0], "stage": ["propose", "qualify"]}
        )
        after = pd.DataFrame(
            {"opp_id": ["a", "b"], "amount": [float("nan"), 300.0], "stage": [LOST, "qualify"]}
        )
        summ = waterfall_summary(before, after, STAGES, won_stage=WON, lost_stage=LOST)
        assert not summ.isna().any().any()
        assert _cat(summ, "lost") == (1, -100.0)  # the BEFORE value left the pipeline
        assert float(summ["amount"].iloc[0]) == 400.0
        assert float(summ["amount"].iloc[-1]) == 300.0
        assert float(summ["running_total"].iloc[-2]) == pytest.approx(300.0)

    def test_deal_closed_won_in_both_snapshots_with_a_null_amount_still_bridges(self):
        before = pd.DataFrame(
            {"opp_id": ["a", "b"], "amount": [float("nan"), 300.0], "stage": [WON, "qualify"]}
        )
        after = pd.DataFrame(
            {"opp_id": ["a", "b"], "amount": [float("nan"), 300.0], "stage": [WON, "qualify"]}
        )
        summ = waterfall_summary(before, after, STAGES, won_stage=WON, lost_stage=LOST)
        assert not summ.isna().any().any()
        # 'a' was already closed before, so it contributes nothing either side
        assert _cat(summ, "won") == (1, 0.0)
        assert float(summ["amount"].iloc[0]) == 300.0
        assert float(summ["amount"].iloc[-1]) == 300.0

    def test_deal_created_and_won_in_period_with_a_null_amount_still_bridges(self):
        before = pd.DataFrame({"opp_id": ["b"], "amount": [300.0], "stage": ["qualify"]})
        after = pd.DataFrame(
            {"opp_id": ["a", "b"], "amount": [float("nan"), 300.0], "stage": [WON, "qualify"]}
        )
        summ = waterfall_summary(before, after, STAGES, won_stage=WON, lost_stage=LOST)
        assert not summ.isna().any().any()
        # never entered the open pipeline, so it moves the bridge by 0
        assert _cat(summ, "won") == (1, 0.0)
        assert float(summ["amount"].iloc[0]) == 300.0
        assert float(summ["amount"].iloc[-1]) == 300.0

    def test_null_on_a_closed_deal_matches_the_populated_answer_exactly(self):
        # blanking the amount of a deal that is closed in BOTH snapshots must
        # not change a single number in the bridge
        populated = _before(), _after()
        blanked_before, blanked_after = _before(), _after()
        # D2 is won and D3 lost in `after`; blank their after amounts
        blanked_after.loc[blanked_after["opp_id"].isin(["D2", "D3"]), "amount"] = float("nan")
        kwargs = dict(close_col="close_date", won_stage=WON, lost_stage=LOST)
        expected = waterfall_summary(*populated, STAGES, **kwargs)
        got = waterfall_summary(blanked_before, blanked_after, STAGES, **kwargs)
        pd.testing.assert_frame_equal(got, expected)

    def test_nullable_integer_na_on_a_closed_deal_still_bridges(self):
        # pd.NA in an Int64 column must survive the same narrowing (it reaches
        # the float conversion in _index_deals rather than being pre-rejected)
        before = pd.DataFrame({"opp_id": ["a", "b"], "stage": [WON, "qualify"]})
        before["amount"] = pd.array([None, 300], dtype="Int64")
        after = pd.DataFrame({"opp_id": ["a", "b"], "stage": [WON, "qualify"]})
        after["amount"] = pd.array([None, 300], dtype="Int64")
        summ = waterfall_summary(before, after, STAGES, won_stage=WON, lost_stage=LOST)
        assert not summ.isna().any().any()
        assert float(summ["amount"].iloc[-1]) == 300.0

    def test_no_closed_stages_declared_means_every_amount_participates(self):
        # with no won/lost stage there is no such thing as a closed deal, so
        # the narrowing must not open a hole: every null is still rejected
        before = pd.DataFrame({"opp_id": ["a"], "amount": [float("nan")], "stage": ["propose"]})
        after = pd.DataFrame({"opp_id": ["a"], "amount": [300.0], "stage": ["propose"]})
        with pytest.raises(DataContractError, match=r"null amount.*'a'"):
            waterfall_summary(before, after, STAGES)


def _random_snapshot(rng, ids, allow_null):
    """A random deal-grain snapshot over ``ids`` (some rows dropped)."""
    all_stages = [*STAGES, WON, LOST]
    rows = []
    for oid in ids:
        if rng.random() < 0.15:  # deal absent from this snapshot
            continue
        amount = round(float(rng.uniform(10.0, 1000.0)), 2)
        if allow_null and rng.random() < 0.15:
            amount = float("nan")
        rows.append(
            {
                "opp_id": oid,
                "amount": amount,
                "stage": all_stages[int(rng.integers(len(all_stages)))],
            }
        )
    return pd.DataFrame(rows, columns=["opp_id", "amount", "stage"])


def _has_open_null(frame):
    """True when a null amount sits on a row the bridge actually reads."""
    return bool((frame["amount"].isna() & ~frame["stage"].isin({WON, LOST})).any())


class TestReconciliationProperty:
    """The bridge identity holds for every input, or the input is rejected.

    ``opening + sum(movements) == closing`` is the whole point of the waterfall,
    so it is asserted over a spread of generated deal universes rather than one
    hand-built fixture. Null amounts are deliberately in the generated space:
    a null the bridge reads (one on an OPEN deal) must produce a loud
    DataContractError, never a quietly non-reconciling frame (the MUST-9
    regression). Nothing in between is an acceptable outcome.

    The rejection is also pinned from the other side, which is what the
    over-broad first version of the guard failed: a snapshot whose only nulls
    sit on closed deals must still bridge, and it must bridge to the identity.
    Accepting more inputs is only safe if the guarantee survives, so every
    accepted scenario is checked for both reconciliation and NaN-freedom.
    """

    @pytest.mark.parametrize("allow_null", [False, True])
    def test_identity_holds_or_raises(self, allow_null):
        import numpy as np

        rng = np.random.default_rng(20240727)
        saw_reconciled = False
        saw_rejected = False
        saw_accepted_with_null = False
        for _ in range(200):
            ids = [f"D{i}" for i in range(int(rng.integers(1, 10)))]
            before = _random_snapshot(rng, ids, allow_null)
            after = _random_snapshot(rng, ids, allow_null)
            reads_a_null = _has_open_null(before) or _has_open_null(after)
            try:
                summ = waterfall_summary(
                    before, after, STAGES, won_stage=WON, lost_stage=LOST
                )
            except DataContractError:
                # only a null the bridge reads may be rejected in this space
                assert allow_null
                assert reads_a_null
                saw_rejected = True
                continue

            # ... and a null it reads is never accepted
            assert not reads_a_null
            assert not summ.isna().any().any()
            if before["amount"].isna().any() or after["amount"].isna().any():
                saw_accepted_with_null = True

            opening = float(summ["amount"].iloc[0])
            closing = float(summ["amount"].iloc[-1])
            movements = summ.iloc[1:-1]
            assert opening + float(movements["amount"].sum()) == pytest.approx(closing)
            if len(movements):
                # the running total after the final movement lands on closing
                assert float(movements["running_total"].iloc[-1]) == pytest.approx(
                    closing
                )
            saw_reconciled = True

        assert saw_reconciled  # the generator produced real bridges
        assert saw_rejected == allow_null  # and read-nulls when it was asked to
        # nulls parked on closed deals reconcile rather than abort the bridge
        assert saw_accepted_with_null == allow_null


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
