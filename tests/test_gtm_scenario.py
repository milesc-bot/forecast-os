"""Tests for driver-based scenario planning (gtm.scenario)."""

import math

import pandas as pd
import pytest

from forecast_os.core.exceptions import DataContractError, ForecastOSError
from forecast_os.core.types import ID_COL, TIME_COL
from forecast_os.gtm.funnel import propagate
from forecast_os.gtm.scenario import Scenario, compare_scenarios


def _baseline():
    # 1000 leads * 30% win * $25k ACV = $7.5M bookings; rep_count is carried
    # but does not enter the default projection.
    return Scenario(
        {"top_of_funnel": 1000, "win_rate": 0.30, "acv": 25000, "rep_count": 10}
    )


class TestConstruction:
    def test_drivers_property_returns_copy(self):
        s = _baseline()
        d = s.drivers
        assert d["win_rate"] == 0.30
        d["win_rate"] = 0.99  # mutating the copy must not affect the scenario
        assert s.drivers["win_rate"] == 0.30

    def test_constructor_arg_stored_as_same_named_attribute(self):
        s = _baseline()
        assert s.baseline_drivers["top_of_funnel"] == 1000.0

    def test_non_mapping_raises(self):
        with pytest.raises(DataContractError, match="mapping"):
            Scenario([("win_rate", 0.3)])

    def test_non_numeric_value_raises(self):
        with pytest.raises(DataContractError, match="number"):
            Scenario({"win_rate": "high"})

    def test_bool_value_rejected(self):
        with pytest.raises(DataContractError, match="number"):
            Scenario({"win_rate": True})

    def test_non_finite_value_raises(self):
        with pytest.raises(DataContractError, match="finite"):
            Scenario({"top_of_funnel": float("nan")})

    def test_non_string_key_raises(self):
        with pytest.raises(DataContractError, match="string"):
            Scenario({3: 0.5})


class TestProjectDefault:
    def test_baseline_projects_known_number(self):
        assert _baseline().project() == pytest.approx(7_500_000.0)

    def test_multiple_rate_drivers_multiply(self):
        s = Scenario(
            {"top_of_funnel": 1000, "lead_rate": 0.5, "win_rate": 0.4, "acv": 100}
        )
        assert s.project() == pytest.approx(1000 * 0.5 * 0.4 * 100)

    def test_acv_absent_returns_deal_count(self):
        s = Scenario({"top_of_funnel": 1000, "win_rate": 0.3})
        assert s.project() == pytest.approx(300.0)

    def test_missing_top_of_funnel_raises(self):
        with pytest.raises(ForecastOSError, match="top_of_funnel"):
            Scenario({"win_rate": 0.3}).project()

    def test_negative_top_of_funnel_raises(self):
        with pytest.raises(ForecastOSError, match="top_of_funnel"):
            Scenario({"top_of_funnel": -5, "win_rate": 0.3}).project()


class TestWithAndBump:
    def test_with_absolute_set(self):
        base = _baseline()
        alt = base.with_(win_rate=0.25)
        assert alt.project() == pytest.approx(6_250_000.0)

    def test_bump_relative_delta(self):
        base = _baseline()
        alt = base.bump(win_rate=-0.05)  # 0.30 - 0.05 = 0.25
        assert alt.project() == pytest.approx(6_250_000.0)

    def test_bump_positive_and_volume(self):
        base = _baseline()
        alt = base.bump(top_of_funnel=200)  # 1200 leads
        assert alt.project() == pytest.approx(1200 * 0.30 * 25000)

    def test_with_does_not_mutate_base(self):
        base = _baseline()
        base.with_(win_rate=0.9)
        assert base.project() == pytest.approx(7_500_000.0)

    def test_bump_does_not_mutate_base(self):
        base = _baseline()
        base.bump(win_rate=0.5)
        assert base.baseline_drivers["win_rate"] == 0.30

    def test_with_unknown_driver_raises(self):
        with pytest.raises(ForecastOSError, match="win_rat"):
            _baseline().with_(win_rat=0.25)

    def test_bump_unknown_driver_raises(self):
        with pytest.raises(ForecastOSError, match="foo"):
            _baseline().bump(foo=1.0)

    def test_returns_new_scenario_instance(self):
        base = _baseline()
        assert isinstance(base.with_(win_rate=0.25), Scenario)
        assert isinstance(base.bump(win_rate=0.01), Scenario)


class TestProjectStageChain:
    def _funnel_scenario(self):
        return Scenario(
            {
                "top_of_funnel": 1000,
                "acv": 100,
                "mql_rate": 0.5,
                "sql_rate": 0.4,
                "won_rate": 0.3,
            }
        )

    def test_stage_volumes_match_propagate(self):
        s = self._funnel_scenario()
        stages = {"mql": "mql_rate", "sql": "sql_rate", "won": "won_rate"}
        vols = s.stage_volumes(stages)
        top = pd.DataFrame({ID_COL: ["scenario"], TIME_COL: [0], "yhat": [1000.0]})
        expected = propagate(top, {"mql": 0.5, "sql": 0.4, "won": 0.3})
        assert vols["mql"] == pytest.approx(expected["mql"].iloc[0])
        assert vols["sql"] == pytest.approx(expected["sql"].iloc[0])
        assert vols["won"] == pytest.approx(expected["won"].iloc[0])
        assert list(vols.index) == ["mql", "sql", "won"]

    def test_project_stage_chain_applies_acv(self):
        s = self._funnel_scenario()
        stages = {"mql": "mql_rate", "sql": "sql_rate", "won": "won_rate"}
        # 1000 * 0.5 * 0.4 * 0.3 = 60 won deals * $100 = 6000
        assert s.project(stages=stages) == pytest.approx(6000.0)

    def test_stage_chain_accepts_literal_rates(self):
        s = self._funnel_scenario()
        assert s.project(stages={"mql": 0.5, "sql": 0.4, "won": 0.3}) == pytest.approx(6000.0)

    def test_with_changes_stage_chain_projection(self):
        s = self._funnel_scenario()
        stages = {"mql": "mql_rate", "sql": "sql_rate", "won": "won_rate"}
        alt = s.with_(won_rate=0.25)  # 1000*0.5*0.4*0.25 = 50 * 100 = 5000
        assert alt.project(stages=stages) == pytest.approx(5000.0)

    def test_unknown_driver_reference_raises(self):
        s = self._funnel_scenario()
        with pytest.raises(ForecastOSError, match="ghost_rate"):
            s.stage_volumes({"mql": "ghost_rate"})

    def test_rate_out_of_unit_interval_raises(self):
        s = self._funnel_scenario()
        with pytest.raises(ValueError, match="rate"):
            s.project(stages={"mql": 1.5})

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="stages"):
            self._funnel_scenario().stage_volumes({})


class TestCompareScenarios:
    def test_baseline_and_alternatives_with_deltas(self):
        base = _baseline()
        worst = base.bump(win_rate=-0.10)  # 0.20 -> 5.0M
        best = base.bump(win_rate=0.10)  # 0.40 -> 10.0M
        df = compare_scenarios(base, worst, best, labels=["base", "worst", "best"])
        assert list(df["label"]) == ["base", "worst", "best"]
        assert list(df.columns) == ["label", "projection", "delta", "pct_delta"]
        proj = dict(zip(df["label"], df["projection"]))
        assert proj["base"] == pytest.approx(7_500_000.0)
        assert proj["worst"] == pytest.approx(5_000_000.0)
        assert proj["best"] == pytest.approx(10_000_000.0)
        deltas = dict(zip(df["label"], df["delta"]))
        assert deltas["base"] == pytest.approx(0.0)
        assert deltas["worst"] == pytest.approx(-2_500_000.0)
        assert deltas["best"] == pytest.approx(2_500_000.0)
        pct = dict(zip(df["label"], df["pct_delta"]))
        assert pct["base"] == pytest.approx(0.0)
        assert pct["worst"] == pytest.approx(-1.0 / 3.0)
        assert pct["best"] == pytest.approx(1.0 / 3.0)

    def test_default_labels(self):
        base = _baseline()
        df = compare_scenarios(base, base.bump(win_rate=0.05))
        assert list(df["label"]) == ["baseline", "scenario_1"]

    def test_wrong_label_count_raises(self):
        base = _baseline()
        with pytest.raises(ValueError, match="labels"):
            compare_scenarios(base, base.bump(win_rate=0.05), labels=["only_one"])

    def test_non_scenario_raises(self):
        with pytest.raises(ForecastOSError, match="Scenario"):
            compare_scenarios(_baseline(), {"not": "a scenario"})

    def test_stages_passed_through(self):
        s = Scenario(
            {"top_of_funnel": 1000, "acv": 100, "mql_rate": 0.5, "won_rate": 0.3}
        )
        stages = {"mql": "mql_rate", "won": "won_rate"}
        alt = s.with_(won_rate=0.15)
        df = compare_scenarios(s, alt, labels=["base", "alt"], stages=stages)
        proj = dict(zip(df["label"], df["projection"]))
        assert proj["base"] == pytest.approx(1000 * 0.5 * 0.3 * 100)  # 15000
        assert proj["alt"] == pytest.approx(1000 * 0.5 * 0.15 * 100)  # 7500

    def test_zero_baseline_pct_delta_is_nan(self):
        base = Scenario({"top_of_funnel": 0, "win_rate": 0.3})
        df = compare_scenarios(base, base.bump(top_of_funnel=100))
        assert math.isnan(df["pct_delta"].iloc[1])
