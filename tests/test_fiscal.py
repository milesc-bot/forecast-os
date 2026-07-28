"""Tests for FiscalCalendar: fiscal year/quarter labels, 4-4-5 weeks, features.

Convention under test (documented in ``preprocessing/fiscal.py``): a fiscal
year is labeled by the calendar year it ends in, so with ``start_month=2``
(Salesforce-style) FY27 runs Feb 2026 through Jan 2027. The 4-4-5 scheme
anchors each fiscal year at the first Monday on or after the 1st of
``start_month``; quarters are 13 Mon-Sun weeks grouped 4-4-5, and the extra
week of a 53-week year extends Q4's final 5-week period.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.exceptions import ForecastOSError
from forecast_os.preprocessing.fiscal import FiscalCalendar


def monthly_panel():
    return pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2026-01-31", periods=24, freq="ME"),
            "y": np.arange(24, dtype=float),
        }
    )


# -- constructor ---------------------------------------------------------------


def test_invalid_start_month_raises():
    for bad in (0, 13, 2.5, "feb"):
        with pytest.raises(ValueError):
            FiscalCalendar(start_month=bad)


def test_invalid_scheme_raises():
    with pytest.raises(ValueError):
        FiscalCalendar(scheme="5-4-4")


def test_constructor_args_stored_as_attributes():
    cal = FiscalCalendar(start_month=2, scheme="4-4-5")
    assert cal.start_month == 2
    assert cal.scheme == "4-4-5"


# -- calendar scheme -----------------------------------------------------------


def test_calendar_quarter_labels_salesforce_fy():
    cal = FiscalCalendar(start_month=2)
    ds = pd.DatetimeIndex(
        [
            "2026-01-31",  # last day of FY26 (Feb 2025 - Jan 2026)
            "2026-02-01",  # first day of FY27
            "2026-04-30",  # last day of FY27 Q1 (Feb-Apr)
            "2026-05-01",  # first day of FY27 Q2
            "2026-12-15",  # FY27 Q4 (Nov-Jan)
            "2027-01-15",  # still FY27 Q4
            "2027-02-01",  # first day of FY28
        ]
    )
    labels = cal.fiscal_quarter(ds)
    assert list(labels) == [
        "FY26Q4",
        "FY27Q1",
        "FY27Q1",
        "FY27Q2",
        "FY27Q4",
        "FY27Q4",
        "FY28Q1",
    ]


def test_calendar_fiscal_year_ints():
    cal = FiscalCalendar(start_month=2)
    fy = cal.fiscal_year(pd.DatetimeIndex(["2026-01-31", "2026-02-01", "2027-01-15"]))
    assert fy.dtype.kind == "i"
    assert list(fy) == [2026, 2027, 2027]


def test_calendar_quarter_end():
    cal = FiscalCalendar(start_month=2)
    ends = cal.quarter_end(
        pd.DatetimeIndex(["2027-01-15", "2026-02-15", "2026-12-15", "2026-01-31"])
    )
    assert list(ends) == [
        pd.Timestamp("2027-01-31"),
        pd.Timestamp("2026-04-30"),
        pd.Timestamp("2027-01-31"),
        pd.Timestamp("2026-01-31"),
    ]


def test_start_month_1_matches_calendar_year():
    cal = FiscalCalendar()  # start_month=1
    ds = pd.DatetimeIndex(["2026-03-15", "2026-12-31"])
    assert list(cal.fiscal_quarter(ds)) == ["FY26Q1", "FY26Q4"]
    assert list(cal.fiscal_year(ds)) == [2026, 2026]
    assert cal.quarter_end(ds)[0] == pd.Timestamp("2026-03-31")


# -- 4-4-5 scheme --------------------------------------------------------------


def test_445_year_boundary_and_labels():
    cal = FiscalCalendar(start_month=2, scheme="4-4-5")
    # FY27 anchor: first Monday of Feb 2026 = 2026-02-02; FY28 anchor 2027-02-01
    ds = pd.DatetimeIndex(
        [
            "2026-02-02",  # first day of FY27
            "2026-02-01",  # Sunday before the anchor -> last day of FY26
            "2027-01-31",  # last day of FY27 (52-week year)
            "2027-02-01",  # first day of FY28
        ]
    )
    assert list(cal.fiscal_quarter(ds)) == ["FY27Q1", "FY26Q4", "FY27Q4", "FY28Q1"]
    assert list(cal.fiscal_year(ds)) == [2027, 2026, 2027, 2028]
    assert cal.quarter_end(ds)[1] == pd.Timestamp("2026-02-01")
    assert cal.quarter_end(ds)[2] == pd.Timestamp("2027-01-31")


def test_445_weeks_per_quarter_sum_to_13():
    cal = FiscalCalendar(start_month=2, scheme="4-4-5")
    # FY27 is exactly 52 weeks: 2026-02-02 .. 2027-01-31
    ds = pd.date_range("2026-02-02", "2027-01-31", freq="D")
    assert len(ds) == 364
    labels = cal.fiscal_quarter(ds)
    weeks = cal.features(pd.DataFrame({"ds": ds}))["period_of_quarter"].to_numpy()
    periods = cal.fiscal_period(ds)
    for q in ("FY27Q1", "FY27Q2", "FY27Q3", "FY27Q4"):
        mask = labels == q
        assert mask.sum() == 91  # 13 weeks of 7 days
        assert set(weeks[mask]) == set(range(1, 14))
        # weeks grouped 4-4-5 into the quarter's three periods
        counts = {p: len(set(weeks[mask & (periods == p)])) for p in (1, 2, 3)}
        assert counts == {1: 4, 2: 4, 3: 5}


def test_445_53_week_year_extends_q4():
    cal = FiscalCalendar(start_month=2, scheme="4-4-5")
    # FY28 runs 2027-02-01 .. 2028-02-06 (371 days = 53 weeks)
    d = pd.DatetimeIndex(["2028-02-06"])  # last day of week 53
    assert list(cal.fiscal_quarter(d)) == ["FY28Q4"]
    assert cal.quarter_end(d)[0] == pd.Timestamp("2028-02-06")
    assert list(cal.fiscal_period(d)) == [3]  # extra week joins the 5-week period
    feats = cal.features(pd.DataFrame({"ds": d}))
    assert feats["period_of_quarter"].iloc[0] == 14  # week 14 of the long Q4


# -- features ------------------------------------------------------------------


FEATURE_COLS = [
    "fiscal_quarter_sin",
    "fiscal_quarter_cos",
    "period_of_quarter",
    "frac_of_quarter_elapsed",
    "days_to_quarter_end",
]


@pytest.mark.parametrize("scheme", ["calendar", "4-4-5"])
def test_features_columns_finite_and_bounded(scheme):
    cal = FiscalCalendar(start_month=2, scheme=scheme)
    out = cal.features(monthly_panel())
    for col in FEATURE_COLS:
        assert col in out.columns
        assert np.isfinite(out[col].to_numpy(dtype=float)).all()
    assert (out["fiscal_quarter_sin"].abs() <= 1.0).all()
    assert (out["fiscal_quarter_cos"].abs() <= 1.0).all()
    assert (out["frac_of_quarter_elapsed"] > 0.0).all()
    assert (out["frac_of_quarter_elapsed"] <= 1.0).all()
    assert (out["days_to_quarter_end"] >= 0).all()
    assert (out["period_of_quarter"] >= 1).all()
    # original columns untouched
    assert list(out["y"]) == list(monthly_panel()["y"])


def test_features_calendar_period_is_month_of_quarter():
    cal = FiscalCalendar(start_month=2)
    ds = pd.DatetimeIndex(["2026-02-10", "2026-03-10", "2026-04-10", "2026-05-10"])
    out = cal.features(pd.DataFrame({"ds": ds}))
    assert list(out["period_of_quarter"]) == [1, 2, 3, 1]


def test_features_quarter_end_day_has_zero_days_left():
    cal = FiscalCalendar(start_month=2)
    out = cal.features(pd.DataFrame({"ds": pd.DatetimeIndex(["2026-04-30"])}))
    assert out["days_to_quarter_end"].iloc[0] == 0
    assert out["frac_of_quarter_elapsed"].iloc[0] == pytest.approx(1.0)


# -- non-datetime inputs -------------------------------------------------------


def test_features_requires_datetime_ds():
    df = pd.DataFrame({"unique_id": "a", "ds": range(5), "y": np.ones(5)})
    with pytest.raises(ForecastOSError):
        FiscalCalendar().features(df)


def test_methods_reject_non_datetime():
    cal = FiscalCalendar(start_month=2)
    for bad in ([1, 2, 3], np.arange(4.0), ["2026-01-01"]):
        with pytest.raises(ForecastOSError):
            cal.fiscal_year(bad)


def test_features_requires_ds_column():
    with pytest.raises(ForecastOSError):
        FiscalCalendar().features(pd.DataFrame({"unique_id": ["a"], "y": [1.0]}))


# -- timezone-aware and NaT ds -------------------------------------------------


@pytest.mark.parametrize("scheme", ["calendar", "4-4-5"])
def test_tz_aware_ds_matches_naive_wall_clock(scheme):
    """A tz-aware ``ds`` used to blow up with raw numpy/stdlib errors
    (``ufunc 'subtract' cannot use operands with types dtype('O')...`` for
    ``features``, ``can't compare offset-naive and offset-aware datetimes`` for
    the whole 4-4-5 scheme) even though such a panel passes ``validate_panel``
    and works end to end through the rest of the library. Fiscal buckets are
    wall-clock buckets, so tz-aware input must give the same answer as the same
    local timestamps with the zone dropped."""
    naive = pd.date_range("2026-01-01", periods=40, freq="10D")
    aware = naive.tz_localize("US/Eastern")
    cal = FiscalCalendar(start_month=2, scheme=scheme)

    np.testing.assert_array_equal(cal.fiscal_year(aware), cal.fiscal_year(naive))
    np.testing.assert_array_equal(cal.fiscal_quarter(aware), cal.fiscal_quarter(naive))
    np.testing.assert_array_equal(cal.fiscal_period(aware), cal.fiscal_period(naive))
    pd.testing.assert_index_equal(cal.quarter_end(aware), cal.quarter_end(naive))

    y = np.arange(len(naive), dtype=float)
    out_aware = cal.features(pd.DataFrame({"unique_id": "a", "ds": aware, "y": y}))
    out_naive = cal.features(pd.DataFrame({"unique_id": "a", "ds": naive, "y": y}))
    for col in FEATURE_COLS:
        np.testing.assert_allclose(
            out_aware[col].to_numpy(dtype=float), out_naive[col].to_numpy(dtype=float)
        )
    # the caller's own ds column keeps its zone
    assert out_aware["ds"].dt.tz is not None


@pytest.mark.parametrize(
    ("zone", "day"),
    [("America/Havana", "2018-03-11"), ("America/Santiago", "2019-09-08")],
)
def test_midnight_dst_zone_raises_and_documented_workaround_succeeds(zone, day):
    """Pins the module docstring's one exception to tz-aware support: where a
    DST transition lands on midnight, that day's local midnight does not exist,
    so ``normalize()`` raises the tz backend's own nonexistent-time error (NOT a
    ForecastOSError). The documented workaround — drop the zone yourself,
    keeping the wall clock — must give the wall-clock answer.

    The exception TYPE is pandas-version dependent and must not be pinned to
    one: pandas 3 is zoneinfo-backed and raises ``ValueError``, pandas 2 is
    pytz-backed and raises ``pytz.exceptions.NonExistentTimeError``, which is
    not a ``ValueError`` subclass. Asserting ``ValueError`` alone passed on the
    dev venv and failed on the supported pandas floor — the same
    version-conditional break that shipped as 0.10.0.
    """
    cal = FiscalCalendar(start_month=2, scheme="4-4-5")
    aware = pd.to_datetime([f"{day} 12:00"]).tz_localize(zone)
    with pytest.raises(Exception) as excinfo:
        cal.fiscal_quarter(aware)
    # Whatever the backend calls it, it must not be our own error type (which
    # would imply we had handled the case) and must name the offending day.
    assert not isinstance(excinfo.value, ForecastOSError)
    assert day in str(excinfo.value)
    naive = aware.tz_localize(None)
    np.testing.assert_array_equal(
        cal.fiscal_quarter(naive), cal.fiscal_quarter(pd.to_datetime([f"{day} 12:00"]))
    )


@pytest.mark.parametrize("scheme", ["calendar", "4-4-5"])
def test_nat_ds_raises_instead_of_fabricating_a_fiscal_year(scheme):
    """``fiscal_year`` used to return 0 for a NaT row (NaN cast to int) and
    ``fiscal_quarter`` then died with a bare ``ValueError: Unknown format code
    'd'``. A missing timestamp falls in no fiscal quarter, so refuse it."""
    cal = FiscalCalendar(start_month=2, scheme=scheme)
    ds = pd.to_datetime(["2026-01-15", None, "2026-03-15"])
    for call in (cal.fiscal_year, cal.fiscal_quarter, cal.quarter_end, cal.fiscal_period):
        with pytest.raises(ForecastOSError, match="NaT"):
            call(ds)
    with pytest.raises(ForecastOSError, match="NaT"):
        cal.features(pd.DataFrame({"unique_id": "a", "ds": ds, "y": [1.0, 2.0, 3.0]}))
