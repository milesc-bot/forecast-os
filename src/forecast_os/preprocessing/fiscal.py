"""Fiscal calendars: fiscal year/quarter mapping and quarter-position features.

A :class:`FiscalCalendar` is a pure function of the ``ds`` column — it holds
no fitted state and works on any panel. Two schemes are supported:

- ``"calendar"``: fiscal quarters are calendar months regrouped so the fiscal
  year starts at ``start_month``. The fiscal year is labeled by the calendar
  year it **ends** in (Salesforce-style): with ``start_month=2``, FY27 runs
  Feb 2026 through Jan 2027, so 2027-01-15 falls in FY27 Q4.
- ``"4-4-5"``: the fiscal year starts on the first Monday **on or after** the
  1st of ``start_month`` (in the calendar year before the label year when
  ``start_month > 1``). Weeks are consecutive Mon-Sun blocks from that
  anchor; each quarter spans 13 weeks grouped into three periods of 4, 4,
  and 5 weeks. Anchors are 52 or 53 weeks apart; in a 53-week year the
  extra week extends Q4's final 5-week period (a 4-4-6 quarter).

All date arithmetic is on normalized (midnight) timestamps. Timezone-aware
``ds`` values are read as local wall clock — the zone is dropped after
normalizing, since a deal closing at 23:00 local on the last day of a quarter
belongs to that quarter — so :meth:`FiscalCalendar.quarter_end` always returns
naive timestamps. Non-datetime inputs and ``NaT`` raise
:class:`ForecastOSError`.

One class of zone is not supported: where a DST transition lands **on**
midnight, the local midnight of that day does not exist, and normalizing a
tz-aware ``ds`` covering it raises the timezone backend's own nonexistent-time
error rather than a :class:`ForecastOSError`. The exception type follows the
installed pandas: pandas 3 is zoneinfo-backed and raises
``ValueError: ... is a nonexistent time due to daylight savings time``, while
pandas 2.x is pytz-backed and raises ``pytz.exceptions.NonExistentTimeError``,
which is *not* a ``ValueError`` — so catch both, or catch ``Exception``
(America/Havana 2018-03-11, America/Santiago
2019-09-08, America/Sao_Paulo 2018-11-04, ...). Drop the zone yourself first —
``df["ds"].dt.tz_localize(None)`` keeps the wall clock these buckets are
defined on — for such panels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.exceptions import ForecastOSError
from ..core.types import TIME_COL

__all__ = ["FiscalCalendar"]

_WEEK = np.timedelta64(7, "D")
_DAY = np.timedelta64(1, "D")


def _make_dates(year: np.ndarray, month: np.ndarray, day: np.ndarray) -> pd.Series:
    """Assemble a datetime Series from integer year/month/day arrays."""
    return pd.to_datetime({"year": year, "month": month, "day": day})


class FiscalCalendar:
    """Map datetime ``ds`` values onto a company fiscal calendar.

    Parameters
    ----------
    start_month:
        Calendar month (1-12) the fiscal year starts in. The fiscal year is
        labeled by the calendar year it ends in, so ``start_month=2`` gives
        Salesforce-style fiscal years (FY27 = Feb 2026 - Jan 2027).
    scheme:
        ``"calendar"`` for month-based fiscal quarters or ``"4-4-5"`` for
        13-week quarters of Mon-Sun weeks grouped 4-4-5 (see module docs
        for the exact anchoring and 53-week conventions).
    """

    _SCHEMES = ("calendar", "4-4-5")

    def __init__(self, start_month: int = 1, scheme: str = "calendar"):
        if not isinstance(start_month, (int, np.integer)) or not 1 <= start_month <= 12:
            raise ValueError(f"start_month must be an integer in 1..12, got {start_month!r}")
        if scheme not in self._SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {self._SCHEMES}")
        self.start_month = int(start_month)
        self.scheme = scheme

    # -- public API ----------------------------------------------------------

    def fiscal_year(self, ds) -> np.ndarray:
        """Fiscal year (as an int array) each ``ds`` value falls in."""
        return self._parts(self._index(ds))["fy"].astype(int)

    def fiscal_quarter(self, ds) -> np.ndarray:
        """Fiscal-quarter labels like ``"FY27Q2"`` (two-digit year)."""
        p = self._parts(self._index(ds))
        return np.array([f"FY{fy % 100:02d}Q{q}" for fy, q in zip(p["fy"], p["q"])])

    def quarter_end(self, ds) -> pd.DatetimeIndex:
        """Timestamp of the last day of each value's fiscal quarter."""
        return pd.DatetimeIndex(self._parts(self._index(ds))["qend"])

    def fiscal_period(self, ds) -> np.ndarray:
        """Period (1..3) within the fiscal quarter.

        The calendar scheme returns the month-of-quarter; the 4-4-5 scheme
        returns the 4/4/5-week group the date's week belongs to (a 53rd week
        joins the final 5-week group).
        """
        return self._parts(self._index(ds))["fiscal_period"].astype(int)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append numeric fiscal features derived from the ``ds`` column.

        Adds ``fiscal_quarter_sin``/``fiscal_quarter_cos`` (cyclical quarter
        encoding), ``period_of_quarter`` (1-based month index within the
        quarter for the calendar scheme, 1-based week index for 4-4-5),
        ``frac_of_quarter_elapsed`` (day-of-quarter / days-in-quarter, in
        (0, 1]), and ``days_to_quarter_end`` (whole days, >= 0).
        """
        if TIME_COL not in df.columns:
            raise ForecastOSError(f"frame must contain the {TIME_COL!r} column")
        out = df.copy()
        idx = self._index(out[TIME_COL])
        p = self._parts(idx)
        angle = 2 * np.pi * (p["q"] - 1) / 4.0
        d = idx.to_numpy()
        days_in_quarter = ((p["qend"] - p["qstart"]) / _DAY).astype(int) + 1
        day_of_quarter = ((d - p["qstart"]) / _DAY).astype(int) + 1
        out["fiscal_quarter_sin"] = np.sin(angle)
        out["fiscal_quarter_cos"] = np.cos(angle)
        out["period_of_quarter"] = p["period_of_quarter"].astype(int)
        out["frac_of_quarter_elapsed"] = day_of_quarter / days_in_quarter
        out["days_to_quarter_end"] = ((p["qend"] - d) / _DAY).astype(int)
        return out

    # -- internals -----------------------------------------------------------

    def _index(self, ds) -> pd.DatetimeIndex:
        if not pd.api.types.is_list_like(ds):
            ds = [ds]
        idx = pd.Index(ds)
        if not pd.api.types.is_datetime64_any_dtype(idx):
            raise ForecastOSError(
                f"FiscalCalendar requires datetime values; got dtype {idx.dtype}"
            )
        idx = pd.DatetimeIndex(idx)
        # A missing timestamp falls in no fiscal quarter; without this the
        # int cast in fiscal_year turns it into year 0 and fiscal_quarter then
        # dies with a bare stdlib formatting error.
        n_nat = int(idx.isna().sum())
        if n_nat:
            raise ForecastOSError(
                f"FiscalCalendar requires a timestamp on every value; "
                f"got {n_nat} NaT. Drop or repair those rows first."
            )
        idx = idx.normalize()
        if idx.tz is not None:
            # Fiscal buckets are wall-clock buckets, and every anchor built
            # below is naive: keeping the zone makes idx.to_numpy() an object
            # array and every comparison a TypeError.
            idx = idx.tz_localize(None)
        return idx

    def _label_year(self, year: np.ndarray, month: np.ndarray) -> np.ndarray:
        """Fiscal year by the month-based rule (ends-in labeling)."""
        if self.start_month == 1:
            return year.copy()
        return np.where(month >= self.start_month, year + 1, year)

    def _parts(self, idx: pd.DatetimeIndex) -> dict:
        """Per-element fiscal decomposition: fy, q, qstart, qend, periods."""
        if self.scheme == "calendar":
            return self._calendar_parts(idx)
        return self._445_parts(idx)

    def _calendar_parts(self, idx: pd.DatetimeIndex) -> dict:
        month = np.asarray(idx.month)
        year = np.asarray(idx.year)
        months_into_fy = (month - self.start_month) % 12
        q = months_into_fy // 3 + 1
        month_of_q = months_into_fy % 3 + 1
        abs_month = year * 12 + (month - 1)
        qstart_abs = abs_month - (month_of_q - 1)
        qend_abs = qstart_abs + 2
        ones = np.ones(len(idx), dtype=int)
        qstart = _make_dates(qstart_abs // 12, qstart_abs % 12 + 1, ones)
        end_first = _make_dates(qend_abs // 12, qend_abs % 12 + 1, ones)
        qend = _make_dates(
            qend_abs // 12, qend_abs % 12 + 1, end_first.dt.days_in_month.to_numpy()
        )
        return {
            "fy": self._label_year(year, month),
            "q": q,
            "qstart": qstart.to_numpy(),
            "qend": qend.to_numpy(),
            "period_of_quarter": month_of_q,
            "fiscal_period": month_of_q,
        }

    def _anchor(self, fy: np.ndarray) -> np.ndarray:
        """4-4-5 fiscal-year start: first Monday on or after ``start_month`` 1st."""
        year = fy - 1 if self.start_month > 1 else fy
        n = len(year)
        first = _make_dates(year, np.full(n, self.start_month), np.ones(n, dtype=int))
        shift = (-first.dt.weekday.to_numpy()) % 7  # Monday == 0
        return (first + pd.to_timedelta(shift, unit="D")).to_numpy()

    def _445_parts(self, idx: pd.DatetimeIndex) -> dict:
        d = idx.to_numpy()
        fy = self._label_year(np.asarray(idx.year), np.asarray(idx.month))
        anchor = self._anchor(fy)
        # dates before the anchor belong to the previous fiscal year; the
        # month-based guess is off by at most one
        early = d < anchor
        if early.any():
            fy = np.where(early, fy - 1, fy)
            anchor = self._anchor(fy)
        week = ((d - anchor) / _DAY).astype(int) // 7  # 0-based week of FY
        q = np.minimum(week // 13, 3) + 1  # week 53 clamps into Q4
        week_of_q = week - 13 * (q - 1)  # 0-based, 13 in a 53-week year
        qstart = anchor + _WEEK * (13 * (q - 1))
        qend = np.where(q < 4, qstart + 13 * _WEEK - _DAY, self._anchor(fy + 1) - _DAY)
        return {
            "fy": fy,
            "q": q,
            "qstart": qstart,
            "qend": qend,
            "period_of_quarter": week_of_q + 1,
            "fiscal_period": np.where(week_of_q < 4, 1, np.where(week_of_q < 8, 2, 3)),
        }
