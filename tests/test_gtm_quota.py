"""Tests for quota attainment probability and pipeline coverage (gtm.quota)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.types import ID_COL, TIME_COL
from forecast_os.gtm import attainment_probability, pipeline_coverage

Z80 = float(stats.norm.ppf(0.9))


def _forecast_frame(sigmas_a=(10.0, 10.0, 10.0), sigmas_b=(10.0, 10.0, 10.0)):
    """Two 3-step forecasts with known per-step sigma encoded in the 80% band."""
    ds = pd.date_range("2024-04-01", periods=3, freq="MS")
    rows = []
    for uid, means, sigmas in [
        ("A", [100.0, 100.0, 100.0], sigmas_a),
        ("B", [50.0, 50.0, 50.0], sigmas_b),
    ]:
        for t, (m, s) in enumerate(zip(means, sigmas)):
            rows.append(
                {
                    ID_COL: uid,
                    TIME_COL: ds[t],
                    "yhat": m,
                    "lo-80": m - Z80 * s,
                    "hi-80": m + Z80 * s,
                }
            )
    return pd.DataFrame(rows)


class TestAttainmentProbability:
    def test_hand_computed_against_scipy_norm(self):
        res = attainment_probability(_forecast_frame(), quota=250.0, level=80)
        assert list(res.columns) == [ID_COL, "expected", "quota", "p_attain"]
        assert list(res[ID_COL]) == ["A", "B"]
        total_sigma = np.sqrt(3 * 10.0**2)

        row_a = res[res[ID_COL] == "A"].iloc[0]
        assert row_a["expected"] == pytest.approx(300.0)
        assert row_a["quota"] == 250.0
        assert row_a["p_attain"] == pytest.approx(
            float(stats.norm.sf(250.0, loc=300.0, scale=total_sigma)), rel=1e-9
        )
        assert row_a["p_attain"] > 0.9  # comfortably above quota

        row_b = res[res[ID_COL] == "B"].iloc[0]
        assert row_b["expected"] == pytest.approx(150.0)
        assert row_b["p_attain"] == pytest.approx(
            float(stats.norm.sf(250.0, loc=150.0, scale=total_sigma)), abs=1e-9
        )
        assert row_b["p_attain"] < 0.1  # far below quota

    def test_varying_step_sigmas_combine_in_quadrature(self):
        res = attainment_probability(
            _forecast_frame(sigmas_a=(10.0, 20.0, 30.0)), quota=310.0, level=80
        )
        total_sigma = np.sqrt(10.0**2 + 20.0**2 + 30.0**2)
        row_a = res[res[ID_COL] == "A"].iloc[0]
        assert row_a["p_attain"] == pytest.approx(
            float(stats.norm.sf(310.0, loc=300.0, scale=total_sigma)), rel=1e-9
        )

    def test_per_series_quota_dict(self):
        res = attainment_probability(_forecast_frame(), quota={"A": 250.0, "B": 120.0}, level=80)
        total_sigma = np.sqrt(300.0)
        row_b = res[res[ID_COL] == "B"].iloc[0]
        assert row_b["quota"] == 120.0
        assert row_b["p_attain"] == pytest.approx(
            float(stats.norm.sf(120.0, loc=150.0, scale=total_sigma)), rel=1e-9
        )
        assert row_b["p_attain"] > 0.9

    def test_per_series_quota_series(self):
        quota = pd.Series({"A": 250.0, "B": 120.0})
        res = attainment_probability(_forecast_frame(), quota=quota, level=80)
        assert res[res[ID_COL] == "B"]["quota"].iloc[0] == 120.0

    def test_zero_sigma_degenerates_to_step(self):
        fc = _forecast_frame(sigmas_a=(0.0, 0.0, 0.0), sigmas_b=(0.0, 0.0, 0.0))
        res = attainment_probability(fc, quota=250.0, level=80)
        assert res[res[ID_COL] == "A"]["p_attain"].iloc[0] == 1.0  # 300 >= 250
        assert res[res[ID_COL] == "B"]["p_attain"].iloc[0] == 0.0  # 150 < 250

    def test_nan_yhat_raises_naming_series(self):
        """Regression: NaN made total_sigma NaN, hitting the sigma==0 branch
        and returning a confident 0.0/1.0 instead of an error."""
        fc = _forecast_frame()
        fc.loc[fc[ID_COL] == "B", "yhat"] = np.nan
        with pytest.raises(ForecastOSError, match="'B'"):
            attainment_probability(fc, quota=250.0, level=80)

    def test_nan_interval_bound_raises_naming_series(self):
        fc = _forecast_frame()
        fc.iloc[0, fc.columns.get_loc("hi-80")] = np.nan  # first row belongs to A
        with pytest.raises(ForecastOSError, match="'A'"):
            attainment_probability(fc, quota=250.0, level=80)

    def test_missing_interval_columns_raises(self):
        fc = _forecast_frame().drop(columns=["lo-80", "hi-80"])
        with pytest.raises(ForecastOSError, match="level"):
            attainment_probability(fc, quota=250.0, level=80)

    def test_wrong_level_raises(self):
        with pytest.raises(ForecastOSError, match="90"):
            attainment_probability(_forecast_frame(), quota=250.0, level=90)

    def test_missing_quota_key_raises(self):
        with pytest.raises(ForecastOSError, match="quota"):
            attainment_probability(_forecast_frame(), quota={"A": 250.0}, level=80)


class TestPipelineCoverage:
    def test_scalar_ratio(self):
        assert pipeline_coverage(300.0, 100.0) == pytest.approx(3.0)
        assert isinstance(pipeline_coverage(300.0, 100.0), float)

    def test_series_pipeline_scalar_quota(self):
        pipe = pd.Series([100.0, 200.0], index=["A", "B"])
        cov = pipeline_coverage(pipe, 100.0)
        assert isinstance(cov, pd.Series)
        assert list(cov) == [1.0, 2.0]

    def test_dict_quota_aligns_by_key(self):
        pipe = pd.Series([300.0, 300.0], index=["A", "B"])
        cov = pipeline_coverage(pipe, {"A": 100.0, "B": 300.0})
        assert cov["A"] == pytest.approx(3.0)
        assert cov["B"] == pytest.approx(1.0)

    def test_zero_scalar_quota_raises(self):
        with pytest.raises(ValueError, match="quota"):
            pipeline_coverage(300.0, 0.0)
