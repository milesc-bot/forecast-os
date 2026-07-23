"""Split-conformal prediction intervals around any registered forecaster.

Per series, the last calibration block is held out; the member model (an
instance or a registry-name string resolved lazily inside :meth:`fit`) is
fitted on the head and its absolute holdout residuals become the
calibration scores. The member is then refitted on the full series, and
intervals at level ``l`` are its point forecast plus/minus the
finite-sample-corrected order statistic of that series' scores
(the ``ceil((n + 1) * l/100)``-th smallest score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseForecaster, _check_level
from ..core.exceptions import ForecastOSError
from ..core.registry import get_model, register
from ..core.types import ID_COL, TARGET_COL, TIME_COL, validate_panel

__all__ = ["ConformalForecaster"]


@register("conformal", family="ensemble")
class ConformalForecaster(BaseForecaster):
    """Wrap a member model with split-conformal prediction intervals.

    Calibration scores are pooled across forecast horizons, so the intervals
    target *marginal* (horizon-averaged) coverage and assume the calibration
    residuals are exchangeable with future residuals; per-step conditional
    coverage is not guaranteed.
    """

    def __init__(
        self,
        model="ses",
        level_calibration_fraction: float = 0.25,
        min_calibration: int = 8,
    ):
        if not 0 < level_calibration_fraction < 1:
            raise ValueError(
                f"level_calibration_fraction must be in (0, 1), "
                f"got {level_calibration_fraction}"
            )
        if min_calibration < 1:
            raise ValueError(f"min_calibration must be >= 1, got {min_calibration}")
        self.model = model
        self.level_calibration_fraction = level_calibration_fraction
        self.min_calibration = min_calibration

    def clone(self) -> ConformalForecaster:
        """Deep-clone a member instance; a registry-name string passes through."""
        params = self.get_params()
        if not isinstance(self.model, str):
            params["model"] = self.model.clone()
        return type(self)(**params)

    def fit(self, df: pd.DataFrame) -> ConformalForecaster:
        df = validate_panel(df)
        proto = get_model(self.model) if isinstance(self.model, str) else self.model.clone()

        n = df.groupby(ID_COL)[TARGET_COL].transform("size").to_numpy()
        n_cal = np.maximum(
            self.min_calibration, (self.level_calibration_fraction * n).astype(int)
        )
        if (n - n_cal < 1).any():
            raise ForecastOSError(
                f"{self.name}: series too short for calibration; every series needs "
                f"more than {self.min_calibration} observations"
            )
        pos = df.groupby(ID_COL).cumcount().to_numpy()
        head = df[pos < n - n_cal]
        cal = df.loc[pos >= n - n_cal, [ID_COL, TIME_COL, TARGET_COL]].copy()
        cal["_step"] = cal.groupby(ID_COL).cumcount()

        pred = proto.clone().fit(head).predict(int(n_cal.max()))
        pred["_step"] = pred.groupby(ID_COL).cumcount()
        cal = cal.merge(pred[[ID_COL, "_step", "yhat"]], on=[ID_COL, "_step"], how="left")

        self._abs_resid_: dict = {}
        for uid, g in cal.groupby(ID_COL, sort=True):
            r = np.abs(g[TARGET_COL].to_numpy(float) - g["yhat"].to_numpy(float))
            r = r[np.isfinite(r)]
            if r.size == 0:
                raise ForecastOSError(
                    f"{self.name}: no finite calibration residuals for series {uid!r}"
                )
            self._abs_resid_[uid] = r

        self._model_ = proto.clone().fit(df)
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        levels = _check_level(level)
        out = self._model_.predict(int(h))[[ID_COL, TIME_COL, "yhat"]].copy()
        for lvl in levels:
            q = {}
            for uid, scores in self._abs_resid_.items():
                # Finite-sample-corrected order statistic: without the
                # (n + 1)/n inflation the plain empirical quantile undercovers.
                n = scores.size
                q_level = min(1.0, (n + 1) * (lvl / 100) / n)
                q[uid] = float(np.quantile(scores, q_level, method="higher"))
            width = out[ID_COL].map(q).to_numpy(dtype=float)
            out[f"lo-{lvl}"] = out["yhat"] - width
            out[f"hi-{lvl}"] = out["yhat"] + width
        return out
