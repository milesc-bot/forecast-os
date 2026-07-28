"""Split-conformal prediction intervals around any registered forecaster.

Per series, the last calibration block is held out; the member model (an
instance or a registry-name string resolved lazily inside :meth:`fit`) is
fitted on the head and its absolute holdout residuals become the
calibration scores. The member is then refitted on the full series, and
intervals at level ``l`` are its point forecast plus/minus the
finite-sample-corrected order statistic of that series' scores
(the ``ceil((n + 1) * l/100)``-th smallest score).

That order statistic does not exist when ``ceil((n + 1) * l/100) > n`` — split
conformal has no finite valid bound at level ``l`` with only ``n`` calibration
residuals, and empirical coverage saturates at ``n / (n + 1)``. The widest
available band (the largest calibration score) is returned in that case, with a
``UserWarning`` naming the series and the level: level ``l`` needs at least
``ceil(l / (100 - l))`` residuals (9 for 90%, 19 for 95%, 99 for 99%).
"""

from __future__ import annotations

import warnings

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
        # A fit that raises part-way must not leave the object looking fitted.
        # fit() consumes the member twice — once on the calibration head, once
        # on the full panel — so a failure in the second fit used to leave the
        # PREVIOUS fit's _model_ paired with THIS fit's calibration scores, and
        # predict() went on to serve stale point forecasts inside freshly
        # calibrated widths. State is built in locals and published only once
        # both fits have succeeded.
        self._is_fitted = False
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

        abs_resid: dict = {}
        for uid, g in cal.groupby(ID_COL, sort=True):
            r = np.abs(g[TARGET_COL].to_numpy(float) - g["yhat"].to_numpy(float))
            r = r[np.isfinite(r)]
            if r.size == 0:
                raise ForecastOSError(
                    f"{self.name}: no finite calibration residuals for series {uid!r}"
                )
            abs_resid[uid] = r

        model = proto.clone().fit(df)
        self._abs_resid_: dict = abs_resid
        self._model_ = model
        self._is_fitted = True
        return self

    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        levels = _check_level(level)
        out = self._model_.predict(int(h))[[ID_COL, TIME_COL, "yhat"]].copy()
        capped: list[tuple[int, object, int]] = []
        for lvl in levels:
            q = {}
            for uid, scores in self._abs_resid_.items():
                # Finite-sample-corrected order statistic: without the
                # (n + 1)/n inflation the plain empirical quantile undercovers.
                n = scores.size
                if np.ceil((n + 1) * (lvl / 100)) > n:
                    # The required order statistic is past the largest score:
                    # min() below clamps to it, so coverage caps at n/(n + 1).
                    capped.append((lvl, uid, n))
                q_level = min(1.0, (n + 1) * (lvl / 100) / n)
                q[uid] = float(np.quantile(scores, q_level, method="higher"))
            width = out[ID_COL].map(q).to_numpy(dtype=float)
            out[f"lo-{lvl}"] = out["yhat"] - width
            out[f"hi-{lvl}"] = out["yhat"] + width
        if capped:
            lvl, uid, n = capped[0]
            extra = f" (and {len(capped) - 1} more series/level pairs)" if len(capped) > 1 else ""
            warnings.warn(
                f"{self.name}: level {lvl} needs >= {int(np.ceil(lvl / (100 - lvl)))} "
                f"calibration residuals but series {uid!r} has {n}{extra}; the "
                f"interval is capped at the largest residual, so its coverage "
                f"saturates at {n}/{n + 1} = {100 * n / (n + 1):.1f}%, not {lvl}%. "
                f"Use longer series, raise level_calibration_fraction/"
                f"min_calibration, or request a lower level.",
                stacklevel=2,
            )
        return out
