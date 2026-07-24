"""Base forecaster interfaces.

``BaseForecaster`` defines the contract every model in the registry satisfies:

- ``fit(df)`` takes a validated (unique_id, ds, y) panel and returns ``self``
- ``predict(h, level=None)`` returns a panel with columns
  ``unique_id, ds, yhat`` plus ``lo-{l}, hi-{l}`` per confidence level
- ``fitted_values()`` returns in-sample one-step fits for residual analysis
- ``get_params()/clone()`` support cross-validation refitting; constructor
  arguments MUST be stored as attributes of the same name.
- ``save(path)`` / module-level :func:`load` persist models as a pickle
  envelope carrying the package version, class path, and registry name.

``PerSeriesForecaster`` is the workhorse base for univariate models: it runs
the per-series groupby loop, generates future timestamps, and provides
Gaussian prediction intervals from in-sample residuals unless the subclass
supplies a model-specific ``_predict_sigma``.

Exogenous covariates are extra NUMERIC columns in the fit panel beyond
``(unique_id, ds, y)``. Models opt in with the ``supports_exog`` class
attribute; models that do not opt in emit :class:`IgnoredCovariatesWarning`
naming the ignored columns (never a silent ignore). Non-numeric extras
(ids, labels) are always silently ignored.
"""

from __future__ import annotations

import inspect
import pickle
import warnings
from abc import ABC, abstractmethod
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import DataContractError, ForecastOSError, NotFittedError
from .types import (
    ID_COL,
    REQUIRED_COLS,
    TARGET_COL,
    TIME_COL,
    future_ds,
    infer_step,
    validate_panel,
)


class IgnoredCovariatesWarning(UserWarning):
    """Numeric covariate columns were ignored by a model without exog support."""


_SIG_ACCEPTS_CACHE: dict[tuple[type, str, str], bool] = {}


def _method_accepts(cls: type, method: str, param: str) -> bool:
    """True when ``cls.<method>`` accepts a parameter named ``param``.

    Signature inspection is cached per (class, method, param) so subclasses
    written before covariate support keep working unmodified — optional
    arguments are only passed to methods that declare them (or ``**kwargs``).
    """
    key = (cls, method, param)
    if key not in _SIG_ACCEPTS_CACHE:
        try:
            params = inspect.signature(getattr(cls, method)).parameters
            accepts = param in params or any(
                p.kind is p.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            accepts = False
        _SIG_ACCEPTS_CACHE[key] = accepts
    return _SIG_ACCEPTS_CACHE[key]


def _package_version() -> str:
    from forecast_os import __version__

    return __version__


def load(path: str | PathLike) -> BaseForecaster:
    """Load a model saved with :meth:`BaseForecaster.save`.

    Returns the deserialized model. Emits a ``UserWarning`` when the file was
    written by a different forecast-os version and raises
    :class:`ForecastOSError` when ``path`` is not a forecast-os model file.
    """
    with open(path, "rb") as f:
        try:
            payload = pickle.load(f)
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
        ) as exc:
            raise ForecastOSError(
                f"{str(path)!r} is not a forecast-os model file (expected the "
                f"pickle envelope written by BaseForecaster.save): {exc}"
            ) from exc
    if not (isinstance(payload, dict) and isinstance(payload.get("model"), BaseForecaster)):
        raise ForecastOSError(
            f"{str(path)!r} is not a forecast-os model file (expected the "
            f"pickle envelope written by BaseForecaster.save)"
        )
    saved = payload.get("forecast_os_version")
    current = _package_version()
    if saved != current:
        warnings.warn(
            f"model was saved with forecast-os {saved} but is being loaded "
            f"with {current}; predictions may differ across versions",
            UserWarning,
            stacklevel=2,
        )
    return payload["model"]


def _check_level(level: list[int] | tuple[int, ...] | None) -> list[int]:
    if level is None:
        return []
    out = []
    for lvl in level:
        if not (0 < lvl < 100):
            raise ValueError(f"confidence levels must be in (0, 100), got {lvl}")
        out.append(int(lvl))
    return sorted(out)


class BaseForecaster(ABC):
    """Abstract base class for all forecast-os models."""

    #: Optional display name; used as the forecast column in cross-validation.
    alias: str | None = None

    #: Whether the model consumes exogenous covariates (extra numeric panel
    #: columns). Models that leave this False warn when covariates are present.
    supports_exog: bool = False

    @property
    def name(self) -> str:
        return self.alias or type(self).__name__

    def get_params(self) -> dict[str, Any]:
        """Constructor parameters, read back from same-named attributes."""
        sig = inspect.signature(type(self).__init__)
        params = {}
        for pname, p in sig.parameters.items():
            if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if not hasattr(self, pname):
                raise ForecastOSError(
                    f"{type(self).__name__} must store constructor argument "
                    f"{pname!r} as attribute self.{pname} (required for clone())"
                )
            params[pname] = getattr(self, pname)
        return params

    def clone(self) -> BaseForecaster:
        """A new unfitted instance with the same constructor parameters."""
        return type(self)(**self.get_params())

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseForecaster:
        """Fit the model on a (unique_id, ds, y) panel."""

    @abstractmethod
    def predict(self, h: int, level: list[int] | None = None) -> pd.DataFrame:
        """Forecast ``h`` steps ahead for every fitted series.

        Returned rows must be sorted ascending by ``ds`` within each
        ``unique_id``; downstream consumers align predictions positionally.
        Subclasses may accept additional optional keywords (e.g. the
        ``X_df`` covariate frame of :class:`PerSeriesForecaster`).
        """

    def save(self, path: str | PathLike) -> None:
        """Serialize this model to ``path`` as a pickle envelope.

        The envelope is a dict with keys ``forecast_os_version``, ``class``,
        ``registry_name``, and ``model``; read it back with :func:`load`,
        which warns when the saved version differs from the installed one.
        """
        payload = {
            "forecast_os_version": _package_version(),
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "registry_name": getattr(type(self), "registry_name", None),
            "model": self,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def fitted_values(self) -> pd.DataFrame:
        """In-sample fitted values as (unique_id, ds, y, fitted)."""
        raise NotImplementedError(f"{type(self).__name__} does not expose fitted values")

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"{self.name} is not fitted; call fit() first")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        except ForecastOSError:
            args = "..."
        return f"{type(self).__name__}({args})"


class PerSeriesForecaster(BaseForecaster):
    """Base class for univariate models fitted independently per series.

    Subclasses implement:

    - ``_fit_series(y: np.ndarray) -> dict`` returning the per-series state.
      If the state includes a ``"fitted"`` array (same length as ``y``, NaN
      for warm-up steps), it is used for residual-based intervals and
      :meth:`fitted_values`; otherwise a one-step-naive fallback is used.
    - ``_predict_series(state: dict, h: int) -> np.ndarray`` of length ``h``.
    - optionally ``_predict_sigma(state, h) -> np.ndarray`` for model-specific
      forecast standard errors (defaults to constant in-sample residual std).

    Exog-capable subclasses set ``supports_exog = True`` and accept two extra
    keywords: ``_fit_series(y, X=None)`` receives a per-series float array of
    covariates (rows aligned with ``y``), and ``_predict_series(state, h,
    X_future=None)`` receives the known future covariate rows. Both are passed
    only when the method declares the parameter, so existing subclasses keep
    working unmodified.
    """

    #: Minimum observations per series required by the model.
    min_train_size: int = 1

    def _resolve_exog_features(self, df: pd.DataFrame) -> list[str]:
        """Sorted numeric covariate columns this fit will consume (may be empty).

        Numeric columns beyond ``(unique_id, ds, y)`` are used as exogenous
        covariates when the class opts in via ``supports_exog`` and its
        ``_fit_series`` accepts an ``X`` parameter. Otherwise a single
        :class:`IgnoredCovariatesWarning` naming the columns is emitted per
        fit call. Non-numeric extras (ids, labels) are silently ignored.
        """
        numeric = sorted(
            c
            for c in df.columns
            if c not in REQUIRED_COLS and pd.api.types.is_numeric_dtype(df[c])
        )
        if not numeric:
            return []
        if self.supports_exog and _method_accepts(type(self), "_fit_series", "X"):
            return numeric
        warnings.warn(
            f"{self.name} does not support exogenous covariates; ignoring "
            f"numeric column(s) {numeric}. Drop them from the panel or use a "
            f"model with supports_exog=True.",
            IgnoredCovariatesWarning,
            stacklevel=3,
        )
        return []

    @staticmethod
    def _check_finite_exog(df: pd.DataFrame, features: list[str], where: str) -> None:
        """Raise :class:`DataContractError` when covariate columns hold NaN/inf.

        Non-finite covariate values would silently poison downstream linear
        algebra (NaN weights, NaN forecasts), so they are rejected up front,
        naming the offending column(s).
        """
        block = df[features].to_numpy(dtype=float)
        col_ok = np.isfinite(block).all(axis=0)
        if not col_ok.all():
            bad = [f for f, ok in zip(features, col_ok) if not ok]
            raise DataContractError(
                f"covariate column(s) {bad} in {where} contain non-finite "
                f"value(s) (NaN/inf); impute them first "
                f"(see forecast_os.preprocessing) or drop the column(s)"
            )

    def fit(self, df: pd.DataFrame) -> PerSeriesForecaster:
        df = validate_panel(df)
        self._exog_features: list[str] = self._resolve_exog_features(df)
        if self._exog_features:
            self._check_finite_exog(df, self._exog_features, "the fit panel")
        self._series_state: dict[Any, dict] = {}
        for uid, g in df.groupby(ID_COL, sort=True):
            y = g[TARGET_COL].to_numpy(dtype=float)
            if len(y) < self.min_train_size:
                raise ForecastOSError(
                    f"{self.name} requires at least {self.min_train_size} "
                    f"observations per series; series {uid!r} has {len(y)}"
                )
            if self._exog_features:
                x = g[self._exog_features].to_numpy(dtype=float)
                state = self._fit_series(y, X=x)
            else:
                state = self._fit_series(y)
            if not isinstance(state, dict):
                raise ForecastOSError(
                    f"{type(self).__name__}._fit_series must return a dict state"
                )
            fitted = state.get("fitted")
            if fitted is None:
                fitted = np.concatenate([[np.nan], y[:-1]])
            fitted = np.asarray(fitted, dtype=float)
            resid = (y - fitted)[~np.isnan(y - fitted)]
            # Uncentered rms with an n-1 denominator: systematic forecast bias
            # widens the intervals instead of being hidden by mean-centering.
            if resid.size >= 2:
                sigma = float(np.sqrt(np.sum(resid**2) / (resid.size - 1)))
            elif len(y) >= 3:
                d = np.diff(y)
                sigma = float(np.sqrt(np.sum(d**2) / (d.size - 1)))
            else:
                sigma = 1.0
            state["fitted"] = fitted
            state["_y"] = y
            state["_ds"] = g[TIME_COL].to_numpy()
            state["_last_ds"] = g[TIME_COL].iloc[-1]
            state["_step"] = infer_step(g[TIME_COL])
            state["_sigma"] = max(sigma, 1e-12)
            self._series_state[uid] = state
        self._is_fitted = True
        return self

    def _check_x_df(
        self, X_df: pd.DataFrame, h: int, features: list[str]
    ) -> dict[Any, np.ndarray]:
        """Validate ``X_df`` and split it into per-series ``(h, n_features)`` arrays.

        Requires the ``(unique_id, ds)`` columns plus every fitted covariate
        column, and exactly ``h`` rows per fitted series; rows are aligned to
        forecast steps by ``ds`` sort order within each series.
        """
        if not isinstance(X_df, pd.DataFrame):
            raise ForecastOSError(
                f"X_df must be a pandas DataFrame, got {type(X_df).__name__}"
            )
        missing = [c for c in (ID_COL, TIME_COL, *features) if c not in X_df.columns]
        if missing:
            raise ForecastOSError(
                f"X_df is missing column(s) {missing}; expected (unique_id, ds) "
                f"plus the fitted covariate column(s) {features}"
            )
        self._check_finite_exog(X_df, features, "X_df")
        xs = X_df.sort_values([ID_COL, TIME_COL], kind="stable")
        groups = dict(iter(xs.groupby(ID_COL, sort=True)))
        out: dict[Any, np.ndarray] = {}
        for uid in self._series_state:
            g = groups.get(uid)
            n_rows = 0 if g is None else len(g)
            if n_rows != h:
                raise ForecastOSError(
                    f"X_df must contain exactly {h} future row(s) for series "
                    f"{uid!r}; got {n_rows}"
                )
            out[uid] = g[features].to_numpy(dtype=float)
        return out

    def predict(
        self,
        h: int,
        level: list[int] | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast ``h`` steps ahead for every fitted series.

        ``X_df`` is a ``(unique_id, ds, <features>)`` frame of KNOWN future
        covariate values with exactly ``h`` rows per fitted series, aligned
        per series by ``ds`` sort order. It is required when the model was
        fitted with exogenous covariates; when the model has none, a supplied
        ``X_df`` is ignored with an :class:`IgnoredCovariatesWarning`.
        """
        self._check_is_fitted()
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError(f"h must be a positive integer, got {h!r}")
        levels = _check_level(level)
        features = getattr(self, "_exog_features", [])
        x_future: dict[Any, np.ndarray] | None = None
        if features:
            if X_df is None:
                raise ForecastOSError(
                    f"{self.name} was fitted with exogenous covariates "
                    f"{features}; predict() requires X_df with {int(h)} future "
                    f"rows per series"
                )
            x_future = self._check_x_df(X_df, int(h), features)
            if not _method_accepts(type(self), "_predict_series", "X_future"):
                raise ForecastOSError(
                    f"{type(self).__name__}._predict_series does not accept "
                    f"'X_future' although the model was fitted with covariates"
                )
        elif X_df is not None:
            warnings.warn(
                f"{self.name} was fitted without exogenous covariates; "
                f"X_df is ignored",
                IgnoredCovariatesWarning,
                stacklevel=2,
            )
        frames = []
        for uid, state in self._series_state.items():
            if x_future is not None:
                raw = self._predict_series(state, int(h), X_future=x_future[uid])
            else:
                raw = self._predict_series(state, int(h))
            mean = np.asarray(raw, dtype=float)
            if mean.shape != (h,):
                raise ForecastOSError(
                    f"{type(self).__name__}._predict_series returned shape "
                    f"{mean.shape}, expected ({h},)"
                )
            data: dict[str, Any] = {
                ID_COL: uid,
                TIME_COL: future_ds(state["_last_ds"], int(h), state["_step"]),
                "yhat": mean,
            }
            if levels:
                sigma = np.asarray(self._predict_sigma(state, int(h)), dtype=float)
                for lvl in levels:
                    z = float(stats.norm.ppf(0.5 + lvl / 200))
                    data[f"lo-{lvl}"] = mean - z * sigma
                    data[f"hi-{lvl}"] = mean + z * sigma
            frames.append(pd.DataFrame(data))
        return pd.concat(frames, ignore_index=True)

    def fitted_values(self) -> pd.DataFrame:
        self._check_is_fitted()
        frames = [
            pd.DataFrame(
                {
                    ID_COL: uid,
                    TIME_COL: state["_ds"],
                    TARGET_COL: state["_y"],
                    "fitted": state["fitted"],
                }
            )
            for uid, state in self._series_state.items()
        ]
        return pd.concat(frames, ignore_index=True)

    # -- hooks ---------------------------------------------------------------

    @abstractmethod
    def _fit_series(self, y: np.ndarray) -> dict:
        """Fit one series; return the state dict.

        Exog-capable subclasses declare ``_fit_series(y, X=None)``; ``X`` is
        the per-series float covariate matrix with rows aligned to ``y``.
        """

    @abstractmethod
    def _predict_series(self, state: dict, h: int) -> np.ndarray:
        """Point forecast of length ``h`` for one series.

        Exog-capable subclasses declare ``_predict_series(state, h,
        X_future=None)``; ``X_future`` holds the ``h`` known future covariate
        rows for the series.
        """

    def _predict_sigma(self, state: dict, h: int) -> np.ndarray:
        """Forecast standard errors; default is constant residual std."""
        return np.full(h, state["_sigma"])
