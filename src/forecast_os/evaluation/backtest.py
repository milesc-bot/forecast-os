"""Walk-forward cross-validation (the evaluation backbone of the engine).

Expanding-window scheme, Nixtla-style output: for each of ``n_windows``
cutoffs, every model is re-fitted on data up to the cutoff and scored on the
next ``h`` observations. The output frame has one column per model (named by
``model.name``) plus ``unique_id, ds, cutoff, y`` and optional
``{model}-lo-{level}`` / ``{model}-hi-{level}`` interval columns.

When the panel carries extra numeric covariate columns, they are threaded
automatically: exog-capable models (``supports_exog=True``) are fitted on the
covariate panel and predict with an ``X_df`` built from the held-out test rows
(known future inputs), while other models run exactly as before with the
per-fit :class:`IgnoredCovariatesWarning` suppressed to avoid warning spam.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import pandas as pd

from ..core.base import BaseForecaster, IgnoredCovariatesWarning, _method_accepts
from ..core.exceptions import DataContractError
from ..core.registry import get_model
from ..core.types import ID_COL, REQUIRED_COLS, TARGET_COL, TIME_COL, validate_panel

__all__ = ["cross_validation"]


def _check_pred_sorted(pred: pd.DataFrame, model_name: str) -> None:
    """Raise unless every series' ``ds`` is strictly increasing.

    Predictions are aligned to test rows by per-series row position, so an
    out-of-order predict() output would silently scramble the merge.
    """
    for uid, g in pred.groupby(ID_COL, sort=False):
        ds = g[TIME_COL]
        if not ds.is_monotonic_increasing or ds.duplicated().any():
            raise DataContractError(
                f"{model_name}.predict() returned rows whose 'ds' is not "
                f"strictly increasing within series {uid!r}; cross_validation "
                f"aligns predictions by row position and requires sorted output"
            )


def _resolve_models(models: Sequence[BaseForecaster | str]) -> list[BaseForecaster]:
    resolved: list[BaseForecaster] = []
    for m in models:
        resolved.append(get_model(m) if isinstance(m, str) else m)
    names = [m.name for m in resolved]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model names in cross_validation: {names}")
    return resolved


def cross_validation(
    df: pd.DataFrame,
    models: Sequence[BaseForecaster | str],
    h: int,
    n_windows: int = 3,
    step_size: int | None = None,
    level: list[int] | None = None,
) -> pd.DataFrame:
    """Walk-forward CV of ``models`` on panel ``df``.

    Each window ``i`` (oldest first) trains on all rows up to its cutoff and
    tests on the following ``h`` rows; consecutive cutoffs are ``step_size``
    rows apart (default ``h``, i.e. non-overlapping test windows).
    """
    if not isinstance(h, int) or h < 1:
        raise ValueError(f"h must be a positive integer, got {h!r}")
    if not isinstance(n_windows, int) or n_windows < 1:
        raise ValueError(f"n_windows must be a positive integer, got {n_windows!r}")
    step_size = h if step_size is None else step_size
    if not isinstance(step_size, int) or step_size < 1:
        raise ValueError(f"step_size must be a positive integer, got {step_size!r}")

    df = validate_panel(df)
    resolved = _resolve_models(models)
    exog_cols = sorted(
        c
        for c in df.columns
        if c not in REQUIRED_COLS and pd.api.types.is_numeric_dtype(df[c])
    )

    sizes = df.groupby(ID_COL)[TARGET_COL].size()
    span = h + (n_windows - 1) * step_size
    too_short = sizes[sizes <= span]
    if len(too_short):
        raise DataContractError(
            f"series too short for h={h}, n_windows={n_windows}, "
            f"step_size={step_size} (need > {span} rows): "
            f"{list(too_short.index[:5])}"
        )

    n = df.groupby(ID_COL)[TARGET_COL].transform("size").to_numpy()
    pos = df.groupby(ID_COL).cumcount().to_numpy()

    out_frames = []
    for i in range(n_windows):
        offset = (n_windows - 1 - i) * step_size
        train_mask = pos < n - h - offset
        test_mask = (pos >= n - h - offset) & (pos < n - offset)
        train = df[train_mask]
        test = df.loc[test_mask, [ID_COL, TIME_COL, TARGET_COL]].copy()

        cutoffs = train.groupby(ID_COL)[TIME_COL].last()
        test.insert(2, "cutoff", test[ID_COL].map(cutoffs))
        test["_step"] = test.groupby(ID_COL).cumcount()

        # Held-out covariate rows are known future inputs for exog models.
        test_x = (
            df.loc[test_mask, [ID_COL, TIME_COL, *exog_cols]].copy() if exog_cols else None
        )

        for model in resolved:
            with warnings.catch_warnings():
                # Non-exog models on covariate panels warn once per fit; one
                # warning per (model, window) here would be pure spam.
                warnings.simplefilter("ignore", IgnoredCovariatesWarning)
                fitted = model.clone().fit(train)
            # Gate on what fit actually consumed, not on the class declaration:
            # a model declaring supports_exog=True whose _fit_series takes no X
            # ends up fitted without exog features, and passing X_df to it
            # would emit an IgnoredCovariatesWarning outside the suppression
            # block above (once per window).
            if (
                test_x is not None
                and getattr(fitted, "_exog_features", None)
                and _method_accepts(type(model), "predict", "X_df")
            ):
                pred = fitted.predict(h, level=level, X_df=test_x)
            else:
                pred = fitted.predict(h, level=level)
            _check_pred_sorted(pred, model.name)
            pred = pred.drop(columns=[TIME_COL])
            pred["_step"] = pred.groupby(ID_COL).cumcount()
            rename = {"yhat": model.name}
            for col in pred.columns:
                if col.startswith(("lo-", "hi-")):
                    rename[col] = f"{model.name}-{col}"
            pred = pred.rename(columns=rename)
            test = test.merge(pred, on=[ID_COL, "_step"], how="left")

        out_frames.append(test.drop(columns="_step"))

    out = pd.concat(out_frames, ignore_index=True)
    return out.sort_values([ID_COL, "cutoff", TIME_COL], kind="stable").reset_index(drop=True)
