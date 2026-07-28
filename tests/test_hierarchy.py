"""Tests for hierarchical aggregation and reconciliation.

The regression panel is engineered (seeded, verified offline) so that
independent AutoETS forecasts of the team total and the sum of its reps
disagree by well over 5% — exactly the incoherence the reconciler removes.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_os.core.base import BaseForecaster, PerSeriesForecaster
from forecast_os.core.exceptions import DataContractError, NotFittedError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL
from forecast_os.models.ets import AutoETS
from forecast_os.models.hierarchy import HierarchicalReconciler, aggregate_panel

METHODS = ("bottom_up", "top_down", "mint_ols")
H = 12


# -- panel builders -------------------------------------------------------------


def _tree_panel(extra_col=False):
    """Hand-built 2x2 tree: west/{alice,bob} and east/{carol,dan}, 4 steps."""
    ds = pd.date_range("2025-01-01", periods=4, freq="D")
    data = {
        "west/alice": [1.0, 2.0, 3.0, 4.0],
        "west/bob": [10.0, 20.0, 30.0, 40.0],
        "east/carol": [100.0, 200.0, 300.0, 400.0],
        "east/dan": [5.0, 5.0, 5.0, 5.0],
    }
    frames = []
    for uid, y in data.items():
        f = pd.DataFrame({ID_COL: uid, TIME_COL: ds, TARGET_COL: y})
        if extra_col:
            f["exog"] = 1.5
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _rep_panel(length=48):
    """5 reps under one team: 4 noisy reps whose small trends AutoETS misses
    per-series (SES wins) plus 1 cleanly trending rep (Holt wins), so the
    independent team forecast and the sum of rep forecasts disagree >5%."""
    rng = np.random.default_rng(2)
    ds = pd.date_range("2025-01-01", periods=length, freq="D")
    t = np.arange(length, dtype=float)
    frames = []
    for i in range(4):
        y = 20 + 0.8 * t + 4.0 * rng.standard_normal(length)
        frames.append(pd.DataFrame({ID_COL: f"team/rep-{i}", TIME_COL: ds, TARGET_COL: y}))
    y = 40 + 2.0 * t + 0.5 * rng.standard_normal(length)
    frames.append(pd.DataFrame({ID_COL: "team/rep-4", TIME_COL: ds, TARGET_COL: y}))
    return pd.concat(frames, ignore_index=True)


def _flat_panel():
    rng = np.random.default_rng(0)
    ds = pd.date_range("2025-01-01", periods=40, freq="D")
    frames = [
        pd.DataFrame({ID_COL: uid, TIME_COL: ds, TARGET_COL: lvl + rng.normal(size=40)})
        for uid, lvl in (("a", 10.0), ("b", 30.0))
    ]
    return pd.concat(frames, ignore_index=True)


def _series(pred, uid, col="yhat"):
    return pred.loc[pred[ID_COL] == uid].sort_values(TIME_COL)[col].to_numpy(dtype=float)


def _assert_coherent(pred, col="yhat", atol=1e-8, sep="/", total_id="total"):
    """Every parent equals the sum of its children, at every horizon step."""
    children: dict[str, list[str]] = {}
    for uid in pred[ID_COL].unique():
        if uid == total_id:
            continue
        parts = uid.split(sep)
        parent = total_id if len(parts) == 1 else sep.join(parts[:-1])
        children.setdefault(parent, []).append(uid)
    assert children, "no parent/child relations found in prediction"
    for parent, kids in children.items():
        got = np.sum([_series(pred, kid, col) for kid in kids], axis=0)
        np.testing.assert_allclose(_series(pred, parent, col), got, rtol=0, atol=atol)


# -- aggregate_panel ------------------------------------------------------------


def test_aggregate_panel_sums_2x2_tree():
    out = aggregate_panel(_tree_panel())
    assert set(out[ID_COL]) == {
        "west/alice", "west/bob", "east/carol", "east/dan", "west", "east", "total",
    }
    np.testing.assert_allclose(_series(out, "west", "y"), [11.0, 22.0, 33.0, 44.0])
    np.testing.assert_allclose(_series(out, "east", "y"), [105.0, 205.0, 305.0, 405.0])
    np.testing.assert_allclose(_series(out, "total", "y"), [116.0, 227.0, 338.0, 449.0])
    np.testing.assert_allclose(_series(out, "west/alice", "y"), [1.0, 2.0, 3.0, 4.0])
    counts = out.groupby(ID_COL).size()
    assert (counts == 4).all()


def test_aggregate_panel_drops_extra_columns():
    out = aggregate_panel(_tree_panel(extra_col=True))
    assert list(out.columns) == [ID_COL, TIME_COL, TARGET_COL]


def test_aggregate_panel_flat_panel_adds_total_only():
    out = aggregate_panel(_flat_panel())
    assert set(out[ID_COL]) == {"a", "b", "total"}
    np.testing.assert_allclose(
        _series(out, "total", "y"), _series(out, "a", "y") + _series(out, "b", "y")
    )


def test_aggregate_panel_misaligned_ds_raises():
    df = _tree_panel()
    shifted = df[TIME_COL] + pd.Timedelta(days=1)
    df.loc[df[ID_COL] == "west/bob", TIME_COL] = shifted[df[ID_COL] == "west/bob"]
    with pytest.raises(DataContractError, match="aligned"):
        aggregate_panel(df)


def test_aggregate_panel_ragged_ds_raises():
    df = _tree_panel()
    df = df.drop(df[df[ID_COL] == "east/dan"].index[-1])
    with pytest.raises(DataContractError, match="aligned"):
        aggregate_panel(df)


def test_aggregate_panel_total_id_collision_raises():
    df = _tree_panel()
    df.loc[df[ID_COL] == "east/dan", ID_COL] = "total"
    with pytest.raises(DataContractError, match="total"):
        aggregate_panel(df)


def test_aggregate_panel_leaf_also_interior_raises():
    df = _tree_panel()
    extra = pd.DataFrame(
        {
            ID_COL: "west",
            TIME_COL: pd.date_range("2025-01-01", periods=4, freq="D"),
            TARGET_COL: [1.0, 1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(DataContractError, match="ambiguous|leaf"):
        aggregate_panel(pd.concat([df, extra], ignore_index=True))


def test_aggregate_panel_custom_sep_and_total_id():
    df = _tree_panel()
    df[ID_COL] = df[ID_COL].str.replace("/", "|")
    out = aggregate_panel(df, sep="|", total_id="ALL")
    assert "ALL" in set(out[ID_COL])
    np.testing.assert_allclose(_series(out, "west", "y"), [11.0, 22.0, 33.0, 44.0])


# -- the incoherence regression -------------------------------------------------


def test_premise_independent_forecasts_disagree_over_5pct():
    """Sanity check of the engineered panel: independent AutoETS forecasts of
    team vs sum-of-reps must disagree by >5% (else the coherence tests below
    would be vacuous)."""
    panel = _rep_panel()
    leaf_pred = AutoETS().fit(panel).predict(H)
    sum_of_reps = leaf_pred.groupby(TIME_COL)["yhat"].sum().to_numpy()
    team_hist = panel.groupby(TIME_COL, as_index=False)[TARGET_COL].sum()
    team_hist[ID_COL] = "team"
    team = AutoETS().fit(team_hist).predict(H)["yhat"].to_numpy()
    rel = np.abs(team - sum_of_reps) / np.abs(team)
    assert rel.max() > 0.05


@pytest.mark.parametrize("method", METHODS)
def test_reconciled_output_is_coherent(method):
    panel = _rep_panel()
    model = HierarchicalReconciler(method=method)
    pred = model.fit(panel).predict(H)
    expected_ids = {f"team/rep-{i}" for i in range(5)} | {"team", "total"}
    assert set(pred[ID_COL]) == expected_ids
    assert (pred.groupby(ID_COL).size() == H).all()
    assert list(pred.columns[:3]) == [ID_COL, TIME_COL, "yhat"]
    assert np.isfinite(pred["yhat"]).all()
    _assert_coherent(pred, atol=1e-8)


def test_top_down_proportions_sum_to_one():
    pred = HierarchicalReconciler(method="top_down").fit(_rep_panel()).predict(H)
    total = _series(pred, "total")
    props = np.vstack([_series(pred, f"team/rep-{i}") for i in range(5)]) / total
    np.testing.assert_allclose(props.sum(axis=0), np.ones(H), rtol=0, atol=1e-8)


def test_mint_ols_holdout_accuracy_soft():
    """mint_ols on the team node is at least as accurate (generously) as the
    WORST of bottom-up and the independent total forecast on a holdout."""
    panel = _rep_panel(length=48 + H)
    cut = np.sort(panel[TIME_COL].unique())[48 - 1]
    train = panel[panel[TIME_COL] <= cut].reset_index(drop=True)
    hold = panel[panel[TIME_COL] > cut]
    actual_team = hold.groupby(TIME_COL)[TARGET_COL].sum().to_numpy()

    bu_team = _series(HierarchicalReconciler(method="bottom_up").fit(train).predict(H), "team")
    mint_team = _series(HierarchicalReconciler(method="mint_ols").fit(train).predict(H), "team")
    team_hist = train.groupby(TIME_COL, as_index=False)[TARGET_COL].sum()
    team_hist[ID_COL] = "team"
    indep_team = AutoETS().fit(team_hist).predict(H)["yhat"].to_numpy()

    def rmse(yhat):
        return float(np.sqrt(np.mean((yhat - actual_team) ** 2)))

    assert rmse(mint_team) <= 1.5 * max(rmse(bu_team), rmse(indep_team)) + 1e-9


class _MeanOrLarge(PerSeriesForecaster):
    """Forecasts the series mean, except near-zero-mean series forecast 100.0.

    Reproduces the verifier setup for the top-down degeneracy fix: signed
    leaves whose means (+-1) nearly cancel, so the aggregated total series is
    ~0 while its independent forecast is large.
    """

    def _fit_series(self, y):
        return {"mean": float(np.mean(y))}

    def _predict_series(self, state, h):
        val = state["mean"] if abs(state["mean"]) > 0.5 else 100.0
        return np.full(h, val)


def test_top_down_near_cancelling_leaves_equal_share_not_explode():
    """a=1.0 / b=-1.0+1e-9 leaves: the old absolute |colsum| < 1e-12 test saw
    the 1e-9 leaf sum as a legitimate denominator, producing ~1e9 proportions
    and ~1e11 leaf forecasts. The scale-relative test recognizes the step as
    degenerate and equal-shares the total instead."""
    ds = pd.date_range("2025-01-01", periods=8, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({ID_COL: "a", TIME_COL: ds, TARGET_COL: 1.0}),
            pd.DataFrame({ID_COL: "b", TIME_COL: ds, TARGET_COL: -1.0 + 1e-9}),
        ],
        ignore_index=True,
    )
    model = HierarchicalReconciler(model=_MeanOrLarge(), method="top_down")
    pred = model.fit(panel).predict(4)
    assert np.abs(pred["yhat"]).max() < 1e3  # no 1e11 blow-up
    total = _series(pred, "total")
    np.testing.assert_allclose(total, np.full(4, 100.0), rtol=0, atol=1e-8)
    for uid in ("a", "b"):  # equal shares of the total forecast
        np.testing.assert_allclose(_series(pred, uid), total / 2.0, rtol=0, atol=1e-8)


def test_mixed_type_unique_ids_raise_data_contract_error():
    """sorted() over mixed str/int ids used to crash with a raw TypeError;
    both entry points must raise a clear DataContractError instead."""
    ds = pd.date_range("2025-01-01", periods=4, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({ID_COL: "a", TIME_COL: ds, TARGET_COL: 1.0}),
            pd.DataFrame({ID_COL: 1, TIME_COL: ds, TARGET_COL: 2.0}),
        ],
        ignore_index=True,
    )
    with pytest.raises(DataContractError, match="uniformly typed strings"):
        aggregate_panel(panel)
    with pytest.raises(DataContractError, match="uniformly typed strings"):
        HierarchicalReconciler(method="bottom_up").fit(panel)


# -- intervals ------------------------------------------------------------------


class _AsymIntervalModel(BaseForecaster):
    """Point forecast = last y; hand-set per-node asymmetric interval offsets
    chosen so the raw mint_ols projection pushes leaf 'a''s 50% hi bound
    (10.6) outside its 90% hi bound (~10.53) — the cross-level crossing the
    nesting guard must repair."""

    _OFFSETS = {
        "total": {50: (-1.0, 1.0), 90: (-2.0, 2.0)},
        "a": {50: (-0.1, 0.1), 90: (-0.2, 0.2)},
        "b": {50: (-30.0, 1.0), 90: (-31.0, 2.0)},
    }

    def fit(self, df):
        self._state = {
            uid: (g[TIME_COL].to_numpy(), float(g[TARGET_COL].iloc[-1]))
            for uid, g in df.groupby(ID_COL, sort=True)
        }
        self._is_fitted = True
        return self

    def predict(self, h, level=None):
        frames = []
        for uid, (ds, last) in self._state.items():
            step = ds[1] - ds[0]
            future = ds[-1] + step * np.arange(1, h + 1)
            data = {ID_COL: uid, TIME_COL: future, "yhat": np.full(h, last)}
            for lvl in level or []:
                off_lo, off_hi = self._OFFSETS[uid][lvl]
                data[f"lo-{lvl}"] = np.full(h, last + off_lo)
                data[f"hi-{lvl}"] = np.full(h, last + off_hi)
            frames.append(pd.DataFrame(data))
        return pd.concat(frames, ignore_index=True)


def test_mint_ols_intervals_nested_across_levels():
    """The OLS projection maps each column independently, so per-level lo/hi
    re-ordering alone can leave a 50% band poking outside the 90% band;
    predict() must enforce monotone nesting across levels."""
    ds = pd.date_range("2025-01-01", periods=6, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({ID_COL: "a", TIME_COL: ds, TARGET_COL: 1.0}),
            pd.DataFrame({ID_COL: "b", TIME_COL: ds, TARGET_COL: 10.0}),
        ],
        ignore_index=True,
    )
    model = HierarchicalReconciler(model=_AsymIntervalModel(), method="mint_ols")
    pred = model.fit(panel).predict(3, level=[50, 90])
    # cross-level nesting: the 90% band contains the 50% band everywhere
    assert (pred["lo-90"] <= pred["lo-50"] + 1e-9).all()
    assert (pred["hi-50"] <= pred["hi-90"] + 1e-9).all()
    # per-level ordering still holds
    for lvl in (50, 90):
        assert (pred[f"lo-{lvl}"] <= pred["yhat"] + 1e-9).all()
        assert (pred["yhat"] <= pred[f"hi-{lvl}"] + 1e-9).all()
    # the crossing was really exercised: leaf 'a''s raw hi-90 (~10.53) sat
    # below its guarded hi-50 (10.6) and nesting widened it to match
    a = pred[pred[ID_COL] == "a"]
    np.testing.assert_allclose(a["hi-50"], np.full(3, 10.6), rtol=0, atol=1e-6)
    np.testing.assert_allclose(a["hi-90"], a["hi-50"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_intervals_present_and_ordered(method):
    pred = HierarchicalReconciler(method=method).fit(_rep_panel()).predict(6, level=[80, 95])
    for lvl in (80, 95):
        assert {f"lo-{lvl}", f"hi-{lvl}"} <= set(pred.columns)
        assert (pred[f"lo-{lvl}"] <= pred["yhat"] + 1e-9).all()
        assert (pred["yhat"] <= pred[f"hi-{lvl}"] + 1e-9).all()


@pytest.mark.parametrize("method", METHODS)
def test_interval_contract_is_ordering_and_nesting_not_additivity(method):
    """Pins what predict() promises for lo/hi when intervals are requested.

    The class docstring used to open with a blanket "coherent to
    floating-point accuracy: children sum to their parent", which reads as a
    promise about every column. It is only true of ``yhat``: under
    ``mint_ols`` the per-column OLS projection can invert a leaf band, and the
    ordering guard that repairs it necessarily breaks additivity (no per-row
    repair can keep both). The docstring is now scoped, and this test pins the
    three things that ARE guaranteed for every method — yhat coherence even
    when levels are requested, ``lo <= yhat <= hi``, and monotone nesting
    across levels — so the guard's real contract is covered rather than the
    one the old wording implied.
    """
    pred = HierarchicalReconciler(method=method).fit(_rep_panel()).predict(6, level=[50, 90])
    _assert_coherent(pred, col="yhat", atol=1e-8)
    for lvl in (50, 90):
        assert (pred[f"lo-{lvl}"] <= pred["yhat"] + 1e-9).all()
        assert (pred["yhat"] <= pred[f"hi-{lvl}"] + 1e-9).all()
    assert (pred["lo-90"] <= pred["lo-50"] + 1e-9).all()
    assert (pred["hi-50"] <= pred["hi-90"] + 1e-9).all()


# -- flat panels ----------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_flat_panel_produces_leaves_plus_total(method):
    pred = HierarchicalReconciler(method=method).fit(_flat_panel()).predict(5)
    assert set(pred[ID_COL]) == {"a", "b", "total"}
    np.testing.assert_allclose(
        _series(pred, "total"), _series(pred, "a") + _series(pred, "b"), rtol=0, atol=1e-8
    )


# -- forecaster conventions -----------------------------------------------------


def test_registered_with_expected_defaults():
    from forecast_os.core.registry import get_model

    model = get_model("reconciled")
    assert isinstance(model, HierarchicalReconciler)
    assert model.get_params() == {
        "model": "auto_ets", "method": "mint_ols", "sep": "/", "total_id": "total",
    }


def test_clone_deep_clones_instance_member():
    inst = AutoETS()
    model = HierarchicalReconciler(model=inst, method="bottom_up", sep=".", total_id="ALL")
    clone = model.clone()
    assert clone is not model
    assert isinstance(clone.model, AutoETS) and clone.model is not inst
    assert (clone.method, clone.sep, clone.total_id) == ("bottom_up", ".", "ALL")


def test_instance_member_not_mutated_by_fit():
    inst = AutoETS()
    HierarchicalReconciler(model=inst, method="bottom_up").fit(_flat_panel())
    assert not getattr(inst, "_is_fitted", False)


def test_unknown_string_member_fails_at_fit_not_construct():
    model = HierarchicalReconciler(model="_test_hier_never_registered")  # must not raise
    with pytest.raises(ValueError, match="unknown model"):
        model.fit(_flat_panel())


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        HierarchicalReconciler(method="middle_out")


def test_invalid_sep_raises():
    with pytest.raises(ValueError, match="sep"):
        HierarchicalReconciler(sep="")


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        HierarchicalReconciler().predict(3)


def test_predict_rejects_bad_h_and_level():
    model = HierarchicalReconciler(method="bottom_up").fit(_flat_panel())
    with pytest.raises(ValueError):
        model.predict(0)
    with pytest.raises(ValueError):
        model.predict(3, level=[150])
