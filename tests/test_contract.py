"""Contract tests: every registered model honors the panel + forecaster contract.

Parametrized over the full registry (populated by importing ``forecast_os``),
so third-party or newly added models are automatically covered.
"""

import numpy as np
import pandas as pd
import pytest

import forecast_os  # noqa: F401  (imports register all built-in models)
from forecast_os.core.base import MAX_HORIZON, BaseForecaster
from forecast_os.core.registry import _REGISTRY, get_model
from forecast_os.core.types import ID_COL, TIME_COL
from forecast_os.datasets.synthetic import generate_series

H = 8


def _contract_panel():
    return generate_series(
        n_series=3, length=80, freq="D", trend=0.3, seasonality=7, season_amp=5.0,
        noise=0.8, seed=21,
    )


def _all_model_names():
    names = sorted(_REGISTRY)
    assert names, "registry is empty — forecast_os import did not register models"
    return names


# Models whose fit() contract restricts inputs to a domain (e.g. retention
# fractions in [0, 1]) cannot fit the generic panel; they carry their own
# contract-equivalent tests next to their implementation.
_DOMAIN_PANEL_MODELS = {"retention_sbg"}  # see tests/test_gtm_retention.py


def _skip_if_domain_restricted(name):
    if name in _DOMAIN_PANEL_MODELS:
        pytest.skip(f"{name} has a domain-restricted input contract; covered by its own tests")


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_contract(name):
    _skip_if_domain_restricted(name)
    model = get_model(name)
    assert isinstance(model, BaseForecaster)

    df = _contract_panel()
    fitted = model.fit(df)
    assert fitted is model, "fit() must return self"

    pred = model.predict(H, level=[80])
    assert list(pred.columns[:3]) == [ID_COL, TIME_COL, "yhat"]
    assert {"lo-80", "hi-80"} <= set(pred.columns)

    # h rows per series, all finite point forecasts. Every training series must
    # be forecast; hierarchical models may add aggregate series (e.g. "total").
    counts = pred.groupby(ID_COL).size()
    assert set(counts.index) >= set(df[ID_COL].unique())
    assert (counts == H).all()
    assert np.isfinite(pred["yhat"]).all(), f"{name} produced non-finite forecasts"
    assert (pred["lo-80"] <= pred["hi-80"] + 1e-9).all()

    # future ds strictly after the training data, strictly increasing.
    # Aggregate series not present in training are checked against the panel max.
    last_train = df.groupby(ID_COL)[TIME_COL].max()
    for uid, g in pred.groupby(ID_COL):
        assert (g[TIME_COL] > last_train.get(uid, last_train.max())).all()
        assert g[TIME_COL].is_monotonic_increasing

    # clone() produces an equivalent unfitted instance
    clone = model.clone()
    assert type(clone) is type(model)
    assert clone.get_params() == model.get_params()
    pred2 = clone.fit(df).predict(H)
    pd.testing.assert_series_equal(pred["yhat"], pred2["yhat"], check_exact=False, atol=1e-6)


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_predict_before_fit_raises(name):
    from forecast_os.core.exceptions import ForecastOSError

    model = get_model(name)
    with pytest.raises(ForecastOSError):
        model.predict(3)


# -- negative predict-argument tests -------------------------------------------
# Fitted models are cached per module run so each registered model is fitted
# once and reused across all parametrized violations (keeps runtime low).

_FITTED_CACHE: dict[str, BaseForecaster] = {}


def _fitted_model(name: str) -> BaseForecaster:
    _skip_if_domain_restricted(name)
    if name not in _FITTED_CACHE:
        panel = generate_series(
            n_series=1, length=40, freq="D", trend=0.2, seasonality=7,
            season_amp=3.0, noise=0.5, seed=7,
        )
        _FITTED_CACHE[name] = get_model(name).fit(panel)
    return _FITTED_CACHE[name]


@pytest.mark.parametrize("bad_h", [0, -1])
@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_predict_rejects_nonpositive_h(name, bad_h):
    model = _fitted_model(name)
    with pytest.raises(ValueError):
        model.predict(bad_h)


@pytest.mark.parametrize("bad_level", [[0], [150]])
@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_predict_rejects_bad_level(name, bad_level):
    model = _fitted_model(name)
    with pytest.raises(ValueError):
        model.predict(3, level=bad_level)


# -- persistence ----------------------------------------------------------------


@pytest.mark.parametrize("name", _all_model_names())
def test_registered_model_save_load_roundtrip(name, tmp_path):
    from forecast_os.core.base import load

    _skip_if_domain_restricted(name)
    model = get_model(name).fit(_contract_panel())
    expected = model.predict(H, level=[80])

    path = tmp_path / f"{name}.pkl"
    model.save(path)
    loaded = load(path)

    assert type(loaded) is type(model)
    pd.testing.assert_frame_equal(loaded.predict(H, level=[80]), expected)


class TestHorizonIsBounded:
    """``h`` sizes the output array before anything else runs.

    Unbounded, one small request allocates without limit — over the REST
    surface a ~650-byte body was measured getting the server process
    OOM-killed at 4.5 GB, because ``np.full(h, ...)`` runs before any
    downstream validation.
    """

    @staticmethod
    def _fitted():
        return get_model("naive").fit(_contract_panel())

    def test_horizon_at_the_cap_is_accepted(self):
        pred = self._fitted().predict(MAX_HORIZON)
        # h is the per-series horizon, so the frame holds h rows for each id.
        assert (pred.groupby(ID_COL).size() == MAX_HORIZON).all()

    @pytest.mark.parametrize("bad_h", [MAX_HORIZON + 1, 1_000_000, 10**12])
    def test_horizon_beyond_the_cap_is_rejected(self, bad_h):
        with pytest.raises(ValueError, match="h must be an integer"):
            self._fitted().predict(bad_h)

    def test_rejection_happens_before_allocation(self):
        """A 1e12 horizon must raise, not attempt an 8 TB fill."""
        import time

        start = time.monotonic()
        with pytest.raises(ValueError):
            self._fitted().predict(10**12)
        assert time.monotonic() - start < 1.0


class TestFailedFitLeavesNoPartialState:
    """A fit() that raises must not leave the model fitted on a partial panel.

    Before this pin, ``PerSeriesForecaster.fit`` cleared ``_series_state`` at
    the top of the method and set ``_is_fitted`` only at the bottom, but never
    cleared ``_is_fitted`` on entry. Refitting an already-fitted model on a
    panel whose second series was too short raised, yet left ``_is_fitted``
    True with ``_series_state`` holding only the series processed before the
    failure — so ``predict()`` silently returned forecasts for a subset of the
    requested panel instead of raising. A failed fit must leave the object
    either unfitted or unchanged.
    """

    @staticmethod
    def _panel(sizes):
        rows = [
            (uid, i, float(i % 7) + 1.0)
            for uid, n in sizes.items()
            for i in range(n)
        ]
        return pd.DataFrame(rows, columns=["unique_id", "ds", "y"])

    def _short_second_series(self):
        from forecast_os.models.ets import HoltWinters

        model = HoltWinters(season_length=7)
        model.fit(self._panel({"a": 30, "b": 30}))
        with pytest.raises(Exception, match="at least"):
            model.fit(self._panel({"a": 30, "b": 3}))
        return model

    def test_failed_refit_does_not_leave_the_model_fitted(self):
        model = self._short_second_series()
        assert model._is_fitted is False

    def test_predict_after_a_failed_refit_raises_instead_of_forecasting_a_subset(self):
        from forecast_os.core.exceptions import NotFittedError

        model = self._short_second_series()
        with pytest.raises(NotFittedError):
            model.predict(2)

    def test_failed_first_fit_also_leaves_the_model_unfitted(self):
        from forecast_os.core.exceptions import NotFittedError
        from forecast_os.models.ets import HoltWinters

        model = HoltWinters(season_length=7)
        with pytest.raises(Exception, match="at least"):
            model.fit(self._panel({"a": 30, "b": 3}))
        assert model._is_fitted is False
        with pytest.raises(NotFittedError):
            model.predict(2)


class TestNonIntegerLevelIsRejected:
    """``level`` is declared ``list[int]``; a float was silently truncated.

    ``_check_level`` validated the raw value against ``0 < lvl < 100`` and then
    appended ``int(lvl)``. ``level=[99.9]`` therefore passed validation and was
    served as a 99% interval — ~22% narrower than requested, and labelled
    ``lo-99``/``hi-99``, a level the caller never asked for. Worse,
    ``level=[0.5]`` passed the guard and truncated to 0, producing zero-width
    ``lo-0``/``hi-0`` columns even though ``level=[0]`` is rejected by the same
    function. Silently re-labelling is not an option here, so a non-integral
    level must raise rather than be honoured under a different name.
    """

    @staticmethod
    def _fitted():
        return get_model("naive").fit(_contract_panel())

    @pytest.mark.parametrize("bad_level", [[99.9], [0.5], [80.5], [1e-9]])
    def test_non_integer_level_raises(self, bad_level):
        with pytest.raises(ValueError, match="whole number"):
            self._fitted().predict(3, level=bad_level)

    def test_integer_valued_float_is_still_accepted(self):
        """80.0 names exactly the 80% interval; only truncation was the bug."""
        pred = self._fitted().predict(3, level=[80.0])
        assert "lo-80" in pred.columns and "hi-80" in pred.columns

    def test_zero_width_lo_0_columns_are_no_longer_reachable(self):
        with pytest.raises(ValueError):
            self._fitted().predict(3, level=[0.5])


class TestFloatRoundingErrorLevelIsAccepted:
    """A whole-number level derived by float arithmetic must still be honoured.

    v0.9.0 accepted ``level=[100 * (1 - 0.34)]`` — the single most common way
    callers derive a level from an alpha — because it truncated. The
    non-integer guard replaced that with an exact ``float(lvl) != int(lvl)``
    test, which rejects 66 written as ``65.99999999999999``: a published
    package started raising on calls that worked before. Float error on a
    level in (0, 100) is bounded by ~1e-13, so a value integral to within
    1e-9 is a whole-number level and is rounded to it, while genuinely
    fractional levels (99.9, 80.5) keep raising.
    """

    @staticmethod
    def _fitted():
        return get_model("naive").fit(_contract_panel())

    @pytest.mark.parametrize(
        "alpha, expected",
        [(0.34, 66), (0.90, 10), (0.41, 59), (0.55, 45), (0.05, 95), (0.2, 80)],
    )
    def test_alpha_derived_level_is_accepted(self, alpha, expected):
        from forecast_os.core.base import _check_level

        assert _check_level([100 * (1 - alpha)]) == [expected]
        pred = self._fitted().predict(3, level=[100 * (1 - alpha)])
        assert f"lo-{expected}" in pred.columns and f"hi-{expected}" in pred.columns

    def test_rounding_error_does_not_truncate_downwards(self):
        """v0.9.0 truncated 65.99999999999999 to 65; it names the 66% band."""
        from forecast_os.core.base import _check_level

        assert _check_level([65.99999999999999]) == [66]
        assert _check_level([59.00000000000001]) == [59]

    def test_genuinely_fractional_levels_still_raise(self):
        from forecast_os.core.base import _check_level

        for bad in (99.9, 80.5, 0.5, 66.001):
            with pytest.raises(ValueError, match="whole number"):
                _check_level([bad])


class TestSigmaIsOverflowSafe:
    """Residual sigma overflowed to inf for large-but-finite targets.

    ``sigma = sqrt(sum(resid**2) / (n - 1))`` squares before reducing, so any
    residual above ~1.3e154 overflows float64. A panel of finite values around
    1e200 therefore produced ``sigma = inf`` and infinite prediction interval
    bounds around a perfectly finite ``yhat``. The reduction must be scale-safe
    so finite input yields finite intervals.
    """

    @staticmethod
    def _big_panel():
        from forecast_os.core.types import to_panel

        return to_panel(np.array([1e200, 2e200, 3e200, 2.5e200, 4e200]))

    def test_large_finite_panel_gives_finite_sigma(self):
        model = get_model("naive").fit(self._big_panel())
        sigma = model._series_state["series-0"]["_sigma"]
        assert np.isfinite(sigma)
        assert sigma == pytest.approx(1.2247448713915889e200, rel=1e-12)

    def test_large_finite_panel_gives_finite_intervals(self):
        pred = get_model("naive").fit(self._big_panel()).predict(2, level=[80])
        assert np.isfinite(pred[["yhat", "lo-80", "hi-80"]].to_numpy()).all()

    def test_no_overflow_warning_is_emitted(self):
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            get_model("naive").fit(self._big_panel())
