"""Tests for the model registry (core.registry).

All throwaway registrations use names prefixed ``_test_`` and are removed
after each test so the global registry is left untouched.
"""

import numpy as np
import pytest

from forecast_os.core.base import PerSeriesForecaster
from forecast_os.core.registry import _REGISTRY, get_model, list_models, register


@pytest.fixture(autouse=True)
def _clean_registry():
    before = set(_REGISTRY)
    yield
    for name in set(_REGISTRY) - before:
        _REGISTRY.pop(name, None)


def _make_dummy(class_name="Dummy", doc="A dummy forecaster for registry tests."):
    class Dummy(PerSeriesForecaster):
        def _fit_series(self, y):
            return {}

        def _predict_series(self, state, h):
            return np.full(h, float(state["_y"][-1]))

    Dummy.__doc__ = doc
    Dummy.__name__ = class_name
    Dummy.__qualname__ = class_name
    return Dummy


def test_register_get_list_round_trip():
    cls = _make_dummy("RTDummy")
    register("_test_rt", family="baseline")(cls)

    model = get_model("_test_rt")
    assert isinstance(model, cls)
    assert cls.registry_name == "_test_rt"

    frame = list_models()
    row = frame[frame["name"] == "_test_rt"]
    assert len(row) == 1
    assert row["family"].iloc[0] == "baseline"
    assert row["class"].iloc[0] == "RTDummy"
    assert row["description"].iloc[0] == "A dummy forecaster for registry tests."


def test_register_returns_class_unchanged():
    cls = _make_dummy("DecoDummy")
    assert register("_test_deco", family="ml")(cls) is cls


def test_duplicate_name_different_class_raises():
    register("_test_dup", family="baseline")(_make_dummy("DupOne"))
    with pytest.raises(ValueError, match="already registered"):
        register("_test_dup", family="baseline")(_make_dummy("DupTwo"))


def test_duplicate_name_same_qualname_different_class_raises():
    # A DIFFERENT class that merely shares the qualname must NOT silently
    # replace the registered one; only identity (existing.cls is cls) may pass.
    register("_test_dup_qn", family="baseline")(_make_dummy("TwinDummy"))
    with pytest.raises(ValueError, match="already registered"):
        register("_test_dup_qn", family="baseline")(_make_dummy("TwinDummy"))


def test_same_class_re_register_is_noop():
    cls = _make_dummy("ReRegDummy")
    register("_test_rereg", family="baseline")(cls)
    register("_test_rereg", family="baseline")(cls)  # no raise
    assert isinstance(get_model("_test_rereg"), cls)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        register("_test_family", family="astrology")


def test_non_forecaster_class_raises_type_error():
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="BaseForecaster"):
        register("_test_notmodel", family="baseline")(NotAModel)


def test_get_model_unknown_name_lists_available():
    register("_test_known", family="baseline")(_make_dummy("KnownDummy"))
    with pytest.raises(ValueError, match="available.*_test_known"):
        get_model("_test_definitely_missing")


def test_get_model_forwards_params():
    class Param(PerSeriesForecaster):
        """Dummy with a constructor parameter."""

        def __init__(self, window=3):
            self.window = window

        def _fit_series(self, y):
            return {}

        def _predict_series(self, state, h):
            return np.full(h, float(state["_y"][-1]))

    register("_test_param", family="ml")(Param)
    assert get_model("_test_param").window == 3
    assert get_model("_test_param", window=9).window == 9


def test_list_models_family_filter():
    register("_test_base_a", family="baseline")(_make_dummy("BaseA"))
    register("_test_ml_b", family="ml")(_make_dummy("MlB"))

    ml = list_models(family="ml")
    assert (ml["family"] == "ml").all()
    assert "_test_ml_b" in set(ml["name"])
    assert "_test_base_a" not in set(ml["name"])


def test_list_models_columns_and_sorting():
    frame = list_models()
    assert list(frame.columns) == ["name", "family", "class", "description"]
    key = list(zip(frame["family"], frame["name"]))
    assert key == sorted(key)
