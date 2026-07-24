"""Tests for the import-guarded TimeGPT foundation-model adapter.

Everything here runs offline. A ``FakeNixtlaClient`` stub stands in for
``nixtla.NixtlaClient`` — it records the arguments the adapter passes and
returns a canned frame in the exact shape the real TimeGPT API produces
(``unique_id, ds, TimeGPT, TimeGPT-lo-XX, TimeGPT-hi-XX``). The import guard
is exercised by monkeypatching the module-level ``_HAS_NIXTLA`` flag, never by
touching the (absent) SDK.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecast_os.adapters import timegpt_adapter as tg
from forecast_os.adapters.timegpt_adapter import TimeGPTAdapter, register_adapters
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.registry import _REGISTRY


class FakeNixtlaClient:
    """Minimal stand-in for ``nixtla.NixtlaClient`` exposing ``.forecast()``.

    Records every call and returns a Nixtla-shaped forecast frame: a point
    column named ``TimeGPT`` plus ``TimeGPT-lo-{level}`` / ``TimeGPT-hi-{level}``
    interval columns when ``level`` is requested.
    """

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    def forecast(self, df, h, freq=None, level=None, **kwargs):
        self.calls.append(
            {"df": df, "h": h, "freq": freq, "level": level, "kwargs": kwargs}
        )
        if self.fail:
            raise RuntimeError("boom from the hosted API")
        ids = list(dict.fromkeys(df["unique_id"].tolist()))
        last = pd.Timestamp(pd.to_datetime(df["ds"]).max())
        offset = freq if isinstance(freq, str) else "D"
        future = pd.date_range(last, periods=h + 1, freq=offset)[1:]
        rows = []
        for uid in ids:
            for i, ts in enumerate(future, start=1):
                row = {"unique_id": uid, "ds": ts, "TimeGPT": 100.0 + i}
                if level:
                    for lvl in level:
                        row[f"TimeGPT-lo-{lvl}"] = 100.0 + i - lvl
                        row[f"TimeGPT-hi-{lvl}"] = 100.0 + i + lvl
                rows.append(row)
        return pd.DataFrame(rows)


def _panel(freq: str = "D", n: int = 30) -> pd.DataFrame:
    ds = pd.date_range("2021-01-01", periods=n, freq=freq)
    frames = [
        pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.arange(n, dtype=float)})
        for uid in ("a", "b")
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot/restore the global registry so registration tests can't leak."""
    before = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(before)


# -- fit / predict happy path --------------------------------------------------


def test_fit_returns_self():
    adapter = TimeGPTAdapter(client=FakeNixtlaClient())
    assert adapter.fit(_panel()) is adapter


def test_fit_predict_maps_to_contract_columns():
    fake = FakeNixtlaClient()
    out = TimeGPTAdapter(client=fake).fit(_panel()).predict(6, level=[80, 90])

    assert list(out.columns[:3]) == ["unique_id", "ds", "yhat"]
    assert {"lo-80", "hi-80", "lo-90", "hi-90"} <= set(out.columns)
    assert (out.groupby("unique_id").size() == 6).all()
    assert set(out["unique_id"].unique()) == {"a", "b"}
    assert np.isfinite(out["yhat"]).all()

    # intervals bracket the point forecast and widen with the level.
    assert (out["lo-80"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["hi-80"]).all()
    assert (out["lo-90"] <= out["lo-80"]).all()
    assert (out["hi-90"] >= out["hi-80"]).all()

    # the exact (unique_id, ds, y) panel schema reached the client.
    call = fake.calls[-1]
    assert list(call["df"].columns[:3]) == ["unique_id", "ds", "y"]
    assert call["h"] == 6
    assert call["level"] == [80, 90]


def test_yhat_carries_the_point_column_values():
    fake = FakeNixtlaClient()
    out = TimeGPTAdapter(client=fake).fit(_panel()).predict(3)
    # FakeNixtlaClient sets TimeGPT = 100 + step for steps 1..h.
    for _, g in out.groupby("unique_id"):
        assert list(g["yhat"]) == [101.0, 102.0, 103.0]


def test_predict_without_level_returns_point_only():
    fake = FakeNixtlaClient()
    out = TimeGPTAdapter(client=fake).fit(_panel()).predict(4)
    assert list(out.columns) == ["unique_id", "ds", "yhat"]
    assert fake.calls[-1]["level"] is None


def test_level_is_sorted_before_the_client_call():
    fake = FakeNixtlaClient()
    TimeGPTAdapter(client=fake).fit(_panel()).predict(3, level=[95, 80])
    assert fake.calls[-1]["level"] == [80, 95]


# -- frequency handling --------------------------------------------------------


def test_freq_inferred_from_daily_data():
    fake = FakeNixtlaClient()
    TimeGPTAdapter(client=fake).fit(_panel(freq="D")).predict(3)
    assert fake.calls[-1]["freq"] == "D"


def test_explicit_freq_overrides_inference():
    fake = FakeNixtlaClient()
    TimeGPTAdapter(client=fake, freq="W").fit(_panel(freq="D")).predict(3)
    assert fake.calls[-1]["freq"] == "W"


# -- credentials / client resolution ------------------------------------------


def test_client_injection_works_without_nixtla(monkeypatch):
    # No SDK present, but an injected client makes the adapter fully usable.
    monkeypatch.setattr(tg, "_HAS_NIXTLA", False)
    out = TimeGPTAdapter(client=FakeNixtlaClient()).fit(_panel()).predict(3)
    assert len(out) == 6


def test_fit_without_key_or_client_raises_forecastoserror(monkeypatch):
    # Pretend the SDK is importable so construction succeeds without a client;
    # with no key anywhere, fit must raise a clear ForecastOSError.
    monkeypatch.setattr(tg, "_HAS_NIXTLA", True)
    monkeypatch.delenv("NIXTLA_API_KEY", raising=False)
    monkeypatch.delenv("TIMEGPT_API_KEY", raising=False)
    adapter = TimeGPTAdapter()
    with pytest.raises(ForecastOSError):
        adapter.fit(_panel())


def _install_fake_nixtla_module(monkeypatch, built: dict):
    module = types.ModuleType("nixtla")

    class _Client:
        def __init__(self, api_key=None):
            built["api_key"] = api_key

    module.NixtlaClient = _Client
    monkeypatch.setitem(sys.modules, "nixtla", module)
    return _Client


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(tg, "_HAS_NIXTLA", True)
    monkeypatch.delenv("NIXTLA_API_KEY", raising=False)
    monkeypatch.setenv("TIMEGPT_API_KEY", "sk-from-env")
    built: dict = {}
    client_cls = _install_fake_nixtla_module(monkeypatch, built)

    adapter = TimeGPTAdapter()
    adapter.fit(_panel())

    assert built["api_key"] == "sk-from-env"
    assert isinstance(adapter._client, client_cls)


def test_explicit_api_key_takes_precedence_over_env(monkeypatch):
    monkeypatch.setattr(tg, "_HAS_NIXTLA", True)
    monkeypatch.setenv("NIXTLA_API_KEY", "env-key")
    built: dict = {}
    _install_fake_nixtla_module(monkeypatch, built)

    TimeGPTAdapter(api_key="arg-key").fit(_panel())
    assert built["api_key"] == "arg-key"


def test_nixtla_api_key_preferred_over_timegpt_api_key(monkeypatch):
    monkeypatch.setattr(tg, "_HAS_NIXTLA", True)
    monkeypatch.setenv("NIXTLA_API_KEY", "primary")
    monkeypatch.setenv("TIMEGPT_API_KEY", "secondary")
    built: dict = {}
    _install_fake_nixtla_module(monkeypatch, built)

    TimeGPTAdapter().fit(_panel())
    assert built["api_key"] == "primary"


# -- import guard --------------------------------------------------------------


def test_construction_without_nixtla_or_client_raises_hint(monkeypatch):
    monkeypatch.setattr(tg, "_HAS_NIXTLA", False)
    with pytest.raises(ImportError, match=r"forecast-os\[timegpt\]"):
        TimeGPTAdapter()


# -- error wrapping / lifecycle ------------------------------------------------


def test_client_errors_wrapped_in_forecastoserror():
    adapter = TimeGPTAdapter(client=FakeNixtlaClient(fail=True)).fit(_panel())
    with pytest.raises(ForecastOSError):
        adapter.predict(3)


def test_predict_before_fit_raises():
    adapter = TimeGPTAdapter(client=FakeNixtlaClient())
    with pytest.raises(ForecastOSError):
        adapter.predict(3)


# -- registration --------------------------------------------------------------


def test_register_adapters_noop_without_nixtla(monkeypatch):
    monkeypatch.setattr(tg, "_HAS_NIXTLA", False)
    assert register_adapters() == []
    assert "timegpt" not in _REGISTRY


def test_register_adapters_registers_when_nixtla_present(monkeypatch):
    monkeypatch.setattr(tg, "_HAS_NIXTLA", True)
    assert register_adapters() == ["timegpt"]
    spec = _REGISTRY["timegpt"]
    assert spec.cls is TimeGPTAdapter
    assert spec.family == "adapter"


# -- clone / params ------------------------------------------------------------


def test_get_params_exposes_all_constructor_args():
    fake = FakeNixtlaClient()
    adapter = TimeGPTAdapter(
        api_key="k", model="timegpt-1-long-horizon", freq="MS", client=fake
    )
    assert adapter.get_params() == {
        "api_key": "k",
        "model": "timegpt-1-long-horizon",
        "freq": "MS",
        "client": fake,
    }


def test_clone_roundtrips_and_passes_client_through():
    fake = FakeNixtlaClient()
    adapter = TimeGPTAdapter(api_key="k", model="timegpt-1", freq="MS", client=fake)
    clone = adapter.clone()

    assert type(clone) is TimeGPTAdapter
    assert clone.get_params() == adapter.get_params()
    assert clone.client is fake  # client passes through by reference
    # cloned instance is independent and usable.
    out = clone.fit(_panel()).predict(3)
    assert len(out) == 6


def test_predict_forwards_model_variant_to_client():
    """The constructor's model= reaches client.forecast (not dropped)."""
    import pandas as pd

    from forecast_os.adapters.timegpt_adapter import TimeGPTAdapter

    class RecordingClient:
        def __init__(self):
            self.calls = []

        def forecast(self, df, h, **kwargs):
            self.calls.append(kwargs)
            uids = df["unique_id"].unique()
            rows = []
            for u in uids:
                for i in range(h):
                    rows.append({"unique_id": u, "ds": i, "TimeGPT": 1.0})
            return pd.DataFrame(rows)

    fc = RecordingClient()
    panel = pd.DataFrame({"unique_id": "a", "ds": range(6), "y": range(6)})
    TimeGPTAdapter(client=fc, model="timegpt-1-long-horizon").fit(panel).predict(2)
    assert fc.calls[0].get("model") == "timegpt-1-long-horizon"


def test_short_datetime_series_freq_falls_back_not_raw_valueerror():
    """A <3-point datetime panel infers a fallback freq instead of raising raw ValueError."""
    import pandas as pd

    from forecast_os.adapters.timegpt_adapter import TimeGPTAdapter

    class Stub:
        def forecast(self, df, h, **kwargs):
            return pd.DataFrame(
                {"unique_id": ["a"] * h, "ds": range(h), "TimeGPT": [1.0] * h}
            )

    panel = pd.DataFrame(
        {"unique_id": "a", "ds": pd.to_datetime(["2020-01-01", "2020-02-01"]), "y": [1.0, 2.0]}
    )
    out = TimeGPTAdapter(client=Stub()).fit(panel).predict(1)  # no raw ValueError
    assert len(out) == 1
