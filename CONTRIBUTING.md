# Contributing to forecast-os

Thanks for your interest! Contributions of all kinds are welcome — new models,
bug fixes, docs, benchmarks.

## Development setup

```bash
git clone https://github.com/milesc-bot/forecast-os
cd forecast-os
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest        # full suite
ruff check src tests
```

## Adding a model

Every model in the engine satisfies one contract. Subclass
`PerSeriesForecaster` (univariate, fitted per series) or `BaseForecaster`
(meta-models), register it, and the shared contract test covers it
automatically:

```python
import numpy as np

from forecast_os import PerSeriesForecaster, register

@register("my_model", family="statistical")
class MyModel(PerSeriesForecaster):
    """One-line description shown in list_models()."""

    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param   # constructor args MUST be stored as-is (clone())

    def _fit_series(self, y):
        return {"last": y[-1], "fitted": np.r_[np.nan, y[:-1]]}

    def _predict_series(self, state, h):
        return np.full(h, state["last"])
```

Ground rules:

- Core stays numpy/pandas/scipy-only. Heavier backends belong in
  `forecast_os/adapters/` behind an optional extra.
- Every registered model must work with default constructor parameters
  (that's what `tests/test_contract.py` exercises) and produce finite
  forecasts with `level=[80]`.
- Seed all randomness through `np.random.default_rng(seed)`.
- Statistical models should include a test that recovers known parameters
  from synthetic data, not just smoke tests.

## README figures

The gallery images in `docs/assets/` are generated — never hand-edited:

```bash
python scripts/generate_figures.py
```

The script is deterministic (seeded simulated data) and writes a light and a
dark variant per figure; the README serves the matching one via `<picture>`.

## Pull requests

- One logical change per PR.
- `pytest` and `ruff check src tests` must pass.
- New behavior needs tests; changed behavior needs updated tests.
