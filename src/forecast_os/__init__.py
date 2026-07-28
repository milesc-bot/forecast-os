"""forecast-os: an open-source forecasting engine.

Statistical, ML, and financial models behind one unified API — one data
contract ``(unique_id, ds, y)``, one model interface, one evaluation harness.

Importing this package registers every built-in model; see
:func:`list_models` for the catalog.
"""

from . import connectors, finance, gtm, preprocessing, snapshots
from .core import (
    BaseForecaster,
    DataContractError,
    ForecastOSError,
    IgnoredCovariatesWarning,
    NotFittedError,
    PerSeriesForecaster,
    get_model,
    list_models,
    load,
    register,
    to_panel,
    validate_panel,
)
from .datasets import generate_returns, generate_series, load_air_passengers
from .engine import ForecastEngine
from .evaluation import cross_validation, evaluate
from .models import *  # noqa: F401,F403  (registers built-in models, re-exports classes)
from .models import __all__ as _models_all
from .uncertainty import ConformalForecaster

__version__ = "0.10.1"

__all__ = [
    "__version__",
    # core
    "BaseForecaster",
    "PerSeriesForecaster",
    "register",
    "get_model",
    "list_models",
    "load",
    "validate_panel",
    "to_panel",
    "ForecastOSError",
    "DataContractError",
    "NotFittedError",
    "IgnoredCovariatesWarning",
    # engine & evaluation
    "ForecastEngine",
    "cross_validation",
    "evaluate",
    # uncertainty
    "ConformalForecaster",
    # data
    "generate_series",
    "generate_returns",
    "load_air_passengers",
    # subpackages
    "connectors",
    "finance",
    "gtm",
    "preprocessing",
    "snapshots",
    *_models_all,
]
