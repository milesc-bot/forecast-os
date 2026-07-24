"""Optional adapters for heavy third-party backends.

Adapters wrap statsforecast, neuralforecast, and Nixtla TimeGPT behind the
registry. Backends are never imported eagerly; call :func:`register_adapters`
to register whichever ones are installed.
"""

from .neuralforecast_adapter import NeuralForecastAdapter
from .neuralforecast_adapter import register_adapters as _register_neural
from .statsforecast_adapter import StatsForecastAdapter
from .statsforecast_adapter import register_adapters as _register_stats
from .timegpt_adapter import TimeGPTAdapter
from .timegpt_adapter import register_adapters as _register_timegpt


def register_adapters() -> list[str]:
    """Register every installed optional backend; returns the registered names."""
    return _register_stats() + _register_neural() + _register_timegpt()


__all__ = [
    "StatsForecastAdapter",
    "NeuralForecastAdapter",
    "TimeGPTAdapter",
    "register_adapters",
]
