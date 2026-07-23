"""Optional adapters for heavy third-party backends (statsforecast, neuralforecast).

Backends are never imported eagerly; call :func:`register_adapters` to register
whichever ones are installed.
"""

from .neuralforecast_adapter import NeuralForecastAdapter
from .neuralforecast_adapter import register_adapters as _register_neural
from .statsforecast_adapter import StatsForecastAdapter
from .statsforecast_adapter import register_adapters as _register_stats


def register_adapters() -> list[str]:
    """Register every installed optional backend; returns the registered names."""
    return _register_stats() + _register_neural()


__all__ = ["StatsForecastAdapter", "NeuralForecastAdapter", "register_adapters"]
