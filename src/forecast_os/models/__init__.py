"""Built-in forecasting models. Importing this package registers them all."""

from .arima import ARIMA, AutoARIMA
from .auto import AutoSelect
from .baselines import Drift, Naive, SeasonalNaive, WindowAverage
from .ensemble import Ensemble
from .ets import SES, AutoETS, Holt, HoltWinters
from .hierarchy import HierarchicalReconciler, aggregate_panel
from .intermittent import TSB, Croston
from .kalman import KalmanForecaster
from .ml import RidgeLag
from .theta import Theta

__all__ = [
    "HierarchicalReconciler",
    "aggregate_panel",
    "Croston",
    "TSB",
    "Naive",
    "SeasonalNaive",
    "Drift",
    "WindowAverage",
    "SES",
    "Holt",
    "HoltWinters",
    "AutoETS",
    "Theta",
    "ARIMA",
    "AutoARIMA",
    "KalmanForecaster",
    "RidgeLag",
    "Ensemble",
    "AutoSelect",
]
