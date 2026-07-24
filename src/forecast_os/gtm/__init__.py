"""GTM semantic layer: CRM events, funnels, quotas, and cohort retention.

The bridge between go-to-market data (CRM exports, funnel events, cohort
tables) and the forecast-os engine. Importing this package registers the
``retention_sbg`` model.

Note: :func:`to_panel` here aggregates event records (unlike the array
helper of the same name in :mod:`forecast_os.core.types`); refer to it as
``forecast_os.gtm.to_panel``.
"""

from .anomaly import detect_anomalies
from .events import to_panel
from .funnel import conversion_rates, propagate, stage_panel
from .opportunities import DealScorer, weighted_pipeline
from .quota import attainment_probability, pipeline_coverage
from .retention import ShiftedBetaGeometric, cohort_panel
from .scenario import Scenario, compare_scenarios

__all__ = [
    "to_panel",
    "stage_panel",
    "conversion_rates",
    "propagate",
    "attainment_probability",
    "pipeline_coverage",
    "cohort_panel",
    "ShiftedBetaGeometric",
    "DealScorer",
    "weighted_pipeline",
    "Scenario",
    "compare_scenarios",
    "detect_anomalies",
]
