"""Plug-and-play data plane: sources fetch records, mappings shape them.

Importing this package registers the built-in platform mapping recipes
(Salesforce, HubSpot, Pipedrive, Stripe, PostHog, GA4, Mixpanel, Amplitude —
see :func:`list_mappings`). REST sources need the ``[connectors]`` extra
(``pip install "forecast-os[connectors]"``).
"""

from . import mappings  # noqa: F401  (registers the built-in recipes)
from .base import (
    SchemaMapping,
    Source,
    apply_mapping,
    get_mapping,
    list_mappings,
    register_mapping,
)
from .files import CSVSource, ParquetSource
from .rest import (
    HubSpotSource,
    PostHogSource,
    RestSource,
    SalesforceSource,
    StripeSource,
)
from .sql import SQLSource

__all__ = [
    "SchemaMapping",
    "Source",
    "apply_mapping",
    "get_mapping",
    "list_mappings",
    "register_mapping",
    "CSVSource",
    "ParquetSource",
    "RestSource",
    "HubSpotSource",
    "PostHogSource",
    "StripeSource",
    "SalesforceSource",
    "SQLSource",
]
