"""Built-in :class:`~forecast_os.connectors.base.SchemaMapping` recipes.

One recipe per popular platform export, registered on import so they are
discoverable by name::

    from forecast_os.connectors import mappings  # registers the recipes
    apply_mapping(pd.read_csv("deals.csv"), "hubspot_deals", freq="W")

Renaming a column that is absent is a no-op, so each recipe's ``renames``
tolerates the platform's common column variants (UI report labels AND API
field names) by mapping several source names onto one canonical name.
CRM/billing recipes also pre-rename the platform's owner column to
``owner`` so an ``id_cols=("owner",)`` override splits the panel per owner
with no extra work. Every recipe can be adjusted at apply time — any
:func:`~forecast_os.gtm.to_panel` argument is an override.
"""

from .base import SchemaMapping, register_mapping

__all__ = [
    "AMPLITUDE_EVENTS",
    "GA4_EVENTS",
    "GENERIC_EVENTS",
    "HUBSPOT_DEALS",
    "MIXPANEL_EVENTS",
    "PIPEDRIVE_DEALS",
    "POSTHOG_EVENTS",
    "SALESFORCE_OPPORTUNITIES",
    "STRIPE_CHARGES",
    "STRIPE_INVOICES",
]

# -- CRM / billing: closed-won value summed per month --------------------------

SALESFORCE_OPPORTUNITIES = register_mapping(
    SchemaMapping(
        name="salesforce_opportunities",
        description=(
            "Salesforce opportunity report or API export: Closed Won Amount "
            "summed by Close Date month. Override id_cols=('owner',) to split "
            "the panel per opportunity owner."
        ),
        date_col="close_date",
        value_col="amount",
        renames={
            "Close Date": "close_date",  # report label
            "CloseDate": "close_date",  # API field
            "Amount": "amount",
            "Stage": "stage",  # report label
            "StageName": "stage",  # API field
            "Opportunity Owner": "owner",  # report label
            "Owner": "owner",  # API relationship
        },
        filters={"stage": ("Closed Won",)},
        freq="MS",
    )
)

HUBSPOT_DEALS = register_mapping(
    SchemaMapping(
        name="hubspot_deals",
        description=(
            "HubSpot deal export (API property names): closedwon amount summed "
            "by closedate month. Override id_cols=('owner',) to split the "
            "panel per deal owner."
        ),
        date_col="close_date",
        value_col="amount",
        renames={
            "closedate": "close_date",
            "hs_close_date": "close_date",
            "dealstage": "stage",
            "hubspot_owner_id": "owner",
        },
        filters={"stage": ("closedwon",)},
        freq="MS",
    )
)

PIPEDRIVE_DEALS = register_mapping(
    SchemaMapping(
        name="pipedrive_deals",
        description=(
            "Pipedrive deal export: won deal value summed by won_time month."
        ),
        date_col="close_date",
        value_col="amount",
        renames={
            "won_time": "close_date",
            "value": "amount",
            "owner_name": "owner",
        },
        filters={"status": ("won",)},
        freq="MS",
    )
)

STRIPE_INVOICES = register_mapping(
    SchemaMapping(
        name="stripe_invoices",
        description=(
            "Stripe invoice export: paid invoice amount_paid summed by created "
            "month. created is a Unix epoch timestamp in seconds. Stripe "
            "amounts are in minor units (cents) — divide by 100 for whole "
            "currency units."
        ),
        date_col="created_at",
        value_col="amount",
        renames={"created": "created_at", "amount_paid": "amount"},
        filters={"status": ("paid",)},
        freq="MS",
        date_unit="s",
    )
)

STRIPE_CHARGES = register_mapping(
    SchemaMapping(
        name="stripe_charges",
        description=(
            "Stripe charge export: succeeded charge amount summed by created "
            "month. created is a Unix epoch timestamp in seconds. Stripe "
            "amounts are in minor units (cents)."
        ),
        date_col="created_at",
        value_col="amount",
        renames={"created": "created_at"},
        filters={"status": ("succeeded",)},
        freq="MS",
        date_unit="s",
    )
)

# -- product analytics: daily event counts per event name ----------------------

POSTHOG_EVENTS = register_mapping(
    SchemaMapping(
        name="posthog_events",
        description=(
            "PostHog events export: daily row counts per event name, bucketed "
            "by timestamp."
        ),
        date_col="ts",
        id_cols=("event",),
        value_col=None,
        renames={"timestamp": "ts"},
        freq="D",
    )
)

GA4_EVENTS = register_mapping(
    SchemaMapping(
        name="ga4_events",
        description=(
            "GA4 (BigQuery) events export: daily row counts per event_name, "
            "bucketed by event_date (YYYYMMDD). Single-variant recipe: for "
            "event_timestamp (epoch microseconds) exports override with "
            "renames={'event_timestamp': 'ts', 'event_name': 'event'}, "
            "date_unit='us', date_format=None."
        ),
        date_col="ts",
        id_cols=("event",),
        value_col=None,
        renames={"event_date": "ts", "event_name": "event"},
        freq="D",
        date_format="%Y%m%d",
    )
)

MIXPANEL_EVENTS = register_mapping(
    SchemaMapping(
        name="mixpanel_events",
        description=(
            "Mixpanel events export: daily row counts per event name, bucketed "
            "by the time column (Unix epoch seconds)."
        ),
        date_col="ts",
        id_cols=("event",),
        value_col=None,
        renames={"time": "ts"},
        freq="D",
        date_unit="s",
    )
)

AMPLITUDE_EVENTS = register_mapping(
    SchemaMapping(
        name="amplitude_events",
        description=(
            "Amplitude events export: daily row counts per event_type, "
            "bucketed by event_time."
        ),
        date_col="ts",
        id_cols=("event",),
        value_col=None,
        renames={"event_time": "ts", "event_type": "event"},
        freq="D",
    )
)

GENERIC_EVENTS = register_mapping(
    SchemaMapping(
        name="generic_events",
        description=(
            "Any event log with a date or timestamp column: daily row counts "
            "as a single series."
        ),
        date_col="ts",
        id_cols=(),
        value_col=None,
        renames={"date": "ts", "timestamp": "ts"},
        freq="D",
    )
)
