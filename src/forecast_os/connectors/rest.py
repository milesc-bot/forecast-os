"""REST API sources: pagination-aware fetching plus thin platform presets.

:class:`RestSource` turns a paginated JSON endpoint into a flat records
DataFrame: point it at a URL, say where the record list lives in the body
(``records_path``, a dot-path) and how the API paginates, and ``fetch()``
walks the pages and concatenates them. The platform subclasses
(:class:`HubSpotSource`, :class:`PostHogSource`, :class:`StripeSource`,
:class:`SalesforceSource`) are presets — endpoint, bearer auth, pagination
style, and a default schema-mapping name — so ``source.to_panel()`` is one
call end to end::

    HubSpotSource(token=...).to_panel()
    RestSource("https://api.example.com/deals", records_path="data").fetch()

``requests`` is an optional dependency; install it with
``pip install "forecast-os[connectors]"``.
"""

from __future__ import annotations

import warnings
from typing import Any
from urllib.parse import urljoin

import pandas as pd

from ..core.exceptions import ForecastOSError
from .base import SchemaMapping, Source

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover - requests is installed in dev
    _HAS_REQUESTS = False

__all__ = [
    "HubSpotSource",
    "PostHogSource",
    "RestSource",
    "SalesforceSource",
    "StripeSource",
]

#: Maximum characters of a non-2xx response body echoed into the error.
_SNIPPET_LEN = 200

#: Required pagination-dict keys per style (styles themselves are the keys).
_STYLE_KEYS = {
    "cursor": ("cursor_param", "cursor_path"),
    "offset": ("offset_param",),
    "page": ("page_param",),
    "next_url": ("next_path",),
    "has_more": (),
}


def _require_requests() -> None:
    """Raise a helpful ImportError when the optional ``requests`` is missing."""
    if not _HAS_REQUESTS:
        raise ImportError(
            "RestSource needs the optional 'requests' dependency; install it "
            'with pip install "forecast-os[connectors]"'
        )


def _check_pagination(pagination: dict | None) -> None:
    """Validate a pagination recipe dict at construction time."""
    if pagination is None:
        return
    if not isinstance(pagination, dict) or "style" not in pagination:
        raise ValueError("pagination must be a dict with a 'style' key")
    style = pagination["style"]
    if style not in _STYLE_KEYS:
        raise ValueError(
            f"unknown pagination style {style!r}; expected one of {sorted(_STYLE_KEYS)}"
        )
    missing = [k for k in _STYLE_KEYS[style] if k not in pagination]
    if missing:
        raise ValueError(f"pagination style {style!r} requires key(s) {missing}")


def _dot_get(obj: Any, path: str) -> Any:
    """Navigate a dot-path (``"a.b.c"``) into nested dicts; None when absent."""
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


class RestSource(Source):
    """Fetch records from a paginated JSON REST endpoint.

    Parameters
    ----------
    url : endpoint to GET.
    headers : HTTP headers sent with every request (e.g. auth).
    params : base query parameters sent with every request.
    records_path : dot-path into the JSON body to the list of records
        (e.g. ``"data.items"``). ``None`` means the body IS the list.
    pagination : how to walk pages — ``None`` fetches a single page, else a
        dict selecting a style::

            {"style": "cursor", "cursor_param": ..., "cursor_path": ...}
            {"style": "offset", "offset_param": ...}
            {"style": "page", "page_param": ...}
            {"style": "next_url", "next_path": ...}
            {"style": "has_more", "more_path": "has_more",
             "cursor_param": "starting_after", "id_field": "id"}

        ``cursor`` reads the next cursor from a body dot-path and stops when
        it is absent. ``offset`` / ``page`` bump their parameter and stop
        when a page returns fewer than ``page_size`` records. ``next_url``
        follows an absolute or relative link read from the body and stops
        when it is absent. ``has_more`` (Stripe-style) sends the last
        record's ``id_field`` as ``cursor_param`` while the body's
        ``more_path`` flag is true.
    page_size : records per full page — the short-page stop rule for the
        ``offset`` and ``page`` styles.
    max_pages : hard cap on pages fetched; hitting it warns and truncates.
    timeout : per-request timeout in seconds.
    mapping : default :class:`~forecast_os.connectors.base.SchemaMapping`
        (or registered mapping name) used by :meth:`to_panel`.
    session : a ``requests.Session``-like object (anything with a matching
        ``get``) for testing/customization; the default builds a
        ``requests.Session`` lazily on first fetch.
    """

    def __init__(
        self,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        records_path: str | None = None,
        pagination: dict | None = None,
        page_size: int = 100,
        max_pages: int = 100,
        timeout: float = 30,
        mapping: SchemaMapping | str | None = None,
        session: Any | None = None,
    ):
        if session is None:
            _require_requests()
        _check_pagination(pagination)
        self.url = url
        self.headers = headers
        self.params = params
        self.records_path = records_path
        self.pagination = pagination
        self.page_size = page_size
        self.max_pages = max_pages
        self.timeout = timeout
        self.mapping = mapping
        self.session = session
        self._default_session: Any | None = None

    def fetch(self) -> pd.DataFrame:
        """GET every page and concatenate the records into one flat DataFrame.

        Nested record fields are flattened by :func:`pandas.json_normalize`
        (``{"a": {"b": 1}}`` becomes column ``a.b``); an endpoint with no
        records yields an empty DataFrame.
        """
        frames: list[pd.DataFrame] = []
        url = self.url
        params: dict | None = dict(self.params) if self.params else {}
        pages = 0
        while True:
            body = self._get_json(url, params)
            records = self._extract_records(body)
            if records:
                frames.append(pd.json_normalize([self._prepare_record(r) for r in records]))
            pages += 1
            nxt = self._next_request(body, records, url, params)
            if nxt is None:
                break
            if pages >= self.max_pages:
                warnings.warn(
                    f"{type(self).__name__} hit max_pages={self.max_pages} with "
                    f"more pages available; results are truncated",
                    UserWarning,
                    stacklevel=2,
                )
                break
            url, params = nxt
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # -- hooks ---------------------------------------------------------------

    def _prepare_record(self, record: dict) -> dict:
        """Adjust one raw record before normalization (default: unchanged)."""
        return record

    # -- request plumbing ----------------------------------------------------

    def _get_session(self) -> Any:
        """The session to request with; builds the default lazily."""
        if self.session is not None:
            return self.session
        if self._default_session is None:
            self._default_session = requests.Session()
        return self._default_session

    def _get_json(self, url: str, params: dict | None) -> Any:
        """GET one page and return the parsed JSON body; raise on non-2xx."""
        resp = self._get_session().get(
            url, headers=self.headers, params=params or None, timeout=self.timeout
        )
        if not 200 <= resp.status_code < 300:
            text = resp.text
            snippet = text[:_SNIPPET_LEN] + ("..." if len(text) > _SNIPPET_LEN else "")
            raise ForecastOSError(f"GET {url} failed with HTTP {resp.status_code}: {snippet}")
        return resp.json()

    def _extract_records(self, body: Any) -> list:
        """The list of records inside one page's body, via ``records_path``."""
        data = body if self.records_path is None else _dot_get(body, self.records_path)
        if not isinstance(data, list):
            where = f"at {self.records_path!r} in" if self.records_path else "as"
            raise ForecastOSError(
                f"expected a list of records {where} the response body; "
                f"got {type(data).__name__}"
            )
        return data

    def _next_request(
        self, body: Any, records: list, url: str, params: dict | None
    ) -> tuple[str, dict | None] | None:
        """The ``(url, params)`` of the next page, or None when done."""
        if self.pagination is None:
            return None
        p = self.pagination
        style = p["style"]
        if style == "cursor":
            cursor = _dot_get(body, p["cursor_path"])
            if cursor in (None, ""):
                return None
            return url, {**(params or {}), p["cursor_param"]: cursor}
        if style == "offset":
            if len(records) < self.page_size:
                return None
            offset = int((params or {}).get(p["offset_param"], 0)) + self.page_size
            return url, {**(params or {}), p["offset_param"]: offset}
        if style == "page":
            if len(records) < self.page_size:
                return None
            page = int((params or {}).get(p["page_param"], 1)) + 1
            return url, {**(params or {}), p["page_param"]: page}
        if style == "next_url":
            nxt = _dot_get(body, p["next_path"])
            if not nxt:
                return None
            return urljoin(url, nxt), None
        # style == "has_more" (styles are validated at construction)
        if not _dot_get(body, p.get("more_path", "has_more")) or not records:
            return None
        id_field = p.get("id_field", "id")
        last = records[-1]
        if not isinstance(last, dict) or id_field not in last:
            keys = sorted(last) if isinstance(last, dict) else f"a {type(last).__name__}"
            raise ForecastOSError(
                f"has_more pagination needs field {id_field!r} on each record "
                f"to build the next-page cursor; the last record has {keys}"
            )
        return url, {**(params or {}), p.get("cursor_param", "starting_after"): last[id_field]}


class HubSpotSource(RestSource):
    """HubSpot CRM objects (deals by default) via the v3 objects API.

    Pages with the ``paging.next.after`` cursor and flattens each record's
    ``properties`` dict into top-level columns (alongside ``id`` /
    ``createdAt``) so mappings can reference property names directly.
    Default mapping: ``"hubspot_deals"``.
    """

    def __init__(
        self,
        token: str,
        object: str = "deals",
        properties: tuple[str, ...] = ("amount", "dealstage", "closedate"),
        base_url: str = "https://api.hubapi.com",
        page_size: int = 100,
        max_pages: int = 100,
        timeout: float = 30,
        mapping: SchemaMapping | str | None = "hubspot_deals",
        session: Any | None = None,
    ):
        self.token = token
        self.object = object
        self.properties = properties
        self.base_url = base_url
        super().__init__(
            url=f"{base_url.rstrip('/')}/crm/v3/objects/{object}",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": page_size, "properties": ",".join(properties)},
            records_path="results",
            pagination={
                "style": "cursor",
                "cursor_param": "after",
                "cursor_path": "paging.next.after",
            },
            page_size=page_size,
            max_pages=max_pages,
            timeout=timeout,
            mapping=mapping,
            session=session,
        )

    def _prepare_record(self, record: dict) -> dict:
        """Promote the per-record ``properties`` dict to top-level keys."""
        out = {k: v for k, v in record.items() if k != "properties"}
        out.update(record.get("properties") or {})
        return out


class PostHogSource(RestSource):
    """PostHog project events via ``/api/projects/{project_id}/{endpoint}/``.

    Follows the body's ``next`` link until absent.
    Default mapping: ``"posthog_events"``.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str | int,
        base_url: str = "https://us.posthog.com",
        endpoint: str = "events",
        max_pages: int = 100,
        timeout: float = 30,
        mapping: SchemaMapping | str | None = "posthog_events",
        session: Any | None = None,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = base_url
        self.endpoint = endpoint
        super().__init__(
            url=f"{base_url.rstrip('/')}/api/projects/{project_id}/{endpoint}/",
            headers={"Authorization": f"Bearer {api_key}"},
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
            max_pages=max_pages,
            timeout=timeout,
            mapping=mapping,
            session=session,
        )


class StripeSource(RestSource):
    """Stripe list endpoints (invoices by default) via ``/v1/{object}``.

    Pages Stripe-style: while the body's ``has_more`` is true, re-request
    with ``starting_after`` set to the last record's ``id``.
    Default mapping: ``"stripe_invoices"``.
    """

    def __init__(
        self,
        api_key: str,
        object: str = "invoices",
        base_url: str = "https://api.stripe.com",
        page_size: int = 100,
        max_pages: int = 100,
        timeout: float = 30,
        mapping: SchemaMapping | str | None = "stripe_invoices",
        session: Any | None = None,
    ):
        self.api_key = api_key
        self.object = object
        self.base_url = base_url
        super().__init__(
            url=f"{base_url.rstrip('/')}/v1/{object}",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": page_size},
            records_path="data",
            pagination={
                "style": "has_more",
                "more_path": "has_more",
                "cursor_param": "starting_after",
                "id_field": "id",
            },
            page_size=page_size,
            max_pages=max_pages,
            timeout=timeout,
            mapping=mapping,
            session=session,
        )


class SalesforceSource(RestSource):
    """Salesforce SOQL query results via the REST ``query`` endpoint.

    Follows the body's ``nextRecordsUrl`` (a relative link) until absent and
    drops the per-record ``attributes`` metadata dict. Token acquisition
    (the OAuth flow) is out of scope — pass an already-obtained
    ``access_token`` for ``instance_url``.
    Default mapping: ``"salesforce_opportunities"``.
    """

    def __init__(
        self,
        instance_url: str,
        access_token: str,
        soql: str,
        api_version: str = "v60.0",
        max_pages: int = 100,
        timeout: float = 30,
        mapping: SchemaMapping | str | None = "salesforce_opportunities",
        session: Any | None = None,
    ):
        self.instance_url = instance_url
        self.access_token = access_token
        self.soql = soql
        self.api_version = api_version
        super().__init__(
            url=f"{instance_url.rstrip('/')}/services/data/{api_version}/query",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"q": soql},
            records_path="records",
            pagination={"style": "next_url", "next_path": "nextRecordsUrl"},
            max_pages=max_pages,
            timeout=timeout,
            mapping=mapping,
            session=session,
        )

    def _prepare_record(self, record: dict) -> dict:
        """Drop the per-record Salesforce ``attributes`` metadata dict."""
        return {k: v for k, v in record.items() if k != "attributes"}
