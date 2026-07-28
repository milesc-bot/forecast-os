"""Tests for REST sources (connectors.rest).

Every HTTP behavior is exercised against a local stdlib ``http.server``
stub started in-thread on 127.0.0.1 with an OS-assigned port; nothing in
this file touches the live network.
"""

import json
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

import forecast_os.connectors.rest as rest_module
from forecast_os.connectors.base import SchemaMapping
from forecast_os.connectors.rest import (
    HubSpotSource,
    PostHogSource,
    RestSource,
    SalesforceSource,
    StripeSource,
)
from forecast_os.core.exceptions import ForecastOSError
from forecast_os.core.types import ID_COL, TARGET_COL, TIME_COL

# an address no test ever connects to (construction-only tests)
UNREACHED = "http://127.0.0.1:9"


class _StubHandler(BaseHTTPRequestHandler):
    """Serves ``self.server.app(path, query, headers) -> (status, body)``."""

    def do_GET(self):
        split = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(split.query).items()}
        status, body = self.server.app(split.path, query, self.headers)
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def serve():
    """Start a stub server for an app callable; returns its base URL."""
    servers = []

    def _serve(app):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        srv.app = app
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        servers.append((srv, thread))
        return f"http://127.0.0.1:{srv.server_port}"

    yield _serve
    for srv, thread in servers:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


class TestSinglePage:
    def test_body_is_the_list_when_records_path_none(self, serve):
        def app(path, query, headers):
            return 200, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

        base = serve(app)
        df = RestSource(base + "/things").fetch()
        assert list(df["a"]) == [1, 2]
        assert list(df["b"]) == ["x", "y"]

    def test_records_path_dot_navigation(self, serve):
        def app(path, query, headers):
            return 200, {"data": {"items": [{"a": 1}, {"a": 2}]}}

        base = serve(app)
        df = RestSource(base, records_path="data.items").fetch()
        assert list(df["a"]) == [1, 2]

    def test_missing_records_path_raises(self, serve):
        def app(path, query, headers):
            return 200, {"nope": []}

        base = serve(app)
        with pytest.raises(ForecastOSError, match="data.items"):
            RestSource(base, records_path="data.items").fetch()

    def test_params_and_headers_are_sent(self, serve):
        seen = []

        def app(path, query, headers):
            seen.append((dict(query), headers.get("Authorization")))
            return 200, []

        base = serve(app)
        df = RestSource(
            base, headers={"Authorization": "Bearer zzz"}, params={"limit": 5}
        ).fetch()
        assert df.empty
        assert seen == [({"limit": "5"}, "Bearer zzz")]


class TestPagination:
    def test_cursor_walks_pages_and_stops_when_absent(self, serve):
        seen = []

        def app(path, query, headers):
            seen.append(dict(query))
            cur = query.get("cursor")
            if cur is None:
                return 200, {"items": [{"n": 0}, {"n": 1}], "meta": {"next": "c1"}}
            if cur == "c1":
                return 200, {"items": [{"n": 2}, {"n": 3}], "meta": {"next": "c2"}}
            return 200, {"items": [{"n": 4}]}  # no meta.next -> stop

        base = serve(app)
        src = RestSource(
            base,
            records_path="items",
            pagination={"style": "cursor", "cursor_param": "cursor", "cursor_path": "meta.next"},
        )
        df = src.fetch()
        assert list(df["n"]) == [0, 1, 2, 3, 4]
        assert len(seen) == 3
        assert seen[1]["cursor"] == "c1"
        assert seen[2]["cursor"] == "c2"

    def test_offset_stops_on_short_page(self, serve):
        rows = [{"n": i} for i in range(5)]
        seen = []

        def app(path, query, headers):
            off = int(query.get("offset", 0))
            seen.append(off)
            return 200, rows[off : off + 2]

        base = serve(app)
        src = RestSource(
            base, page_size=2, pagination={"style": "offset", "offset_param": "offset"}
        )
        df = src.fetch()
        assert list(df["n"]) == [0, 1, 2, 3, 4]
        assert seen == [0, 2, 4]

    def test_offset_exact_multiple_stops_on_empty_page(self, serve):
        rows = [{"n": i} for i in range(4)]
        seen = []

        def app(path, query, headers):
            off = int(query.get("offset", 0))
            seen.append(off)
            return 200, rows[off : off + 2]

        base = serve(app)
        src = RestSource(
            base, page_size=2, pagination={"style": "offset", "offset_param": "offset"}
        )
        df = src.fetch()
        assert list(df["n"]) == [0, 1, 2, 3]
        assert seen == [0, 2, 4]

    def test_page_number_stops_on_short_page(self, serve):
        rows = [{"n": i} for i in range(5)]
        seen = []

        def app(path, query, headers):
            page = int(query.get("page", 1))
            seen.append(page)
            return 200, {"results": rows[(page - 1) * 2 : page * 2]}

        base = serve(app)
        src = RestSource(
            base,
            records_path="results",
            page_size=2,
            pagination={"style": "page", "page_param": "page"},
        )
        df = src.fetch()
        assert list(df["n"]) == [0, 1, 2, 3, 4]
        assert seen == [1, 2, 3]

    def test_next_url_follows_absolute_and_relative_links(self, serve):
        state = {}

        def app(path, query, headers):
            if path == "/start":
                return 200, {"results": [{"n": 0}], "next": state["base"] + "/page2"}
            if path == "/page2":
                return 200, {"results": [{"n": 1}], "next": "/page3"}
            if path == "/page3":
                return 200, {"results": [{"n": 2}], "next": None}
            return 404, {"error": "not found"}

        base = serve(app)
        state["base"] = base
        src = RestSource(
            base + "/start",
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
        )
        df = src.fetch()
        assert list(df["n"]) == [0, 1, 2]

    def test_has_more_missing_id_field_names_field_and_keys(self, serve):
        def app(path, query, headers):
            return 200, {"data": [{"amount": 1, "currency": "usd"}], "has_more": True}

        base = serve(app)
        src = RestSource(
            base,
            records_path="data",
            pagination={
                "style": "has_more",
                "more_path": "has_more",
                "cursor_param": "starting_after",
                "id_field": "id",
            },
        )
        with pytest.raises(ForecastOSError, match="'id'") as excinfo:
            src.fetch()
        # the error names the record's actual keys so the fix is obvious
        assert "amount" in str(excinfo.value)
        assert "currency" in str(excinfo.value)

    def test_max_pages_cap_warns_and_stops(self, serve):
        calls = []

        def app(path, query, headers):
            calls.append(1)
            return 200, {"items": [{"n": len(calls)}], "next_cursor": "more"}

        base = serve(app)
        src = RestSource(
            base,
            records_path="items",
            max_pages=3,
            pagination={"style": "cursor", "cursor_param": "cursor", "cursor_path": "next_cursor"},
        )
        with pytest.warns(UserWarning, match="max_pages"):
            df = src.fetch()
        assert list(df["n"]) == [1, 2, 3]
        assert len(calls) == 3


class TestErrors:
    def test_non_2xx_raises_with_status_and_body(self, serve):
        def app(path, query, headers):
            return 500, {"error": "boom"}

        base = serve(app)
        with pytest.raises(ForecastOSError, match="500") as excinfo:
            RestSource(base).fetch()
        assert "boom" in str(excinfo.value)

    def test_error_body_snippet_is_truncated(self, serve):
        def app(path, query, headers):
            return 503, "x" * 1000

        base = serve(app)
        with pytest.raises(ForecastOSError, match="503") as excinfo:
            RestSource(base).fetch()
        msg = str(excinfo.value)
        assert "x" * 100 in msg
        assert "x" * 1000 not in msg

    def test_unknown_pagination_style_raises_at_construction(self):
        with pytest.raises(ValueError, match="pagination style"):
            RestSource(UNREACHED, pagination={"style": "zigzag"})

    def test_missing_pagination_key_raises_at_construction(self):
        with pytest.raises(ValueError, match="cursor_path"):
            RestSource(UNREACHED, pagination={"style": "cursor", "cursor_param": "c"})

    def test_pagination_without_style_raises_at_construction(self):
        with pytest.raises(ValueError, match="style"):
            RestSource(UNREACHED, pagination={"cursor_param": "c"})


class _FakeResponse:
    status_code = 200
    text = "[]"

    def json(self):
        return [{"n": 1}]


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, headers, params, timeout))
        return _FakeResponse()


class _ScriptedSession:
    """Serves canned JSON bodies in order, recording ``(url, headers)``.

    The local ``http.server`` stub cannot bind port 443 or answer for a
    named host, so origin-comparison tests script the session instead.
    """

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, headers))
        body = self.bodies.pop(0)
        return type(
            "_Scripted", (), {"status_code": 200, "text": "", "json": lambda self: body}
        )()


class TestSessionAndAttrs:
    def test_user_supplied_session_is_used(self):
        sess = _FakeSession()
        df = RestSource(UNREACHED + "/x", timeout=7, session=sess).fetch()
        assert list(df["n"]) == [1]
        assert sess.calls[0][0] == UNREACHED + "/x"
        assert sess.calls[0][3] == 7

    def test_constructor_args_stored_as_same_named_attributes(self):
        pagination = {"style": "offset", "offset_param": "o"}
        src = RestSource(
            UNREACHED,
            headers={"X": "1"},
            params={"a": 1},
            records_path="r",
            pagination=pagination,
            page_size=5,
            max_pages=7,
            timeout=9,
            mapping="m",
        )
        assert src.url == UNREACHED
        assert src.headers == {"X": "1"}
        assert src.params == {"a": 1}
        assert src.records_path == "r"
        assert src.pagination == pagination
        assert src.page_size == 5
        assert src.max_pages == 7
        assert src.timeout == 9
        assert src.mapping == "m"
        assert src.session is None

    def test_default_mapping_names(self):
        assert HubSpotSource(token="t").mapping == "hubspot_deals"
        assert PostHogSource(api_key="k", project_id="1").mapping == "posthog_events"
        assert StripeSource(api_key="k").mapping == "stripe_invoices"
        sfdc = SalesforceSource(instance_url=UNREACHED, access_token="t", soql="q")
        assert sfdc.mapping == "salesforce_opportunities"


def _hubspot_app(seen):
    page1 = [
        {
            "id": "1",
            "createdAt": "2024-01-02T00:00:00Z",
            "properties": {"amount": 100, "dealstage": "closedwon", "closedate": "2024-01-15"},
        },
        {
            "id": "2",
            "createdAt": "2024-01-03T00:00:00Z",
            "properties": {"amount": 200, "dealstage": "closedwon", "closedate": "2024-01-20"},
        },
    ]
    page2 = [
        {
            "id": "3",
            "createdAt": "2024-02-01T00:00:00Z",
            "properties": {"amount": 999, "dealstage": "closedlost", "closedate": "2024-02-10"},
        },
        {
            "id": "4",
            "createdAt": "2024-02-20T00:00:00Z",
            "properties": {"amount": 50, "dealstage": "closedwon", "closedate": "2024-03-05"},
        },
    ]

    def app(path, query, headers):
        seen.append((path, dict(query), headers.get("Authorization")))
        if path != "/crm/v3/objects/deals":
            return 404, {"error": "not found"}
        if "after" not in query:
            return 200, {"results": page1, "paging": {"next": {"after": "pg2"}}}
        return 200, {"results": page2}

    return app


class TestHubSpot:
    def test_fetch_flattens_properties_and_walks_cursor(self, serve):
        seen = []
        base = serve(_hubspot_app(seen))
        src = HubSpotSource(token="secret-token", base_url=base, page_size=2)
        df = src.fetch()
        assert list(df["id"]) == ["1", "2", "3", "4"]
        assert list(df["amount"]) == [100, 200, 999, 50]
        assert list(df["dealstage"]) == ["closedwon", "closedwon", "closedlost", "closedwon"]
        assert "createdAt" in df.columns
        assert not any(c == "properties" or c.startswith("properties.") for c in df.columns)
        # endpoint, auth, and query params
        first_path, first_query, first_auth = seen[0]
        assert first_path == "/crm/v3/objects/deals"
        assert first_auth == "Bearer secret-token"
        assert first_query["limit"] == "2"
        # hubspot_owner_id joins the defaults so id_cols=("owner",) works
        assert first_query["properties"] == "amount,dealstage,closedate,hubspot_owner_id"
        # cursor from paging.next.after sent as "after" on page 2
        assert seen[1][1]["after"] == "pg2"
        assert len(seen) == 2

    def test_to_panel_with_inline_mapping(self, serve):
        base = serve(_hubspot_app([]))
        src = HubSpotSource(token="secret-token", base_url=base, page_size=2)
        mapping = SchemaMapping(
            name="hubspot_inline",
            description="closed-won deal amounts by close month",
            date_col="closedate",
            value_col="amount",
            filters={"dealstage": ("closedwon",)},
        )
        panel = src.to_panel(mapping=mapping)
        assert list(panel[ID_COL].unique()) == ["hubspot_inline"]
        assert list(panel[TIME_COL]) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]
        # Jan = 100 + 200 closed-won, Feb = gap fill, Mar = 50; closedlost filtered
        assert list(panel[TARGET_COL]) == [300.0, 0.0, 50.0]


class TestStripe:
    def test_has_more_walk_sends_starting_after(self, serve):
        seen = []
        page1 = [{"id": "in_1", "amount_due": 100}, {"id": "in_2", "amount_due": 200}]
        page2 = [{"id": "in_3", "amount_due": 300}]

        def app(path, query, headers):
            seen.append((path, dict(query), headers.get("Authorization")))
            if path != "/v1/invoices":
                return 404, {"error": "not found"}
            if "starting_after" not in query:
                return 200, {"object": "list", "data": page1, "has_more": True}
            return 200, {"object": "list", "data": page2, "has_more": False}

        base = serve(app)
        src = StripeSource(api_key="sk_test_x", base_url=base, page_size=2)
        df = src.fetch()
        assert list(df["id"]) == ["in_1", "in_2", "in_3"]
        assert list(df["amount_due"]) == [100, 200, 300]
        assert seen[0][1]["limit"] == "2"
        assert seen[0][2] == "Bearer sk_test_x"
        # cursor is the LAST record id of the previous page
        assert seen[1][1]["starting_after"] == "in_2"
        assert len(seen) == 2


class TestSalesforce:
    def test_next_records_url_walk_and_attributes_drop(self, serve):
        seen = []
        attrs = {"type": "Opportunity", "url": "/services/data/v60.0/sobjects/Opportunity/006"}
        page1 = {
            "totalSize": 3,
            "done": False,
            "records": [
                {"attributes": attrs, "Name": "a", "Amount": 100.0},
                {"attributes": attrs, "Name": "b", "Amount": 200.0},
            ],
            "nextRecordsUrl": "/services/data/v60.0/query/01g-2000",
        }
        page2 = {
            "totalSize": 3,
            "done": True,
            "records": [{"attributes": attrs, "Name": "c", "Amount": 300.0}],
        }

        def app(path, query, headers):
            seen.append((path, dict(query), headers.get("Authorization")))
            if path == "/services/data/v60.0/query":
                return 200, page1
            if path == "/services/data/v60.0/query/01g-2000":
                return 200, page2
            return 404, {"error": "not found"}

        base = serve(app)
        src = SalesforceSource(
            instance_url=base,
            access_token="tok",
            soql="SELECT Name, Amount FROM Opportunity",
        )
        df = src.fetch()
        assert list(df["Name"]) == ["a", "b", "c"]
        assert list(df["Amount"]) == [100.0, 200.0, 300.0]
        assert not any(c == "attributes" or c.startswith("attributes.") for c in df.columns)
        assert seen[0][1]["q"] == "SELECT Name, Amount FROM Opportunity"
        assert seen[0][2] == "Bearer tok"
        # next page hits nextRecordsUrl relative to the instance, without q
        assert seen[1][0] == "/services/data/v60.0/query/01g-2000"
        assert "q" not in seen[1][1]


class TestPostHog:
    def test_next_link_walk(self, serve):
        state = {}
        seen = []

        def app(path, query, headers):
            seen.append((path, dict(query), headers.get("Authorization")))
            if path != "/api/projects/123/events/":
                return 404, {"error": "not found"}
            if "before" not in query:
                nxt = state["base"] + "/api/projects/123/events/?before=x"
                return 200, {"results": [{"event": "signup"}], "next": nxt}
            return 200, {"results": [{"event": "purchase"}], "next": None}

        base = serve(app)
        state["base"] = base
        src = PostHogSource(api_key="phx", project_id="123", base_url=base)
        df = src.fetch()
        assert list(df["event"]) == ["signup", "purchase"]
        assert seen[0][0] == "/api/projects/123/events/"
        assert seen[0][2] == "Bearer phx"
        assert seen[1][1]["before"] == "x"
        assert len(seen) == 2


class TestOriginGuard:
    """Regression: `next_url` pagination follows a link the response BODY
    chooses, and ``_get_json`` attached ``self.headers`` to whatever came
    back — no scheme check, no host check. A body naming a third-party host
    therefore received the bearer token, and a DRF/PostHog-style ``next``
    link rewritten to ``http://`` by a misconfigured reverse proxy sent the
    API key in cleartext. ``requests`` strips Authorization across a
    cross-host redirect for exactly this reason; the library's own hop must
    behave the same. Off-origin links stay followable (signed/CDN page links
    are legitimate) — just unauthenticated, and with a warning."""

    def test_headers_are_not_sent_to_a_host_named_by_the_response_body(self, serve):
        third_party = []

        def other(path, query, headers):
            third_party.append((path, headers.get("Authorization")))
            return 200, {"results": [{"n": 1}], "next": None}

        other_base = serve(other)

        def home(path, query, headers):
            return 200, {"results": [{"n": 0}], "next": other_base + "/exfil"}

        home_base = serve(home)
        src = RestSource(
            home_base,
            headers={"Authorization": "Bearer ph_SUPER_SECRET_TOKEN"},
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
        )
        with pytest.warns(UserWarning, match="off-origin"):
            df = src.fetch()
        assert list(df["n"]) == [0, 1]
        # the page was still fetched, but with no credentials attached
        assert third_party == [("/exfil", None)]

    def test_headers_survive_a_same_origin_next_link(self, serve):
        seen = []

        def app(path, query, headers):
            seen.append((path, headers.get("Authorization")))
            if path == "/start":
                return 200, {"results": [{"n": 0}], "next": "/page2"}
            return 200, {"results": [{"n": 1}], "next": None}

        base = serve(app)
        src = RestSource(
            base + "/start",
            headers={"Authorization": "Bearer keepme"},
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
        )
        df = src.fetch()
        assert list(df["n"]) == [0, 1]
        assert seen == [("/start", "Bearer keepme"), ("/page2", "Bearer keepme")]

    def test_default_port_spelled_out_is_the_same_origin(self):
        """Regression: ``https://h/v`` vs ``https://h:443/v`` lost the token.

        The origin compare was a raw ``scheme://netloc`` string match, so a
        ``next`` link naming the default port explicitly read as a different
        host. Django REST Framework (what PostHogSource paginates against)
        builds its absolute ``next`` from the ``Host`` header, so a proxy
        forwarding ``Host: api.good.com:443`` produced exactly this and
        turned page 2 into a 401 on a setup that paginated fine in v0.9.0.
        """
        pages = [
            {"results": [{"n": 0}], "next": "https://api.good.com:443/v?page=2"},
            {"results": [{"n": 1}], "next": None},
        ]
        sess = _ScriptedSession(pages)
        src = RestSource(
            "https://api.good.com/v",
            headers={"Authorization": "Bearer SECRET"},
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
            session=sess,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no off-origin warning: same host
            assert list(src.fetch()["n"]) == [0, 1]
        assert [h for _, h in sess.calls] == [
            {"Authorization": "Bearer SECRET"},
            {"Authorization": "Bearer SECRET"},
        ]

    def test_only_credential_headers_are_withheld_off_origin(self):
        """Content negotiation must survive a legitimate CDN page link.

        Dropping *every* header off-origin meant a source configured with
        only ``Accept``/versioning headers — no token at all — lost them on a
        signed CDN link, which can flip the CDN to an HTML response and turn
        a working fetch into a "body that is not JSON" error. ``requests``
        strips the credentials across a cross-host redirect and forwards the
        rest; so does this hop.
        """
        pages = [
            {"results": [{"n": 0}], "next": "https://cdn.example.net/p2"},
            {"results": [{"n": 1}], "next": None},
        ]
        sess = _ScriptedSession(pages)
        src = RestSource(
            "https://api.good.com/v",
            headers={
                "Accept": "application/json",
                "X-Api-Version": "2",
                "Authorization": "Bearer SECRET",
                "X-Api-Key": "k",
                "Cookie": "sid=1",
            },
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
            session=sess,
        )
        with pytest.warns(UserWarning, match="off-origin"):
            assert list(src.fetch()["n"]) == [0, 1]
        assert sess.calls[1][1] == {"Accept": "application/json", "X-Api-Version": "2"}

    def test_no_credential_headers_means_no_warning_and_nothing_dropped(self):
        pages = [
            {"results": [{"n": 0}], "next": "https://cdn.example.net/p2"},
            {"results": [{"n": 1}], "next": None},
        ]
        sess = _ScriptedSession(pages)
        src = RestSource(
            "https://api.good.com/v",
            headers={"Accept": "application/json"},
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
            session=sess,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # nothing was withheld: nothing to warn about
            src.fetch()
        assert sess.calls[1][1] == {"Accept": "application/json"}

    def test_non_http_next_link_is_refused(self, serve):
        def app(path, query, headers):
            return 200, {"results": [{"n": 0}], "next": "file:///etc/passwd"}

        base = serve(app)
        src = RestSource(
            base,
            records_path="results",
            pagination={"style": "next_url", "next_path": "next"},
        )
        with pytest.raises(ForecastOSError, match="http"):
            src.fetch()


class TestPageStyleFirstPage:
    """Regression: the first `page`-style request omitted ``page_param``
    entirely and the next-page arithmetic then asserted the page just
    returned was page 1. Against a 0-indexed API that silently skipped a
    whole page of records. The first page number is now sent explicitly and
    ``first_page`` says what it is."""

    def test_first_request_sends_the_page_number_explicitly(self, serve):
        rows = [{"n": i} for i in range(3)]
        seen = []

        def app(path, query, headers):
            seen.append(query.get("page"))
            page = int(query.get("page", 1))
            return 200, {"results": rows[(page - 1) * 2 : page * 2]}

        base = serve(app)
        src = RestSource(
            base,
            records_path="results",
            page_size=2,
            pagination={"style": "page", "page_param": "page"},
        )
        assert list(src.fetch()["n"]) == [0, 1, 2]
        assert seen == ["1", "2"]

    def test_first_page_zero_fetches_the_zeroth_page(self, serve):
        rows = [{"n": i} for i in range(6)]
        seen = []

        def app(path, query, headers):
            page = int(query["page"])  # a 0-indexed API: no implicit default
            seen.append(page)
            return 200, {"results": rows[page * 2 : (page + 1) * 2]}

        base = serve(app)
        src = RestSource(
            base,
            records_path="results",
            page_size=2,
            pagination={"style": "page", "page_param": "page", "first_page": 0},
        )
        # every row, including page 0's — which the old code never requested
        assert list(src.fetch()["n"]) == [0, 1, 2, 3, 4, 5]
        assert seen == [0, 1, 2, 3]


class TestServerShapeErrors:
    """Regression: this module wraps every other server-shape problem in a
    ForecastOSError with a body snippet, but a non-JSON 200 escaped as a raw
    ``requests`` JSONDecodeError and a list of non-objects escaped as a raw
    pandas TypeError."""

    def test_non_json_200_body_raises_forecast_error_with_snippet(self, serve):
        def app(path, query, headers):
            return 200, "<html>maintenance</html>"

        base = serve(app)
        with pytest.raises(ForecastOSError, match="not JSON") as excinfo:
            RestSource(base).fetch()
        assert "maintenance" in str(excinfo.value)

    def test_non_object_record_names_index_and_type(self, serve):
        def app(path, query, headers):
            return 200, {"data": [{"a": 1}, 2, {"a": 3}]}

        base = serve(app)
        with pytest.raises(ForecastOSError, match="item 1") as excinfo:
            RestSource(base, records_path="data").fetch()
        assert "int" in str(excinfo.value)

    def test_null_records_are_still_accepted(self, serve):
        """json_normalize treats NA-like entries as blank rows; keep that."""

        def app(path, query, headers):
            return 200, {"data": [{"a": 1}, None]}

        base = serve(app)
        df = RestSource(base, records_path="data").fetch()
        assert len(df) == 2

    def test_null_record_does_not_crash_the_platform_presets(self, serve):
        """Regression: a ``null`` in ``results`` died as a raw AttributeError.

        ``_extract_records`` explicitly permits null entries, but ``fetch``
        then handed each one to ``_prepare_record``, and both
        ``HubSpotSource`` (``record.items()``) and ``SalesforceSource`` (a
        dict comprehension over ``record.items()``) require a dict — so the
        one value the validator waves through was the one that still escaped
        as ``'NoneType' object has no attribute 'items'``. It stays a blank
        row, as it already is for the base source.
        """

        def app(path, query, headers):
            return 200, {"results": [{"id": "1", "properties": {"amount": "5"}}, None]}

        base = serve(app)
        df = HubSpotSource(token="t", base_url=base).fetch()
        assert len(df) == 2
        assert df["amount"].iloc[0] == "5"
        assert df["amount"].isna().sum() == 1  # the null entry: a blank row

    def test_null_record_does_not_crash_salesforce(self, serve):
        def app(path, query, headers):
            return 200, {"records": [{"attributes": {"type": "X"}, "Amount": 5}, None]}

        base = serve(app)
        df = SalesforceSource(instance_url=base, access_token="t", soql="q").fetch()
        assert len(df) == 2
        assert list(df.columns) == ["Amount"]


def _hubspot_faithful_app(seen):
    """HubSpot v3 semantics: only the requested properties come back."""
    deals = [
        {
            "id": "1",
            "properties": {
                "amount": "100",
                "dealstage": "closedwon",
                "closedate": "2026-01-15",
                "hubspot_owner_id": "42",
            },
        },
        {
            "id": "2",
            "properties": {
                "amount": "250",
                "dealstage": "closedwon",
                "closedate": "2026-01-20",
                "hubspot_owner_id": "7",
            },
        },
    ]

    def app(path, query, headers):
        wanted = query.get("properties", "").split(",")
        seen.append(wanted)
        results = [
            {
                "id": d["id"],
                "properties": {k: v for k, v in d["properties"].items() if k in wanted},
            }
            for d in deals
        ]
        return 200, {"results": results}

    return app


class TestHubSpotOwner:
    def test_readme_owner_split_works_against_a_faithful_stub(self, serve):
        """Regression: HubSpot returns ONLY the properties you ask for, and
        the default tuple omitted hubspot_owner_id — so the README's
        headline example, ``HubSpotSource(token=...).to_panel(id_cols=("owner",))``,
        died with "mapping 'hubspot_deals' needs column(s) ['owner']" even
        though the recipe pre-wires the rename. The owner property is now
        requested by default."""
        seen = []
        base = serve(_hubspot_faithful_app(seen))
        panel = HubSpotSource(token="t", base_url=base).to_panel(id_cols=("owner",))
        assert "hubspot_owner_id" in seen[0]
        assert set(panel[ID_COL]) == {"42", "7"}
        assert list(panel[panel[ID_COL] == "42"][TARGET_COL]) == [100.0]


class TestImportGuard:
    def test_missing_requests_raises_with_extras_hint(self, monkeypatch):
        monkeypatch.setattr(rest_module, "_HAS_REQUESTS", False)
        with pytest.raises(ImportError, match=r"forecast-os\[connectors\]"):
            RestSource(UNREACHED)

    def test_platform_subclass_is_guarded_too(self, monkeypatch):
        monkeypatch.setattr(rest_module, "_HAS_REQUESTS", False)
        with pytest.raises(ImportError, match=r"forecast-os\[connectors\]"):
            HubSpotSource(token="t")

    def test_user_supplied_session_needs_no_requests(self, monkeypatch):
        monkeypatch.setattr(rest_module, "_HAS_REQUESTS", False)
        src = RestSource(UNREACHED, session=_FakeSession())
        assert src.session is not None
