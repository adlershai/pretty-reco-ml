"""DB API client tests. Responses are mocked; production is never called."""

from __future__ import annotations

import json

import pytest
import requests

from data.config import DbApiSettings
from data.db_client import (
    DbApiClient,
    DbApiError,
    DbApiHttpError,
    DbApiResponseError,
    DbApiTimeoutError,
)


def settings(**overrides: object) -> DbApiSettings:
    values = {
        "base_url": "https://db.adler-backend.com",
        "db_name": "payments",
        "api_key": None,
        "api_token": None,
        "connect_timeout": 1.0,
        "read_timeout": 2.0,
    }
    values.update(overrides)
    return DbApiSettings(**values)  # type: ignore[arg-type]


class FakeResponse:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = body if isinstance(body, str) else json.dumps(body)


class FakeSession:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, json: dict | None = None, headers: dict | None = None, timeout: object = None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(responses: list[object], **setting_overrides: object) -> tuple[DbApiClient, FakeSession]:
    session = FakeSession(responses)
    client = DbApiClient(
        settings=settings(**setting_overrides),
        session=session,
        sleep=lambda _seconds: None,
    )
    return client, session


def test_get_view_posts_expected_body_and_returns_rows() -> None:
    client, session = make_client([FakeResponse(200, [{"purchase_id": 1, "model": "A"}])])
    rows = client.get_view("vw_reco_purchase_events_v1")
    assert rows == [{"purchase_id": 1, "model": "A"}]
    assert session.calls[0]["url"] == "https://db.adler-backend.com/api/v1/db/payments"
    assert session.calls[0]["json"] == {"type": "view", "viewName": "vw_reco_purchase_events_v1"}
    headers = session.calls[0]["headers"]
    assert headers["Content-Type"] == "application/json"
    assert "X-API-Key" not in headers
    assert "Authorization" not in headers


def test_get_view_sends_optional_auth_headers() -> None:
    client, session = make_client(
        [FakeResponse(200, [])],
        api_key="secret-key",
        api_token="secret-token",
    )
    client.get_view("vw_reco_model_representation_v1")
    headers = session.calls[0]["headers"]
    assert headers["X-API-Key"] == "secret-key"
    assert headers["Authorization"] == "Bearer secret-token"


def test_get_view_unwraps_data_envelope() -> None:
    client, _session = make_client([FakeResponse(200, {"data": [{"customer_id": 9}]})])
    assert client.get_view("vw_reco_customer_representation_v1") == [{"customer_id": 9}]


def test_get_view_unwraps_fields_values() -> None:
    client, _session = make_client(
        [FakeResponse(200, {"fields": ["model", "color"], "values": [["40724_001", "black"]]})]
    )
    assert client.get_view("vw_reco_model_representation_v1") == [
        {"model": "40724_001", "color": "black"}
    ]


def test_retries_transient_http_then_succeeds() -> None:
    client, session = make_client(
        [
            FakeResponse(503, {"error": "busy"}),
            FakeResponse(200, [{"ok": 1}]),
        ]
    )
    assert client.get_view("vw_reco_purchase_events_v1") == [{"ok": 1}]
    assert len(session.calls) == 2


def test_does_not_retry_unauthorized() -> None:
    client, session = make_client([FakeResponse(401, {"error": "nope"})])
    with pytest.raises(DbApiHttpError, match="DB API 401: nope"):
        client.get_view("vw_reco_purchase_events_v1")
    assert len(session.calls) == 1


def test_retries_timeout_then_raises() -> None:
    client, session = make_client(
        [
            requests.Timeout("slow"),
            requests.Timeout("still slow"),
            requests.Timeout("nope"),
        ]
    )
    with pytest.raises(DbApiTimeoutError, match="timed out"):
        client.get_view("vw_reco_purchase_events_v1")
    assert len(session.calls) == 3


def test_malformed_json_raises() -> None:
    client, _session = make_client([FakeResponse(200, "<html>nope</html>")])
    with pytest.raises(DbApiResponseError, match="non-JSON"):
        client.get_view("vw_reco_purchase_events_v1")


def test_error_object_raises() -> None:
    client, _session = make_client([FakeResponse(200, {"error": "view not found"})])
    with pytest.raises(DbApiResponseError, match="view not found"):
        client.get_view("vw_reco_purchase_events_v1")


def test_non_object_row_raises() -> None:
    client, _session = make_client([FakeResponse(200, ["not-a-row"])])
    with pytest.raises(DbApiResponseError, match="str, expected an object"):
        client.get_view("vw_reco_purchase_events_v1")


def test_empty_view_name_raises() -> None:
    client, session = make_client([])
    with pytest.raises(DbApiError, match="view name is required"):
        client.get_view("  ")
    assert session.calls == []


def test_all_posts_parameterized_select() -> None:
    client, session = make_client([FakeResponse(200, [{"model": "49166_008"}])])
    rows = client.all(
        "SELECT * FROM vw_reco_model_representation_v1 WHERE model IN (?)",
        ["49166_008"],
    )
    assert rows == [{"model": "49166_008"}]
    assert session.calls[0]["json"] == {
        "type": "all",
        "sql": "SELECT * FROM vw_reco_model_representation_v1 WHERE model IN (?)",
        "values": ["49166_008"],
    }


def test_all_empty_sql_raises() -> None:
    client, session = make_client([])
    with pytest.raises(DbApiError, match="sql is required"):
        client.all("  ")
    assert session.calls == []
