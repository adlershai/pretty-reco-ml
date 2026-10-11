"""HTTP client for step-db-server (db.adler-backend.com).

No direct MySQL. Optional auth comes from environment variables.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from data.config import DbApiSettings, db_api_settings

logger = logging.getLogger("pretty-reco-ml.data")

USER_AGENT = "pretty-reco-ml/data"
MAX_ATTEMPTS = 3
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
RETRY_BACKOFF_SECONDS = (0.5, 1.0)


class DbApiError(Exception):
    """DB API request failed."""


class DbApiTimeoutError(DbApiError):
    """DB API request timed out."""


class DbApiHttpError(DbApiError):
    """DB API returned a non-success HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class DbApiResponseError(DbApiError):
    """DB API returned a body that is not a view result."""


class DbApiClient:
    """POST /api/v1/db/{dbName} with type=view."""

    def __init__(
        self,
        *,
        settings: DbApiSettings | None = None,
        session: requests.Session | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Any = time.sleep,
    ) -> None:
        self.settings = settings or db_api_settings()
        self.session = session or requests.Session()
        self.max_attempts = max(1, int(max_attempts))
        self._sleep = sleep
        if not self.settings.db_name:
            raise DbApiError("DB_NAME is required")
        if not self.settings.base_url:
            raise DbApiError("DB_API_URL is required")

    @property
    def url(self) -> str:
        return f"{self.settings.base_url}/api/v1/db/{self.settings.db_name}"

    def get_view(self, view_name: str) -> list[dict[str, Any]]:
        name = str(view_name or "").strip()
        if not name:
            raise DbApiError("view name is required")
        payload = {"type": "view", "viewName": name}
        body = self._post(payload)
        rows = normalize_view_payload(body)
        return [_normalize_record(row, name) for row in rows]

    def select(
        self,
        table: str,
        *,
        fields: list[str] | None = None,
        where: str | None = None,
        values: list[Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """Unfiltered table SELECT via type=select. Filtered reads use proc()."""
        name = str(table or "").strip()
        if not name:
            raise DbApiError("table is required")
        if where:
            raise DbApiError("filtered select is not allowed; use proc")
        options: dict[str, Any] = {}
        if fields:
            options["fields"] = list(fields)
        if values is not None:
            options["values"] = list(values)
        if order_by:
            options["orderBy"] = str(order_by)
        if limit is not None:
            options["limit"] = int(limit)
        if offset is not None:
            options["offset"] = int(offset)
        payload = {"type": "select", "table_name": name, "options": options}
        body = self._post(payload)
        rows = normalize_view_payload(body)
        return [_normalize_record(row, name) for row in rows]

    def proc(self, proc_name: str, value: Any = None) -> list[dict[str, Any]]:
        """CALL procName(?) via type=proc. JSON object/list values are serialized."""
        name = str(proc_name or "").strip()
        if not name:
            raise DbApiError("proc name is required")
        if value is None:
            payload_value = "{}"
        elif isinstance(value, (dict, list)):
            payload_value = json.dumps(value, separators=(",", ":"))
        else:
            payload_value = str(value)
        payload = {"type": "proc", "procName": name, "value": payload_value}
        body = self._post(payload)
        rows = normalize_view_payload(_coerce_proc_payload(body))
        return [_normalize_record(row, name) for row in rows]

    def all(self, sql: str, values: list[Any] | None = None) -> list[dict[str, Any]]:
        raise DbApiError("all is not allowed")

    def insert(
        self,
        table: str,
        fields: list[str],
        values: list[Any],
        *,
        on_duplicate_update: list[str] | None = None,
    ) -> Any:
        """Insert one row via type=insert."""
        name = str(table or "").strip()
        if not name:
            raise DbApiError("table is required")
        if not fields:
            raise DbApiError("fields are required")
        options: dict[str, Any] = {}
        if on_duplicate_update:
            options["on_duplicate_update"] = list(on_duplicate_update)
        payload: dict[str, Any] = {
            "type": "insert",
            "table_name": name,
            "fields": list(fields),
            "values": list(values),
        }
        if options:
            payload["options"] = options
        return self._post(payload)

    def update(self, table: str, row_id: int, fields: list[str], values: list[Any]) -> Any:
        """Update one row by primary key via type=update."""
        name = str(table or "").strip()
        if not name:
            raise DbApiError("table is required")
        if int(row_id) <= 0:
            raise DbApiError("row_id must be >= 1")
        if not fields:
            raise DbApiError("fields are required")
        return self._post(
            {
                "type": "update",
                "table_name": name,
                "row_id": int(row_id),
                "fields": list(fields),
                "values": list(values),
            }
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.settings.api_key:
            headers["X-API-Key"] = self.settings.api_key
        if self.settings.api_token:
            headers["Authorization"] = f"Bearer {self.settings.api_token}"
        return headers

    def _post(self, payload: dict[str, Any]) -> Any:
        timeout = (self.settings.connect_timeout, self.settings.read_timeout)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout,
                )
            except requests.Timeout as exc:
                last_error = DbApiTimeoutError(f"DB API timed out calling {self.url}: {exc}")
                logger.warning("DB API timeout attempt %s/%s: %s", attempt, self.max_attempts, exc)
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise last_error from exc
            except requests.RequestException as exc:
                last_error = DbApiError(f"DB API request failed: {exc}")
                logger.warning(
                    "DB API network error attempt %s/%s: %s",
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise last_error from exc

            if response.status_code in RETRYABLE_STATUS and attempt < self.max_attempts:
                logger.warning(
                    "DB API HTTP %s attempt %s/%s",
                    response.status_code,
                    attempt,
                    self.max_attempts,
                )
                self._backoff(attempt)
                continue

            if not response.ok:
                raise DbApiHttpError(
                    response.status_code,
                    _http_error_message(response),
                )
            return _parse_json_body(response)

        raise last_error or DbApiError("DB API request failed")

    def _backoff(self, attempt: int) -> None:
        index = min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
        self._sleep(RETRY_BACKOFF_SECONDS[index])


def _parse_json_body(response: requests.Response) -> Any:
    text = response.text or ""
    if not text.strip():
        raise DbApiResponseError(f"DB API {response.status_code}: empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[:200].replace("\n", " ")
        raise DbApiResponseError(
            f"DB API {response.status_code}: non-JSON response: {snippet}"
        ) from exc


def _http_error_message(response: requests.Response) -> str:
    text = response.text or ""
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("error"):
        return f"DB API {response.status_code}: {parsed['error']}"
    snippet = text[:200].replace("\n", " ") if text else ""
    if snippet:
        return f"DB API {response.status_code}: {snippet}"
    return f"DB API {response.status_code}"


def _coerce_proc_payload(payload: Any) -> Any:
    """step-db returns a list, a single row object, or an envelope."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and not payload.get("error"):
        for key in ("data", "rows", "result", "records"):
            if isinstance(payload.get(key), list):
                return payload
        if isinstance(payload.get("fields"), list) and isinstance(payload.get("values"), list):
            return payload
        return [payload]
    return payload


def normalize_view_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise DbApiResponseError(f"unexpected view payload type: {type(payload).__name__}")
    if payload.get("error"):
        raise DbApiResponseError(str(payload["error"]))
    for key in ("data", "rows", "result", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    fields = payload.get("fields")
    values = payload.get("values")
    if isinstance(fields, list) and isinstance(values, list):
        return [dict(zip(fields, row, strict=False)) for row in values]
    raise DbApiResponseError("unexpected view payload shape")


def _normalize_record(row: Any, view_name: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DbApiResponseError(
            f"{view_name} row is {type(row).__name__}, expected an object"
        )
    return {str(key): value for key, value in row.items()}
