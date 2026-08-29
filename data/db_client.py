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

    def all(self, sql: str, values: list[Any] | None = None) -> list[dict[str, Any]]:
        """Parameterized SELECT via type=all. No direct MySQL."""
        statement = str(sql or "").strip()
        if not statement:
            raise DbApiError("sql is required")
        payload = {"type": "all", "sql": statement, "values": list(values or [])}
        body = self._post(payload)
        rows = normalize_view_payload(body)
        return [_normalize_record(row, "all") for row in rows]

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
