"""Environment loading for the training-data pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ROOT = REPO_ROOT / "local"
DEFAULT_DB_API_URL = "https://db.adler-backend.com"
DEFAULT_DB_NAME = "payments"
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 180.0
DEFAULT_SNAPSHOT_ROOT = LOCAL_ROOT / "snapshots"
LEGACY_SNAPSHOT_ROOT = REPO_ROOT / "snapshots"
DEFAULT_OUTPUT_ROOT = LOCAL_ROOT / "outputs"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs without overwriting existing environment values."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class DbApiSettings:
    base_url: str
    db_name: str
    api_key: str | None
    api_token: str | None
    connect_timeout: float
    read_timeout: float


def db_api_settings() -> DbApiSettings:
    raw_timeout = os.environ.get("DB_API_TIMEOUT", "").strip()
    read_timeout = float(raw_timeout) if raw_timeout else DEFAULT_READ_TIMEOUT
    if read_timeout <= 0:
        raise ValueError("DB_API_TIMEOUT must be > 0")
    api_key = os.environ.get("DB_API_KEY", "").strip() or None
    api_token = os.environ.get("DB_API_TOKEN", "").strip() or None
    return DbApiSettings(
        base_url=os.environ.get("DB_API_URL", DEFAULT_DB_API_URL).rstrip("/"),
        db_name=os.environ.get("DB_NAME", DEFAULT_DB_NAME).strip() or DEFAULT_DB_NAME,
        api_key=api_key,
        api_token=api_token,
        connect_timeout=DEFAULT_CONNECT_TIMEOUT,
        read_timeout=read_timeout,
    )
