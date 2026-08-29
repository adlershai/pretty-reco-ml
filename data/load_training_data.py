"""Fetch reco training views and write an immutable Parquet snapshot.

Usage:
    python -m data.load_training_data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.config import DEFAULT_SNAPSHOT_ROOT, REPO_ROOT, db_api_settings, load_dotenv
from data.db_client import DbApiClient, DbApiError
from data.schemas import (
    CUSTOMER_COLUMNS,
    CUSTOMER_VIEW,
    MODEL_COLUMNS,
    MODEL_VIEW,
    PURCHASE_COLUMNS,
    PURCHASE_VIEW,
    SNAPSHOT_CUSTOMERS,
    SNAPSHOT_METADATA,
    SNAPSHOT_MODELS,
    SNAPSHOT_PURCHASES,
    SOURCE_VIEWS,
    SchemaError,
    require_columns,
)

logger = logging.getLogger("pretty-reco-ml.data")


@dataclass(frozen=True)
class TrainingViews:
    purchases: list[dict[str, Any]]
    customers: list[dict[str, Any]]
    models: list[dict[str, Any]]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def snapshot_timestamp(now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H%M%S")


def validate_view_rows(rows: list[dict[str, Any]], required: tuple[str, ...], source: str) -> None:
    if not rows:
        logger.warning("%s returned 0 rows", source)
        return
    require_columns(rows[0].keys(), required, source)


def fetch_training_views(client: DbApiClient) -> TrainingViews:
    logger.info("fetching %s", PURCHASE_VIEW)
    purchases = client.get_view(PURCHASE_VIEW)
    validate_view_rows(purchases, PURCHASE_COLUMNS, PURCHASE_VIEW)

    logger.info("fetching %s", CUSTOMER_VIEW)
    customers = client.get_view(CUSTOMER_VIEW)
    validate_view_rows(customers, CUSTOMER_COLUMNS, CUSTOMER_VIEW)

    logger.info("fetching %s", MODEL_VIEW)
    models = client.get_view(MODEL_VIEW)
    validate_view_rows(models, MODEL_COLUMNS, MODEL_VIEW)

    return TrainingViews(purchases=purchases, customers=customers, models=models)


def records_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def write_snapshot(
    views: TrainingViews,
    *,
    snapshot_root: Path,
    created_at: datetime | None = None,
    db_name: str,
) -> Path:
    created = created_at or datetime.now(timezone.utc)
    directory = snapshot_root / snapshot_timestamp(created)
    directory.mkdir(parents=True, exist_ok=False)

    purchases = records_to_frame(views.purchases)
    customers = records_to_frame(views.customers)
    models = records_to_frame(views.models)

    purchases.to_parquet(directory / SNAPSHOT_PURCHASES, engine="pyarrow", index=False)
    customers.to_parquet(directory / SNAPSHOT_CUSTOMERS, engine="pyarrow", index=False)
    models.to_parquet(directory / SNAPSHOT_MODELS, engine="pyarrow", index=False)

    metadata = {
        "created_at": created.isoformat(),
        "db_name": db_name,
        "purchase_rows": int(len(purchases)),
        "customer_rows": int(len(customers)),
        "model_rows": int(len(models)),
        "source_views": dict(SOURCE_VIEWS),
    }
    (directory / SNAPSHOT_METADATA).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def print_load_report(directory: Path, views: TrainingViews) -> None:
    lines = [
        "=== Raw training snapshot ===",
        f"snapshot: {directory}",
        f"total purchases loaded: {len(views.purchases)}",
        f"total customers loaded: {len(views.customers)}",
        f"total models loaded: {len(views.models)}",
    ]
    print("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch reco views and write a Parquet snapshot.")
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Directory that will contain timestamped snapshot folders",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)
    settings = db_api_settings()
    client = DbApiClient(settings=settings)
    try:
        views = fetch_training_views(client)
        directory = write_snapshot(
            views,
            snapshot_root=args.snapshot_root,
            db_name=settings.db_name,
        )
    except (SchemaError, DbApiError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    print_load_report(directory, views)
    return 0


if __name__ == "__main__":
    sys.exit(main())
