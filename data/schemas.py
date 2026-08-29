"""View names, required columns, and snapshot layout for the training pipeline."""

from __future__ import annotations

from typing import Iterable

PURCHASE_VIEW = "vw_reco_purchase_events_v1"
CUSTOMER_VIEW = "vw_reco_customer_representation_v1"
MODEL_VIEW = "vw_reco_model_representation_v1"

SOURCE_VIEWS = {
    "purchases": PURCHASE_VIEW,
    "customers": CUSTOMER_VIEW,
    "models": MODEL_VIEW,
}

PURCHASE_COLUMNS = (
    "purchase_id",
    "invoice_number",
    "purchase_date",
    "customer_id",
    "model",
    "sku",
    "size",
    "quantity",
    "discount",
    "season",
    "last_name",
)

CUSTOMER_COLUMNS = (
    "customer_id",
    "birthday_year",
    "age_group",
    "preferred_size",
    "agent_id",
    "join_date",
    "join_shop_id",
    "purchase_count",
    "invoice_count",
    "unique_models",
    "first_purchase_date",
    "last_purchase_date",
    "days_since_last_purchase",
    "history_confidence",
    "full_price_purchase_count",
    "discount_purchase_count",
    "full_price_ratio",
    "discount_purchase_ratio",
    "avg_discount",
    "season_count",
)

MODEL_COLUMNS = (
    "model_id",
    "model",
    "model_name",
    "main_category",
    "sub_category",
    "material",
    "material_1",
    "material_2",
    "theme",
    "color",
    "season",
    "last_type",
    "embedding_model",
    "embedding_dimension",
    "main_embedding",
    "pers_embedding",
    "side_embedding",
)

MODEL_FEATURE_COLUMNS = (
    "main_category",
    "sub_category",
    "material",
    "material_1",
    "material_2",
    "theme",
    "color",
    "season",
    "last_type",
)

VISUAL_EMBEDDING_COLUMNS = ("main_embedding", "pers_embedding", "side_embedding")

SNAPSHOT_PURCHASES = "purchases.parquet"
SNAPSHOT_CUSTOMERS = "customers.parquet"
SNAPSHOT_MODELS = "models.parquet"
SNAPSHOT_METADATA = "metadata.json"

DATASET_DIRNAME = "dataset"
DATASET_TRAIN = "train.parquet"
DATASET_VALIDATION = "validation.parquet"
DATASET_TEST = "test.parquet"
DATASET_METADATA = "dataset_metadata.json"

DEFAULT_TRAIN_LOOKBACK_DAYS = 120
DEFAULT_VALIDATION_LOOKBACK_DAYS = 60
DEFAULT_TRAIN_QUANTILE = 0.70
DEFAULT_VALIDATION_QUANTILE = 0.85


class SchemaError(ValueError):
    """Required columns are missing from a view or snapshot table."""


def missing_columns(present: Iterable[str], required: Iterable[str]) -> list[str]:
    present_set = set(present)
    return [name for name in required if name not in present_set]


def require_columns(present: Iterable[str], required: Iterable[str], source: str) -> None:
    missing = missing_columns(present, required)
    if missing:
        raise SchemaError(f"{source} is missing required columns: {', '.join(missing)}")
