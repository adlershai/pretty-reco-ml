"""Build leakage-free sequential training examples from a snapshot.

Usage:
    python -m data.dataset_builder --snapshot local/snapshots/<timestamp>
    python -m data.dataset_builder --latest
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.config import DEFAULT_SNAPSHOT_ROOT, LEGACY_SNAPSHOT_ROOT, REPO_ROOT, load_dotenv
from data.schemas import (
    ANONYMOUS_CUSTOMER_ID,
    CUSTOMER_COLUMNS,
    DATASET_DIRNAME,
    DATASET_METADATA,
    DATASET_TEST,
    DATASET_TRAIN,
    DATASET_VALIDATION,
    DEFAULT_TRAIN_LOOKBACK_DAYS,
    DEFAULT_TRAIN_QUANTILE,
    DEFAULT_VALIDATION_LOOKBACK_DAYS,
    DEFAULT_VALIDATION_QUANTILE,
    MODEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OBSERVATION_START,
    PURCHASE_COLUMNS,
    SNAPSHOT_CUSTOMERS,
    SNAPSHOT_METADATA,
    SNAPSHOT_MODELS,
    SNAPSHOT_PURCHASES,
    SOURCE_VIEWS,
    VISUAL_EMBEDDING_COLUMNS,
    SchemaError,
    require_columns,
)

logger = logging.getLogger("pretty-reco-ml.data")

TIE_BREAK_COLUMNS = ("purchase_id", "invoice_number", "sku", "model")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def parse_embedding(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.ndarray):
        parsed: Any = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "nan"}:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
    elif isinstance(value, (list, tuple)):
        parsed = value
    else:
        return None
    try:
        array = np.asarray(parsed, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return array


def combine_visual_embeddings(*vectors: np.ndarray | None) -> np.ndarray | None:
    available = [vector for vector in vectors if vector is not None and vector.size > 0]
    if not available:
        return None
    dim_counts = Counter(int(vector.size) for vector in available)
    common_dim = dim_counts.most_common(1)[0][0]
    aligned = [vector for vector in available if int(vector.size) == common_dim]
    mean = np.mean(np.stack(aligned, axis=0), axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    return (mean / norm).astype(np.float32)


def normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer():
            return str(int(value))
    if isinstance(value, np.integer):
        return str(int(value))
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and (math.isnan(float(value)) or math.isinf(float(value))):
        return True
    if isinstance(value, pd.Timestamp):
        return bool(pd.isna(value))
    return False


def json_safe(value: Any) -> Any:
    if _is_null(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def as_utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def observation_start_ts(start: str | pd.Timestamp | None = None) -> pd.Timestamp:
    return as_utc(start or OBSERVATION_START)


def observed_tenure_days(at: pd.Timestamp, first_observed: Any) -> float | None:
    """History duration from the first ML-visible purchase, not pre-2019 join tenure."""
    if first_observed is None or _is_null(first_observed):
        return None
    days = (as_utc(at) - as_utc(first_observed)).total_seconds() / 86400.0
    return max(float(days), 0.0)


def parse_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _days_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return (pd.Timestamp(later) - pd.Timestamp(earlier)).total_seconds() / 86400.0


def load_snapshot_frames(snapshot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    purchases = pd.read_parquet(snapshot_dir / SNAPSHOT_PURCHASES, engine="pyarrow")
    customers = pd.read_parquet(snapshot_dir / SNAPSHOT_CUSTOMERS, engine="pyarrow")
    models = pd.read_parquet(snapshot_dir / SNAPSHOT_MODELS, engine="pyarrow")
    if not purchases.empty:
        require_columns(purchases.columns, PURCHASE_COLUMNS, SNAPSHOT_PURCHASES)
    if not customers.empty:
        require_columns(customers.columns, CUSTOMER_COLUMNS, SNAPSHOT_CUSTOMERS)
    if not models.empty:
        require_columns(models.columns, MODEL_COLUMNS, SNAPSHOT_MODELS)
    return purchases, customers, models


def find_latest_snapshot(snapshot_root: Path) -> Path:
    candidates: list[Path] = []
    if snapshot_root.is_dir():
        candidates = [
            path
            for path in snapshot_root.iterdir()
            if path.is_dir() and (path / SNAPSHOT_PURCHASES).is_file()
        ]
    looking_at_default = Path(snapshot_root).resolve() == DEFAULT_SNAPSHOT_ROOT.resolve()
    if not candidates and looking_at_default and LEGACY_SNAPSHOT_ROOT.is_dir():
        candidates = [
            path
            for path in LEGACY_SNAPSHOT_ROOT.iterdir()
            if path.is_dir() and (path / SNAPSHOT_PURCHASES).is_file()
        ]
    if not candidates:
        if not snapshot_root.is_dir() and not looking_at_default:
            raise FileNotFoundError(f"snapshot root does not exist: {snapshot_root}")
        raise FileNotFoundError(f"no snapshots found under {snapshot_root}")
    return max(candidates, key=lambda path: path.name)


def model_visual_map(models: pd.DataFrame) -> dict[str, np.ndarray | None]:
    visuals: dict[str, np.ndarray | None] = {}
    if models.empty:
        return visuals
    for row in models.itertuples(index=False):
        model_code = normalize_id(getattr(row, "model", None))
        if not model_code:
            continue
        vectors = [parse_embedding(getattr(row, column, None)) for column in VISUAL_EMBEDDING_COLUMNS]
        visuals[model_code] = combine_visual_embeddings(*vectors)
    return visuals


def model_feature_map(models: pd.DataFrame) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    if models.empty:
        return features
    keep = ("model_id", "model", "model_name", *MODEL_FEATURE_COLUMNS)
    present = [column for column in keep if column in models.columns]
    for row in models[present].to_dict("records"):
        model_code = normalize_id(row.get("model"))
        if not model_code:
            continue
        features[model_code] = {key: json_safe(row.get(key)) for key in present}
    return features


def customer_static_map(customers: pd.DataFrame) -> dict[str, dict[str, Any]]:
    static: dict[str, dict[str, Any]] = {}
    if customers.empty:
        return static
    keep = ("customer_id", "age_group", "join_date", "join_shop_id", "agent_id", "birthday_year")
    present = [column for column in keep if column in customers.columns]
    frame = customers[present].copy()
    if "join_date" in frame.columns:
        frame["join_date"] = parse_datetime_series(frame["join_date"])
    for row in frame.to_dict("records"):
        customer_id = normalize_id(row.get("customer_id"))
        if not customer_id:
            continue
        static[customer_id] = {key: json_safe(row.get(key)) for key in present if key != "customer_id"}
    return static


def _as_float(value: Any) -> float | None:
    if _is_null(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_discount_purchase(discount: Any) -> bool:
    numeric = _as_float(discount)
    return numeric is not None and numeric > 0.0


def discount_value(discount: Any) -> float:
    numeric = _as_float(discount)
    return 0.0 if numeric is None else numeric


def prepare_purchases(purchases: pd.DataFrame) -> pd.DataFrame:
    frame = purchases.copy()
    frame["customer_id"] = frame["customer_id"].map(normalize_id)
    frame["model"] = frame["model"].map(normalize_id)
    frame["purchase_date"] = parse_datetime_series(frame["purchase_date"])
    frame = frame[frame["customer_id"].notna() & frame["model"].notna() & frame["purchase_date"].notna()]
    frame = frame[frame["customer_id"] != ANONYMOUS_CUSTOMER_ID]
    frame = frame[frame["purchase_date"] >= observation_start_ts()]
    if "quantity" in frame.columns:
        quantity = pd.to_numeric(frame["quantity"], errors="coerce")
        frame = frame[quantity > 0]
    if "paid_amount" in frame.columns:
        paid = pd.to_numeric(frame["paid_amount"], errors="coerce")
        frame = frame[paid > 0]
    sort_columns = ["customer_id", "purchase_date"]
    for column in TIE_BREAK_COLUMNS:
        if column in frame.columns:
            sort_columns.append(column)
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def build_sequential_examples(
    purchases: pd.DataFrame,
    *,
    customer_static: dict[str, dict[str, Any]],
    model_features: dict[str, dict[str, Any]],
    model_visuals: dict[str, np.ndarray | None],
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {
        "excluded_examples_missing_target_model": 0,
        "excluded_examples_missing_visual": 0,
        "sequential_candidates": 0,
    }
    records: list[dict[str, Any]] = []
    grouped = purchases.groupby("customer_id", sort=False)
    for customer_id, group in grouped:
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        history_models: list[str] = []
        history_dates: list[str] = []
        full_price_count = 0
        discount_count = 0
        discount_sum = 0.0
        season_counts: Counter[str] = Counter()
        for index, row in enumerate(rows):
            model_code = row["model"]
            purchase_date = row["purchase_date"]
            if index > 0:
                counts["sequential_candidates"] += 1
                visual = model_visuals.get(model_code)
                features = model_features.get(model_code)
                if features is None:
                    counts["excluded_examples_missing_target_model"] += 1
                elif visual is None:
                    counts["excluded_examples_missing_visual"] += 1
                else:
                    target_date = pd.Timestamp(purchase_date)
                    last_history_date = pd.Timestamp(rows[index - 1]["purchase_date"])
                    recency_days = _days_between(target_date, last_history_date)
                    purchase_count_before = len(history_models)
                    total_priced = full_price_count + discount_count
                    first_observed = pd.Timestamp(rows[0]["purchase_date"]) if history_models else None
                    tenure_days = observed_tenure_days(target_date, first_observed)
                    static_features = dict(customer_static.get(str(customer_id)) or {})
                    static_features["tenure_days"] = tenure_days
                    behavior = {
                        "purchase_count_before": purchase_count_before,
                        "history_length": purchase_count_before,
                        "recency_days": recency_days,
                        "full_price_count_before": full_price_count,
                        "discount_count_before": discount_count,
                        "full_price_ratio_before": (full_price_count / total_priced) if total_priced else None,
                        "discount_ratio_before": (discount_count / total_priced) if total_priced else None,
                        "avg_discount_before": (discount_sum / purchase_count_before) if purchase_count_before else None,
                        "season_counts_before": dict(season_counts),
                    }
                    records.append(
                        {
                            "customer_id": str(customer_id),
                            "target_model": model_code,
                            "target_purchase_date": target_date,
                            "history_model_ids": list(history_models),
                            "history_purchase_dates": list(history_dates),
                            "customer_static_features": json.dumps(static_features, sort_keys=True),
                            "customer_behavior_features": json.dumps(behavior, sort_keys=True),
                            "target_model_features": json.dumps(features, sort_keys=True),
                            "target_visual_embedding": visual.astype(np.float32).tolist(),
                        }
                    )
            history_models.append(model_code)
            history_dates.append(pd.Timestamp(purchase_date).isoformat())
            if is_discount_purchase(row.get("discount")):
                discount_count += 1
            else:
                full_price_count += 1
            discount_sum += discount_value(row.get("discount"))
            season = json_safe(row.get("season"))
            if season is not None and str(season):
                season_counts[str(season)] += 1
    examples = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=[
            "customer_id",
            "target_model",
            "target_purchase_date",
            "history_model_ids",
            "history_purchase_dates",
            "customer_static_features",
            "customer_behavior_features",
            "target_model_features",
            "target_visual_embedding",
        ]
    )
    return examples, counts


def resolve_split_bounds(
    purchase_dates: pd.Series,
    train_end: str | pd.Timestamp | None,
    validation_end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    dates = pd.to_datetime(purchase_dates, utc=True, errors="coerce").dropna()
    if dates.empty:
        raise ValueError("no purchase dates available to compute a time split")
    latest = pd.Timestamp(dates.max())
    if train_end is not None or validation_end is not None:
        if train_end is None or validation_end is None:
            raise ValueError("TRAIN_END and VALIDATION_END must be supplied together")
        start = pd.Timestamp(train_end)
        end = pd.Timestamp(validation_end)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        if start >= end:
            raise ValueError("TRAIN_END must be before VALIDATION_END")
        return start, end, "configured"

    calendar_train_end = latest - pd.Timedelta(int(DEFAULT_TRAIN_LOOKBACK_DAYS), unit="D")
    calendar_validation_end = latest - pd.Timedelta(int(DEFAULT_VALIDATION_LOOKBACK_DAYS), unit="D")
    if dates.min() <= calendar_train_end:
        return calendar_train_end, calendar_validation_end, "default_calendar"

    unique_days = np.sort(dates.dt.normalize().unique())
    if unique_days.size < 3:
        train_cut = pd.Timestamp(unique_days[0])
        validation_cut = pd.Timestamp(unique_days[min(1, unique_days.size - 1)])
        return train_cut, validation_cut, "default_quantiles"

    train_index = min(max(int(unique_days.size * DEFAULT_TRAIN_QUANTILE), 0), unique_days.size - 2)
    validation_index = min(
        max(int(unique_days.size * DEFAULT_VALIDATION_QUANTILE), train_index + 1),
        unique_days.size - 1,
    )
    return (
        pd.Timestamp(unique_days[train_index]),
        pd.Timestamp(unique_days[validation_index]),
        "default_quantiles",
    )


def split_examples(
    examples: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if examples.empty:
        empty = examples.copy()
        return empty, empty.copy(), empty.copy()
    dates = pd.to_datetime(examples["target_purchase_date"], utc=True)
    train = examples.loc[dates <= train_end].copy()
    validation = examples.loc[(dates > train_end) & (dates <= validation_end)].copy()
    test = examples.loc[dates > validation_end].copy()
    return train, validation, test


def customer_history_counts(purchases: pd.DataFrame, customers: pd.DataFrame) -> dict[str, int]:
    purchase_counts = (
        purchases.groupby("customer_id").size() if not purchases.empty else pd.Series(dtype=int)
    )
    customers_in_view = set(customers["customer_id"].map(normalize_id).dropna()) if not customers.empty else set()
    customers_in_purchases = set(purchase_counts.index.astype(str))
    zero = len(customers_in_view - customers_in_purchases)
    one = int((purchase_counts == 1).sum()) if not purchase_counts.empty else 0
    two_plus = int((purchase_counts >= 2).sum()) if not purchase_counts.empty else 0
    return {
        "customers_with_0_purchases": zero,
        "customers_with_1_purchase": one,
        "customers_with_2plus_purchases": two_plus,
    }


def history_length_distribution(examples: pd.DataFrame) -> dict[str, int]:
    if examples.empty:
        return {}
    lengths = examples["history_model_ids"].map(len)
    return {str(int(key)): int(value) for key, value in sorted(lengths.value_counts().items())}


def availability_stats(values: list[Any]) -> dict[str, Any]:
    present = 0
    missing = 0
    distribution: Counter[str] = Counter()
    for value in values:
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            missing += 1
            continue
        present += 1
        distribution[str(value)] += 1
    total = present + missing
    return {
        "present": present,
        "missing": missing,
        "share_present": (present / total) if total else None,
        "distribution": dict(distribution),
    }


def build_quality_report(
    *,
    snapshot_dir: Path,
    purchases: pd.DataFrame,
    customers: pd.DataFrame,
    models: pd.DataFrame,
    examples: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    example_counts: dict[str, int],
    model_visuals: dict[str, np.ndarray | None],
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    split_strategy: str,
) -> dict[str, Any]:
    known_models = set(model_visuals)
    missing_model_purchases = (
        int((~purchases["model"].isin(known_models)).sum()) if not purchases.empty else 0
    )
    models_without_visual = sum(visual is None for visual in model_visuals.values())
    earliest = purchases["purchase_date"].min() if not purchases.empty else None
    latest = purchases["purchase_date"].max() if not purchases.empty else None

    age_groups: list[Any] = []
    seasons: list[Any] = []
    for raw in examples["customer_static_features"] if not examples.empty else []:
        payload = json.loads(raw)
        age_groups.append(payload.get("age_group"))
    for raw in examples["target_model_features"] if not examples.empty else []:
        payload = json.loads(raw)
        seasons.append(payload.get("season"))

    snapshot_meta_path = snapshot_dir / SNAPSHOT_METADATA
    snapshot_meta = {}
    if snapshot_meta_path.is_file():
        snapshot_meta = json.loads(snapshot_meta_path.read_text(encoding="utf-8"))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(snapshot_dir),
        "db_name": snapshot_meta.get("db_name"),
        "source_views": snapshot_meta.get("source_views", dict(SOURCE_VIEWS)),
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "split_strategy": split_strategy,
        "observation_start": observation_start_ts().isoformat(),
        "total_purchases_loaded": int(len(purchases)),
        "total_customers_loaded": int(len(customers)),
        "total_models_loaded": int(len(models)),
        **customer_history_counts(purchases, customers),
        "training_examples_produced": int(len(examples)),
        "sequential_candidates": example_counts.get("sequential_candidates", 0),
        "purchases_with_missing_model": missing_model_purchases,
        "models_without_visual_embedding": int(models_without_visual),
        "excluded_examples_missing_target_model": example_counts.get(
            "excluded_examples_missing_target_model", 0
        ),
        "excluded_examples_missing_visual": example_counts.get("excluded_examples_missing_visual", 0),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "earliest_purchase_date": json_safe(earliest),
        "latest_purchase_date": json_safe(latest),
        "history_length_distribution": history_length_distribution(examples),
        "age_group_availability": availability_stats(age_groups),
        "season_availability": availability_stats(seasons),
    }


def format_quality_report(report: dict[str, Any]) -> str:
    history = report.get("history_length_distribution") or {}
    if history:
        lengths = sorted(int(key) for key in history)
        history_summary = f"min={lengths[0]} max={lengths[-1]} distinct={len(lengths)}"
    else:
        history_summary = "{}"
    lines = [
        "=== Training dataset quality report ===",
        f"total purchases loaded: {report['total_purchases_loaded']}",
        f"total customers loaded: {report['total_customers_loaded']}",
        f"total models loaded: {report['total_models_loaded']}",
        "",
        f"customers with 0 purchases: {report['customers_with_0_purchases']}",
        f"customers with 1 purchase: {report['customers_with_1_purchase']}",
        f"customers with 2+ purchases: {report['customers_with_2plus_purchases']}",
        "",
        f"training examples produced: {report['training_examples_produced']}",
        "",
        f"missing target models: {report['purchases_with_missing_model']}",
        f"models missing image embeddings: {report['models_without_visual_embedding']}",
        f"excluded examples (missing target model): {report['excluded_examples_missing_target_model']}",
        f"excluded examples (missing visual): {report['excluded_examples_missing_visual']}",
        "",
        f"train rows: {report['train_rows']}",
        f"validation rows: {report['validation_rows']}",
        f"test rows: {report['test_rows']}",
        f"split strategy: {report['split_strategy']}",
        f"observation start: {report.get('observation_start')}",
        f"TRAIN_END: {report['train_end']}",
        f"VALIDATION_END: {report['validation_end']}",
        "",
        f"earliest purchase date: {report['earliest_purchase_date']}",
        f"latest purchase date: {report['latest_purchase_date']}",
        "",
        f"history length: {history_summary}",
        f"age group availability: present={report['age_group_availability']['present']} "
        f"missing={report['age_group_availability']['missing']}",
        f"season availability: present={report['season_availability']['present']} "
        f"missing={report['season_availability']['missing']}",
    ]
    return "\n".join(lines)


def write_dataset_frames(
    output_dir: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / DATASET_TRAIN, engine="pyarrow", index=False)
    validation.to_parquet(output_dir / DATASET_VALIDATION, engine="pyarrow", index=False)
    test.to_parquet(output_dir / DATASET_TEST, engine="pyarrow", index=False)
    (output_dir / DATASET_METADATA).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def build_dataset(
    snapshot_dir: Path,
    *,
    train_end: str | None = None,
    validation_end: str | None = None,
    output_dirname: str = DATASET_DIRNAME,
) -> Path:
    snapshot_dir = snapshot_dir.resolve()
    purchases_raw, customers_raw, models_raw = load_snapshot_frames(snapshot_dir)
    purchases = prepare_purchases(purchases_raw) if not purchases_raw.empty else purchases_raw
    model_visuals = model_visual_map(models_raw)
    model_features = model_feature_map(models_raw)
    customer_static = customer_static_map(customers_raw)
    examples, example_counts = build_sequential_examples(
        purchases,
        customer_static=customer_static,
        model_features=model_features,
        model_visuals=model_visuals,
    )
    split_train_end, split_validation_end, strategy = resolve_split_bounds(
        purchases["purchase_date"] if not purchases.empty else pd.Series(dtype="datetime64[ns, UTC]"),
        train_end,
        validation_end,
    )
    train, validation, test = split_examples(examples, split_train_end, split_validation_end)
    report = build_quality_report(
        snapshot_dir=snapshot_dir,
        purchases=purchases,
        customers=customers_raw,
        models=models_raw,
        examples=examples,
        train=train,
        validation=validation,
        test=test,
        example_counts=example_counts,
        model_visuals=model_visuals,
        train_end=split_train_end,
        validation_end=split_validation_end,
        split_strategy=strategy,
    )
    report["raw_purchases_in_snapshot"] = int(len(purchases_raw))
    report["purchases_dropped_before_observation"] = int(len(purchases_raw) - len(purchases)) if not purchases_raw.empty else 0
    output_dir = snapshot_dir / output_dirname
    write_dataset_frames(output_dir, train, validation, test, report)
    print(format_quality_report(report))
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequential training datasets from a snapshot.")
    parser.add_argument("--snapshot", type=Path, help="Path to a snapshot directory")
    parser.add_argument("--latest", action="store_true", help="Use the newest snapshot under --snapshot-root")
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Root directory containing timestamped snapshots",
    )
    parser.add_argument("--train-end", help="Inclusive end date for the train split (YYYY-MM-DD)")
    parser.add_argument("--validation-end", help="Inclusive end date for the validation split (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dirname",
        default=DATASET_DIRNAME,
        help="Directory name under the snapshot for train/validation/test parquet",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)
    if args.snapshot and args.latest:
        logger.error("use either --snapshot or --latest, not both")
        return 1
    if not args.snapshot and not args.latest:
        logger.error("provide --snapshot PATH or --latest")
        return 1

    train_end = args.train_end or _env_date("TRAIN_END")
    validation_end = args.validation_end or _env_date("VALIDATION_END")
    try:
        snapshot_dir = args.snapshot if args.snapshot else find_latest_snapshot(args.snapshot_root)
        output_dir = build_dataset(
            snapshot_dir,
            train_end=train_end,
            validation_end=validation_end,
            output_dirname=args.output_dirname,
        )
    except (FileNotFoundError, SchemaError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    print(f"dataset written to {output_dir}")
    return 0


def _env_date(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


if __name__ == "__main__":
    sys.exit(main())
