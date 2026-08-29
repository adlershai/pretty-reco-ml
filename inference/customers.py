"""Build current (not as-of-T) customer inputs and encode with the Customer Tower."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.dataset_builder import (
    customer_static_map,
    discount_value,
    is_discount_purchase,
    json_safe,
    normalize_id,
    prepare_purchases,
)
from training.config import TrainConfig
from training.dataset import CatalogModel, collate_batch
from training.feature_encoders import FeatureEncoders
from training.two_tower import TwoTowerModel

from inference.csv_models import ANONYMOUS_CUSTOMER_ID

EXCLUDE_NO_PURCHASES = "no_purchases"
EXCLUDE_ANONYMOUS = "anonymous"
EXCLUDE_INVALID_ID = "invalid_customer_id"
EXCLUDE_NO_CATALOG_HISTORY = "no_catalog_history"


def _normalized_ids(frame: pd.DataFrame, column: str = "customer_id") -> set[str]:
    if frame.empty or column not in frame.columns:
        return set()
    return {cid for cid in (normalize_id(value) for value in frame[column]) if cid}


def customer_profile_map(customers: pd.DataFrame) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if customers.empty:
        return profiles
    for row in customers.to_dict("records"):
        customer_id = normalize_id(row.get("customer_id"))
        if not customer_id:
            continue
        profiles[customer_id] = {
            "preferred_size": json_safe(row.get("preferred_size")),
            "history_confidence": json_safe(row.get("history_confidence")),
        }
    return profiles


def _utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def build_current_customers(
    purchases: pd.DataFrame,
    customers: pd.DataFrame,
    catalog: dict[str, CatalogModel],
    *,
    new_model_codes: set[str],
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One record per eligible customer using full history (current inference state)."""
    moment = _utc_timestamp(now or datetime.now(timezone.utc))
    reasons: Counter[str] = Counter()
    prepared = prepare_purchases(purchases) if not purchases.empty else purchases
    customer_ids = _normalized_ids(customers)
    purchase_ids = _normalized_ids(prepared) if prepared is not None and not prepared.empty else set()
    all_ids = customer_ids | purchase_ids
    static_map = customer_static_map(customers)
    profiles = customer_profile_map(customers)
    records: list[dict[str, Any]] = []

    grouped = prepared.groupby("customer_id", sort=False) if not prepared.empty else []
    for customer_id, group in grouped:
        cid = str(customer_id) if customer_id is not None else ""
        if not cid:
            reasons[EXCLUDE_INVALID_ID] += 1
            continue
        if cid == ANONYMOUS_CUSTOMER_ID:
            reasons[EXCLUDE_ANONYMOUS] += 1
            continue
        rows = group.to_dict("records")
        if not rows:
            reasons[EXCLUDE_NO_PURCHASES] += 1
            continue

        history_ids: list[str] = []
        full_price_count = 0
        discount_count = 0
        discount_sum = 0.0
        season_counts: Counter[str] = Counter()
        recent_models: list[str] = []
        for row in rows:
            code = str(row["model"])
            if code in new_model_codes:
                continue
            history_ids.append(code)
            if is_discount_purchase(row.get("discount")):
                discount_count += 1
            else:
                full_price_count += 1
            discount_sum += discount_value(row.get("discount"))
            season = json_safe(row.get("season"))
            if season is not None and str(season):
                season_counts[str(season)] += 1
            recent_models.append(code)

        if not history_ids:
            reasons[EXCLUDE_NO_PURCHASES] += 1
            continue

        last_row = rows[-1]
        last_date = pd.Timestamp(last_row["purchase_date"])
        recency_days = float((moment - _utc_timestamp(last_date)).total_seconds() / 86400.0)
        if recency_days < 0:
            recency_days = 0.0
        last_name = None
        last_size = json_safe(last_row.get("size"))
        for row in reversed(rows):
            name = json_safe(row.get("last_name"))
            if name is not None and str(name).strip():
                last_name = str(name).strip()
                break
        purchase_count = len(history_ids)
        total_priced = full_price_count + discount_count
        static = dict(static_map.get(cid) or {})
        join_raw = static.get("join_date")
        join_date = pd.to_datetime(join_raw, utc=True, errors="coerce") if join_raw else pd.NaT
        tenure_days = (
            float((moment - join_date).total_seconds() / 86400.0) if pd.notna(join_date) else None
        )
        static["tenure_days"] = tenure_days
        behavior = {
            "purchase_count_before": purchase_count,
            "history_length": purchase_count,
            "recency_days": recency_days,
            "full_price_count_before": full_price_count,
            "discount_count_before": discount_count,
            "full_price_ratio_before": (full_price_count / total_priced) if total_priced else None,
            "discount_ratio_before": (discount_count / total_priced) if total_priced else None,
            "avg_discount_before": (discount_sum / purchase_count) if purchase_count else None,
            "season_counts_before": dict(season_counts),
        }
        catalog_history = [code for code in history_ids if code in catalog]
        if not catalog_history:
            reasons[EXCLUDE_NO_CATALOG_HISTORY] += 1
            continue
        profile = profiles.get(cid) or {}
        records.append(
            {
                "customer_id": cid,
                "customer_name": last_name,
                "history_model_ids": history_ids,
                "catalog_history_ids": catalog_history,
                "static": static,
                "behavior": behavior,
                "history_length": purchase_count,
                "last_purchase_date": _utc_timestamp(last_date).date().isoformat(),
                "preferred_size": profile.get("preferred_size"),
                "last_purchased_size": last_size,
                "history_confidence": profile.get("history_confidence"),
                "recent_models": recent_models[-8:],
            }
        )

    encoded_ids = {record["customer_id"] for record in records}
    for cid in customer_ids - purchase_ids:
        if cid in encoded_ids:
            continue
        if cid == ANONYMOUS_CUSTOMER_ID:
            reasons[EXCLUDE_ANONYMOUS] += 1
        else:
            reasons[EXCLUDE_NO_PURCHASES] += 1

    reason_counts = {key: int(value) for key, value in reasons.items()}
    stats = {
        "total_customers": len(all_ids),
        "encoded_customers": len(records),
        "excluded_customers": len(all_ids) - len(records),
        "reason_counts": reason_counts,
    }
    return records, stats


def encode_customer_vectors(
    model: TwoTowerModel,
    records: list[dict[str, Any]],
    catalog: dict[str, CatalogModel],
    *,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    encoders: FeatureEncoders = model.encoders
    config: TrainConfig = model.config
    if not records:
        return np.zeros((0, config.embedding_dim), dtype=np.float32)
    rows = []
    for record in records:
        history_visuals = []
        history_cats = []
        for code in record["catalog_history_ids"][-config.max_history :]:
            item = catalog.get(code)
            if item is None:
                continue
            history_visuals.append(item.visual)
            history_cats.append(encoders.encode_model_categoricals(item.features))
        rows.append(
            {
                "target_visual": history_visuals[0]
                if history_visuals
                else np.zeros(encoders.visual_input_dim, dtype=np.float32),
                "target_cats": history_cats[0] if history_cats else encoders.encode_model_categoricals({}),
                "history_visuals": history_visuals,
                "history_cats": history_cats,
                "numeric": encoders.encode_numeric(record["static"], record["behavior"]),
                "age_id": encoders.encode_age(record["static"].get("age_group")),
                "target_code": record["customer_id"],
            }
        )
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = collate_batch(rows[start : start + batch_size], config)
            encoded = model.encode_customers(batch, device=device)
            chunks.append(encoded.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)
