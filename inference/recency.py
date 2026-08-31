"""Serving-layer recency policy: exclude recent buyers. Recency never changes like_score."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from typing import Any, Sequence

import numpy as np

from inference.csv_models import ANONYMOUS_CUSTOMER_ID

RECENCY_EXCLUDE_DAYS = 60
DAYS_1Y = 365
DAYS_3Y = 1095  # 365 * 3

BUCKET_LT_60 = "<60d"
BUCKET_60_180 = "60-180d"
BUCKET_181_365 = "181-365d"
BUCKET_1Y_3Y = "1-3y"
BUCKET_3Y_PLUS = "3y+"
BUCKET_UNKNOWN = "unknown"


def utc_today(now: date | datetime | None = None) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.date()
        return now.astimezone(timezone.utc).date()
    return now


def days_since_last_purchase(last_purchase_date: str | None, *, now: date | datetime | None = None) -> float | None:
    if last_purchase_date is None:
        return None
    text = str(last_purchase_date).strip()
    if not text or text.lower() in {"none", "nan", "nat", "null"}:
        return None
    try:
        purchased = date.fromisoformat(text[:10])
    except ValueError:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        purchased = parsed.date()
    return float((utc_today(now) - purchased).days)


def is_excluded_recent(days_since: float | None) -> bool:
    return days_since is not None and days_since < RECENCY_EXCLUDE_DAYS


def recency_bucket(days_since: float | None) -> str:
    if days_since is None:
        return BUCKET_UNKNOWN
    if days_since < RECENCY_EXCLUDE_DAYS:
        return BUCKET_LT_60
    if days_since <= 180:
        return BUCKET_60_180
    if days_since < DAYS_1Y:
        return BUCKET_181_365
    if days_since < DAYS_3Y:
        return BUCKET_1Y_3Y
    return BUCKET_3Y_PLUS


def rank_customers_with_recency(
    similarities: np.ndarray,
    customer_ids: Sequence[str],
    last_purchase_dates: Sequence[str | None],
    *,
    top_k: int,
    like_scores: np.ndarray | None = None,
    now: date | datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score everyone, drop <60d buyers, then Top N by like_score (or similarity)."""
    scores = np.asarray(similarities, dtype=np.float32).reshape(-1)
    n_customers = int(scores.shape[0])
    likes = scores if like_scores is None else np.asarray(like_scores, dtype=np.float32).reshape(-1)
    if likes.shape[0] != n_customers:
        raise ValueError("like_scores must align with similarities")
    if n_customers == 0:
        empty = _empty_diagnostics()
        empty["total_scored"] = 0
        return [], empty

    today = utc_today(now)
    days = np.full(n_customers, np.nan, dtype=np.float64)
    for index, raw_date in enumerate(last_purchase_dates):
        value = days_since_last_purchase(raw_date, now=today)
        if value is not None:
            days[index] = value

    recent_mask = np.isfinite(days) & (days < RECENCY_EXCLUDE_DAYS)
    anonymous_mask = np.array(
        [str(customer_id) == ANONYMOUS_CUSTOMER_ID for customer_id in customer_ids],
        dtype=np.bool_,
    )
    eligible_mask = ~recent_mask & ~anonymous_mask
    excluded_recent = int(recent_mask.sum())
    eligible_count = int(eligible_mask.sum())
    ranking = np.where(eligible_mask, likes, np.float32("-inf"))

    if eligible_count == 0:
        diagnostics = _diagnostics(
            total_scored=n_customers,
            excluded_lt_60d=excluded_recent,
            eligible=0,
            returned_rows=[],
            days_since=[],
        )
        return [], diagnostics

    k = min(int(top_k), eligible_count)
    eligible_index = np.flatnonzero(eligible_mask)
    eligible_ranking = ranking[eligible_index]
    chosen_local = np.argpartition(-eligible_ranking, kth=k - 1)[:k]
    chosen_local = chosen_local[np.argsort(-eligible_ranking[chosen_local], kind="mergesort")]
    chosen = eligible_index[chosen_local]

    rows: list[dict[str, Any]] = []
    returned_days: list[float | None] = []
    for rank, row_index in enumerate(chosen, start=1):
        days_value = None if not np.isfinite(days[int(row_index)]) else float(days[int(row_index)])
        returned_days.append(days_value)
        rows.append(
            {
                "customer_id": str(customer_ids[int(row_index)]),
                "similarity_score": float(scores[int(row_index)]),
                "like_score": float(likes[int(row_index)]),
                "rank": int(rank),
            }
        )
    diagnostics = _diagnostics(
        total_scored=n_customers,
        excluded_lt_60d=excluded_recent,
        eligible=eligible_count,
        returned_rows=rows,
        days_since=returned_days,
    )
    return rows, diagnostics


def _empty_diagnostics() -> dict[str, Any]:
    return _diagnostics(total_scored=0, excluded_lt_60d=0, eligible=0, returned_rows=[], days_since=[])


def _diagnostics(
    *,
    total_scored: int,
    excluded_lt_60d: int,
    eligible: int,
    returned_rows: Sequence[dict[str, Any]],
    days_since: Sequence[float | None],
) -> dict[str, Any]:
    buckets = Counter(recency_bucket(value) for value in days_since)
    return {
        "total_scored": int(total_scored),
        "excluded_lt_60d": int(excluded_lt_60d),
        "eligible": int(eligible),
        "returned": len(returned_rows),
        "returned_distribution": {
            BUCKET_60_180: int(buckets[BUCKET_60_180]),
            BUCKET_181_365: int(buckets[BUCKET_181_365]),
            BUCKET_1Y_3Y: int(buckets[BUCKET_1Y_3Y]),
            BUCKET_3Y_PLUS: int(buckets[BUCKET_3Y_PLUS]),
            BUCKET_UNKNOWN: int(buckets[BUCKET_UNKNOWN]),
        },
    }


def recency_distribution(days_values: Sequence[float | None]) -> dict[str, int]:
    buckets = Counter(recency_bucket(value) for value in days_values)
    return {
        BUCKET_LT_60: int(buckets[BUCKET_LT_60]),
        BUCKET_60_180: int(buckets[BUCKET_60_180]),
        BUCKET_181_365: int(buckets[BUCKET_181_365]),
        BUCKET_1Y_3Y: int(buckets[BUCKET_1Y_3Y]),
        BUCKET_3Y_PLUS: int(buckets[BUCKET_3Y_PLUS]),
        BUCKET_UNKNOWN: int(buckets[BUCKET_UNKNOWN]),
    }
