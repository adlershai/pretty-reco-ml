"""Shoe → customer cosine ranking from L2-normalized two-tower vectors."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

RANK_LIMITS = (10, 25, 50, 100)
EXPORT_TOP_K = 100
CUSTOMER_TOP_MODELS = 6

HISTORICAL_EVAL_LIMITATION = (
    "Historical shoe→customer evaluation was skipped. True as-of-T ranking would "
    "re-encode every candidate customer from history available before each target "
    "purchase T (no future leakage). That is a separate evaluation architecture; "
    "this phase ranks current customer vectors against new cold-start shoes."
)


def score_matrix(customer_vectors: np.ndarray, model_vectors: np.ndarray) -> np.ndarray:
    return customer_vectors @ model_vectors.T


def rank_customers_for_model(
    scores: np.ndarray,
    customer_ids: Sequence[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank customers for a single model column (1D scores)."""
    column = np.asarray(scores, dtype=np.float32).reshape(-1)
    n_customers = column.shape[0]
    if n_customers == 0:
        return []
    k = min(int(top_k), n_customers)
    index = np.argpartition(-column, kth=k - 1)[:k]
    index = index[np.argsort(-column[index], kind="mergesort")]
    return [
        {
            "customer_id": str(customer_ids[int(row_index)]),
            "similarity_score": float(column[int(row_index)]),
            "rank": int(rank),
        }
        for rank, row_index in enumerate(index, start=1)
    ]


def rank_customers_for_models(
    scores: np.ndarray,
    customer_ids: Sequence[str],
    model_codes: Sequence[str],
    *,
    top_k: int = EXPORT_TOP_K,
) -> list[dict[str, Any]]:
    n_customers, n_models = scores.shape
    if n_customers == 0 or n_models == 0:
        return []
    k = min(int(top_k), n_customers)
    ranked: list[dict[str, Any]] = []
    for column, model in enumerate(model_codes):
        col = scores[:, column]
        index = np.argpartition(-col, kth=k - 1)[:k]
        index = index[np.argsort(-col[index], kind="mergesort")]
        for rank, row_index in enumerate(index, start=1):
            ranked.append(
                {
                    "model": str(model),
                    "customer_rank": int(rank),
                    "customer_id": str(customer_ids[int(row_index)]),
                    "similarity_score": float(col[int(row_index)]),
                    "customer_index": int(row_index),
                }
            )
    ranked.sort(key=lambda row: (row["model"], row["customer_rank"]))
    return ranked


def rank_models_for_customers(
    scores: np.ndarray,
    customer_ids: Sequence[str],
    model_codes: Sequence[str],
    *,
    top_k: int = CUSTOMER_TOP_MODELS,
) -> list[dict[str, Any]]:
    n_customers, n_models = scores.shape
    if n_customers == 0 or n_models == 0:
        return []
    k = min(int(top_k), n_models)
    ranked: list[dict[str, Any]] = []
    for row_index, customer_id in enumerate(customer_ids):
        row = scores[row_index]
        index = np.argpartition(-row, kth=k - 1)[:k]
        index = index[np.argsort(-row[index], kind="mergesort")]
        for rank, column in enumerate(index, start=1):
            ranked.append(
                {
                    "customer_id": str(customer_id),
                    "model": str(model_codes[int(column)]),
                    "rank_for_customer": int(rank),
                    "similarity_score": float(row[int(column)]),
                }
            )
    ranked.sort(key=lambda row: (row["customer_id"], row["rank_for_customer"]))
    return ranked


def enrich_customer_ranking(
    ranked: list[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(record["customer_id"]): record for record in records}
    enriched = []
    for row in ranked:
        record = by_id.get(row["customer_id"]) or {}
        enriched.append(
            {
                "model": row["model"],
                "customer_rank": row["customer_rank"],
                "customer_id": row["customer_id"],
                "customer_name": record.get("customer_name"),
                "similarity_score": row["similarity_score"],
                "history_length": record.get("history_length"),
                "last_purchase_date": record.get("last_purchase_date"),
                "preferred_size": record.get("preferred_size"),
                "last_purchased_size": record.get("last_purchased_size"),
            }
        )
    return enriched


def per_model_summary(
    ranked: list[dict[str, Any]],
    *,
    top_50: int = 50,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_model[str(row["model"])].append(row)
    summaries = []
    for model, rows in sorted(by_model.items()):
        ordered = sorted(rows, key=lambda item: int(item["customer_rank"]))
        scores = np.array([float(item["similarity_score"]) for item in ordered], dtype=np.float32)
        top = scores[: min(top_50, len(scores))]
        summaries.append(
            {
                "model": model,
                "top_10_customers": [item["customer_id"] for item in ordered if int(item["customer_rank"]) <= 10],
                "top_25_customers": [item["customer_id"] for item in ordered if int(item["customer_rank"]) <= 25],
                "top_50_customers": [item["customer_id"] for item in ordered if int(item["customer_rank"]) <= 50],
                "score_max": float(top[0]) if len(top) else None,
                "score_median_top_50": float(np.median(top)) if len(top) else None,
                "score_min_top_50": float(top[-1]) if len(top) else None,
            }
        )
    return summaries


def overlap_stats(ranked: list[dict[str, Any]], *, top_n: int = 25) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in ranked:
        if int(row["customer_rank"]) <= top_n:
            counts[str(row["customer_id"])] += 1
    unique = len(counts)
    values = list(counts.values())
    highest = max(values) if values else 0
    average = (sum(values) / unique) if unique else 0.0
    top_customer, top_count = counts.most_common(1)[0] if counts else (None, 0)
    return {
        "top_n": top_n,
        "unique_customers": unique,
        "average_models_per_selected_customer": average,
        "max_models_assigned_to_one_customer": highest,
        "highest_repeated_customer_id": top_customer,
        "highest_repeated_customer_count": top_count,
    }


def unique_customers_in_top(ranked: list[dict[str, Any]], top_n: int) -> int:
    return len({row["customer_id"] for row in ranked if int(row["customer_rank"]) <= top_n})
