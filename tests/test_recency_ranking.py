"""Recency exclusion tests. Recency never changes like_score."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from inference.csv_models import ANONYMOUS_CUSTOMER_ID
from inference.ranking import rank_customers_for_model
from inference.recency import (
    days_since_last_purchase,
    is_excluded_recent,
    rank_customers_with_recency,
)


TODAY = date(2026, 8, 31)


def _iso(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def test_10_day_customer_is_excluded() -> None:
    assert days_since_last_purchase(_iso(10), now=TODAY) == 10
    assert is_excluded_recent(10) is True


def test_59_day_customer_is_excluded() -> None:
    assert is_excluded_recent(59) is True
    rows, diagnostics = rank_customers_with_recency(
        np.array([0.9], dtype=np.float32),
        ["1"],
        [_iso(59)],
        top_k=10,
        now=TODAY,
    )
    assert rows == []
    assert diagnostics["excluded_lt_60d"] == 1
    assert diagnostics["eligible"] == 0


def test_60_day_customer_is_eligible() -> None:
    assert is_excluded_recent(60) is False
    rows, _diagnostics = rank_customers_with_recency(
        np.array([0.4], dtype=np.float32),
        ["1"],
        [_iso(60)],
        top_k=10,
        now=TODAY,
    )
    assert len(rows) == 1
    assert rows[0]["customer_id"] == "1"


def test_ranking_uses_all_eligible_customers_before_top_n() -> None:
    n_recent = 100
    n_dormant = 40
    similarities = np.concatenate(
        [
            np.full(n_recent, 0.9, dtype=np.float32),
            np.full(n_dormant, 0.1, dtype=np.float32),
        ]
    )
    ids = [str(i) for i in range(n_recent + n_dormant)]
    dates = [_iso(10)] * n_recent + [_iso(1500)] * n_dormant
    similarity_then_filter = rank_customers_for_model(similarities, ids, top_k=100)
    remaining = [
        row
        for row in similarity_then_filter
        if not is_excluded_recent(days_since_last_purchase(dates[int(row["customer_id"])], now=TODAY))
    ]
    assert len(remaining) == 0

    ranked, diagnostics = rank_customers_with_recency(
        similarities,
        ids,
        dates,
        top_k=100,
        now=TODAY,
    )
    assert diagnostics["total_scored"] == 140
    assert diagnostics["excluded_lt_60d"] == 100
    assert diagnostics["eligible"] == 40
    assert len(ranked) == 40
    assert all(row["customer_id"] not in {str(i) for i in range(n_recent)} for row in ranked)
