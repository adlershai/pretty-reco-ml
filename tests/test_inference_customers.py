"""Inference CSV and current-customer encoding tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from inference.csv_models import ANONYMOUS_CUSTOMER_ID, read_model_codes
from inference.customers import build_current_customers
from inference.sanity import check_history_excludes_new_models
from training.dataset import CatalogModel


def test_read_model_codes_skips_blank_and_header(tmp_path: Path) -> None:
    path = tmp_path / "load.csv"
    path.write_text("model,name\n49166_008,Ella\n\n49166_013,Ella\n", encoding="utf-8")
    assert read_model_codes(path) == ["49166_008", "49166_013"]


def test_read_model_codes_headerless_first_column(tmp_path: Path) -> None:
    path = tmp_path / "load.csv"
    path.write_text("49166_008,Ella,899\n54158_003,Ella,1100\n", encoding="utf-8")
    assert read_model_codes(path) == ["49166_008", "54158_003"]


def _catalog(*codes: str) -> dict[str, CatalogModel]:
    visual = np.ones(8, dtype=np.float32) / np.sqrt(8)
    return {
        code: CatalogModel(code=code, features={"color": "black"}, visual=visual)
        for code in codes
    }


def test_build_current_customers_excludes_anonymous_and_new_models() -> None:
    purchases = pd.DataFrame(
        [
            {
                "purchase_id": 1,
                "invoice_number": "A",
                "purchase_date": "2026-01-01",
                "customer_id": "100",
                "model": "OLD1",
                "sku": "s1",
                "size": "37",
                "quantity": 1,
                "discount": 0,
                "season": "W26",
                "last_name": "Cohen",
            },
            {
                "purchase_id": 2,
                "invoice_number": "B",
                "purchase_date": "2026-02-01",
                "customer_id": "100",
                "model": "NEW1",
                "sku": "s2",
                "size": "38",
                "quantity": 1,
                "discount": 0,
                "season": "W26",
                "last_name": "Cohen",
            },
            {
                "purchase_id": 3,
                "invoice_number": "C",
                "purchase_date": "2026-01-15",
                "customer_id": ANONYMOUS_CUSTOMER_ID,
                "model": "OLD1",
                "sku": "s3",
                "size": "36",
                "quantity": 1,
                "discount": 0,
                "season": "W26",
                "last_name": "Walkin",
            },
            {
                "purchase_id": 4,
                "invoice_number": "D",
                "purchase_date": "2026-03-01",
                "customer_id": "200",
                "model": "NEW1",
                "sku": "s4",
                "size": "39",
                "quantity": 1,
                "discount": 0,
                "season": "W26",
                "last_name": "OnlyNew",
            },
            {
                "purchase_id": 5,
                "invoice_number": "E",
                "purchase_date": "2026-01-20",
                "customer_id": "400",
                "model": "GONE",
                "sku": "s5",
                "size": "38",
                "quantity": 1,
                "discount": 0,
                "season": "W26",
                "last_name": "OldSku",
            },
        ]
    )
    customers = pd.DataFrame(
        [
            {
                "customer_id": "100",
                "age_group": "30-39",
                "join_date": "2024-01-01",
                "preferred_size": "37",
                "history_confidence": "high",
                "join_shop_id": 1,
                "agent_id": 2,
                "birthday_year": 1990,
            },
            {
                "customer_id": "200",
                "age_group": "18-29",
                "join_date": "2025-01-01",
                "preferred_size": "39",
                "history_confidence": "low",
                "join_shop_id": 1,
                "agent_id": 2,
                "birthday_year": 1998,
            },
            {
                "customer_id": "300",
                "age_group": "40-49",
                "join_date": "2020-01-01",
                "preferred_size": "36",
                "history_confidence": "high",
                "join_shop_id": 1,
                "agent_id": 2,
                "birthday_year": 1980,
            },
            {
                "customer_id": "400",
                "age_group": "50-59",
                "join_date": "2018-01-01",
                "preferred_size": "38",
                "history_confidence": "low",
                "join_shop_id": 1,
                "agent_id": 2,
                "birthday_year": 1970,
            },
        ]
    )
    records, stats = build_current_customers(
        purchases,
        customers,
        _catalog("OLD1"),
        new_model_codes={"NEW1"},
        now=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert [row["customer_id"] for row in records] == ["100"]
    assert records[0]["history_model_ids"] == ["OLD1"]
    assert records[0]["customer_name"] == "Cohen"
    assert records[0]["last_purchased_size"] == "38"
    assert records[0]["preferred_size"] == "37"
    check_history_excludes_new_models(records, {"NEW1"})
    assert stats["encoded_customers"] == 1
    assert stats["reason_counts"]["anonymous"] == 1
    assert stats["reason_counts"]["no_purchases"] == 2
    assert stats["reason_counts"]["no_catalog_history"] == 1
    assert stats["excluded_customers"] == 4
    assert ANONYMOUS_CUSTOMER_ID not in {row["customer_id"] for row in records}
