"""Dataset builder tests: leakage, visuals, missing models, temporal split."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.dataset_builder import (
    build_dataset,
    build_sequential_examples,
    combine_visual_embeddings,
    parse_embedding,
    prepare_purchases,
    resolve_split_bounds,
    split_examples,
)
from data.load_training_data import TrainingViews, write_snapshot
from data.schemas import CUSTOMER_COLUMNS, MODEL_COLUMNS, PURCHASE_COLUMNS, SchemaError, require_columns


def test_require_columns_fails_clearly() -> None:
    with pytest.raises(SchemaError, match="missing required columns: customer_id, model"):
        require_columns(["purchase_id"], ["purchase_id", "customer_id", "model"], "purchases")


def test_visual_mean_of_three_is_normalized() -> None:
    combined = combine_visual_embeddings(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert combined is not None
    expected = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)
    np.testing.assert_allclose(combined, expected, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(combined), 1.0, rtol=1e-6)


def test_visual_one_or_two_images_work() -> None:
    one = combine_visual_embeddings(np.array([3.0, 0.0, 4.0], dtype=np.float32), None, None)
    assert one is not None
    np.testing.assert_allclose(one, np.array([0.6, 0.0, 0.8], dtype=np.float32), rtol=1e-6)

    two = combine_visual_embeddings(
        np.array([1.0, 0.0], dtype=np.float32),
        None,
        np.array([0.0, 1.0], dtype=np.float32),
    )
    assert two is not None
    expected = np.array([0.5, 0.5], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(two, expected, rtol=1e-6)


def test_visual_missing_all_images_is_invalid() -> None:
    assert combine_visual_embeddings(None, None, None) is None
    assert combine_visual_embeddings(np.array([], dtype=np.float32)) is None
    assert parse_embedding(None) is None
    assert parse_embedding("null") is None
    assert parse_embedding("[]") is None


def test_parse_embedding_json_string() -> None:
    parsed = parse_embedding("[1, 2, 3]")
    assert parsed is not None
    np.testing.assert_array_equal(parsed, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def _embedding(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    return (array / np.linalg.norm(array)).tolist()


def _model_row(model: str, embedding: list[float] | None, **overrides: object) -> dict[str, object]:
    row = {column: None for column in MODEL_COLUMNS}
    row.update(
        {
            "model_id": overrides.get("model_id", hash(model) % 10_000),
            "model": model,
            "model_name": model,
            "main_category": "ballerina",
            "sub_category": "classic",
            "material": "leather",
            "material_1": "nappa",
            "material_2": None,
            "theme": "core",
            "color": "black",
            "season": "W26",
            "last_type": "angelis",
            "embedding_model": "google/siglip-base-patch16-224",
            "embedding_dimension": 4 if embedding is not None else None,
            "main_embedding": embedding,
            "pers_embedding": embedding,
            "side_embedding": embedding,
        }
    )
    row.update(overrides)
    return row


def _customer_row(customer_id: object, **overrides: object) -> dict[str, object]:
    row = {column: None for column in CUSTOMER_COLUMNS}
    row.update(
        {
            "customer_id": customer_id,
            "age_group": "35-44",
            "join_date": "2020-01-15",
            "join_shop_id": 1,
            "agent_id": 7,
            "purchase_count": 99,
            "invoice_count": 99,
            "unique_models": 99,
            "history_confidence": 0.9,
            "full_price_ratio": 0.1,
        }
    )
    row.update(overrides)
    return row


def _purchase_row(
    purchase_id: int,
    customer_id: object,
    model: str,
    purchase_date: str,
    **overrides: object,
) -> dict[str, object]:
    row = {column: None for column in PURCHASE_COLUMNS}
    row.update(
        {
            "purchase_id": purchase_id,
            "invoice_number": 1000 + purchase_id,
            "purchase_date": purchase_date,
            "customer_id": customer_id,
            "model": model,
            "sku": f"{model}-36",
            "size": 36,
            "quantity": 1,
            "discount": 0,
            "season": "W22",
            "last_name": "Test",
        }
    )
    row.update(overrides)
    return row


def _examples_from_rows(
    purchase_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    customer_rows: list[dict[str, object]] | None = None,
):
    from data.dataset_builder import customer_static_map, model_feature_map, model_visual_map

    purchases = prepare_purchases(pd.DataFrame(purchase_rows))
    models = pd.DataFrame(model_rows)
    customers = pd.DataFrame(customer_rows or [])
    examples, counts = build_sequential_examples(
        purchases,
        customer_static=customer_static_map(customers),
        model_features=model_feature_map(models),
        model_visuals=model_visual_map(models),
    )
    return examples, counts


def test_chronological_examples_have_no_future_leakage() -> None:
    embedding = _embedding([1.0, 0.0, 0.0, 0.0])
    purchases = [
        _purchase_row(1, 123, "A", "2022-01-01", season="W22", discount=0),
        _purchase_row(2, 123, "B", "2023-01-01", season="W23", discount=50),
        _purchase_row(3, 123, "C", "2024-01-01", season="W24", discount=0),
    ]
    models = [_model_row("A", embedding), _model_row("B", embedding), _model_row("C", embedding)]
    customers = [_customer_row(123, purchase_count=99, full_price_ratio=0.1)]
    examples, counts = _examples_from_rows(purchases, models, customers)

    assert len(examples) == 2
    assert counts["excluded_examples_missing_target_model"] == 0
    by_target = {row.target_model: row for row in examples.itertuples(index=False)}

    assert by_target["B"].history_model_ids == ["A"]
    assert "B" not in by_target["B"].history_model_ids
    assert "C" not in by_target["B"].history_model_ids

    assert by_target["C"].history_model_ids == ["A", "B"]
    assert "C" not in by_target["C"].history_model_ids

    behavior_b = json.loads(by_target["B"].customer_behavior_features)
    assert behavior_b["purchase_count_before"] == 1
    assert behavior_b["full_price_count_before"] == 1
    assert behavior_b["discount_count_before"] == 0
    assert behavior_b["full_price_ratio_before"] == 1.0
    assert behavior_b["season_counts_before"] == {"W22": 1}

    behavior_c = json.loads(by_target["C"].customer_behavior_features)
    assert behavior_c["purchase_count_before"] == 2
    assert behavior_c["discount_count_before"] == 1
    assert behavior_c["full_price_count_before"] == 1


def test_first_purchase_is_not_a_training_example() -> None:
    embedding = _embedding([0.0, 1.0, 0.0, 0.0])
    purchases = [
        _purchase_row(1, 5, "A", "2022-06-01"),
        _purchase_row(2, 5, "B", "2023-06-01"),
    ]
    models = [_model_row("A", embedding), _model_row("B", embedding)]
    examples, _counts = _examples_from_rows(purchases, models)
    assert list(examples["target_model"]) == ["B"]
    assert examples.iloc[0]["history_model_ids"] == ["A"]


def test_same_day_order_is_deterministic_by_purchase_id() -> None:
    embedding = _embedding([0.0, 0.0, 1.0, 0.0])
    purchases = [
        _purchase_row(20, 8, "B", "2023-01-01"),
        _purchase_row(10, 8, "A", "2023-01-01"),
        _purchase_row(30, 8, "C", "2023-01-02"),
    ]
    models = [_model_row("A", embedding), _model_row("B", embedding), _model_row("C", embedding)]
    examples, _counts = _examples_from_rows(purchases, models)
    targeting_b = examples.loc[examples["target_model"] == "B"].iloc[0]
    targeting_c = examples.loc[examples["target_model"] == "C"].iloc[0]
    assert targeting_b["history_model_ids"] == ["A"]
    assert targeting_c["history_model_ids"] == ["A", "B"]


def test_missing_target_model_is_reported_and_excluded() -> None:
    embedding = _embedding([1.0, 1.0, 0.0, 0.0])
    purchases = [
        _purchase_row(1, 1, "A", "2022-01-01"),
        _purchase_row(2, 1, "MISSING", "2023-01-01"),
        _purchase_row(3, 1, "C", "2024-01-01"),
    ]
    models = [_model_row("A", embedding), _model_row("C", embedding)]
    examples, counts = _examples_from_rows(purchases, models)
    assert counts["excluded_examples_missing_target_model"] == 1
    assert list(examples["target_model"]) == ["C"]
    assert examples.iloc[0]["history_model_ids"] == ["A", "MISSING"]


def test_target_without_visual_embedding_is_excluded() -> None:
    embedding = _embedding([1.0, 0.0, 1.0, 0.0])
    purchases = [
        _purchase_row(1, 1, "A", "2022-01-01"),
        _purchase_row(2, 1, "NOVIS", "2023-01-01"),
        _purchase_row(3, 1, "C", "2024-01-01"),
    ]
    models = [
        _model_row("A", embedding),
        _model_row("NOVIS", None, main_embedding=None, pers_embedding=None, side_embedding=None),
        _model_row("C", embedding),
    ]
    examples, counts = _examples_from_rows(purchases, models)
    assert counts["excluded_examples_missing_visual"] == 1
    assert list(examples["target_model"]) == ["C"]


def test_temporal_split_is_by_target_date_not_random() -> None:
    embedding = _embedding([1.0, 0.0, 0.0, 1.0])
    purchases = [
        _purchase_row(1, 1, "A", "2022-01-01"),
        _purchase_row(2, 1, "B", "2023-01-01"),
        _purchase_row(3, 1, "C", "2024-01-01"),
        _purchase_row(4, 1, "D", "2025-01-01"),
    ]
    models = [_model_row(code, embedding) for code in ("A", "B", "C", "D")]
    examples, _counts = _examples_from_rows(purchases, models)
    train, validation, test = split_examples(
        examples,
        pd.Timestamp("2023-06-01", tz="UTC"),
        pd.Timestamp("2024-06-01", tz="UTC"),
    )
    assert list(train["target_model"]) == ["B"]
    assert list(validation["target_model"]) == ["C"]
    assert list(test["target_model"]) == ["D"]


def test_default_split_uses_latest_purchase_date() -> None:
    dates = pd.to_datetime(
        ["2024-01-01", "2024-06-01", "2025-01-01", "2025-06-01"],
        utc=True,
    )
    train_end, validation_end, strategy = resolve_split_bounds(pd.Series(dates), None, None)
    latest = dates.max()
    assert strategy == "default_calendar"
    assert train_end == pd.Timestamp(latest) - pd.Timedelta(120, unit="D")
    assert validation_end == pd.Timestamp(latest) - pd.Timedelta(60, unit="D")


def test_write_snapshot_and_build_dataset(tmp_path: Path) -> None:
    embedding = _embedding([1.0, 0.0, 0.0, 0.0])
    views = TrainingViews(
        purchases=[
            _purchase_row(1, 11, "A", "2022-01-01"),
            _purchase_row(2, 11, "B", "2023-01-01"),
            _purchase_row(3, 11, "C", "2024-01-01"),
            _purchase_row(4, 12, "A", "2023-06-01"),
        ],
        customers=[_customer_row(11), _customer_row(12), _customer_row(13)],
        models=[_model_row("A", embedding), _model_row("B", embedding), _model_row("C", embedding)],
    )
    snapshot_dir = write_snapshot(
        views,
        snapshot_root=tmp_path,
        created_at=datetime(2026, 8, 27, 4, 0, 0, tzinfo=timezone.utc),
        db_name="payments",
    )
    assert snapshot_dir.name == "2026-08-27T040000"
    assert (snapshot_dir / "purchases.parquet").is_file()
    assert (snapshot_dir / "metadata.json").is_file()

    output_dir = build_dataset(
        snapshot_dir,
        train_end="2023-06-01",
        validation_end="2023-12-31",
    )
    metadata = json.loads((output_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(output_dir / "train.parquet")
    validation = pd.read_parquet(output_dir / "validation.parquet")
    test = pd.read_parquet(output_dir / "test.parquet")

    assert metadata["total_purchases_loaded"] == 4
    assert metadata["total_customers_loaded"] == 3
    assert metadata["total_models_loaded"] == 3
    assert metadata["customers_with_0_purchases"] == 1
    assert metadata["customers_with_1_purchase"] == 1
    assert metadata["customers_with_2plus_purchases"] == 1
    assert metadata["training_examples_produced"] == 2
    assert metadata["train_rows"] == 1
    assert metadata["validation_rows"] == 0
    assert metadata["test_rows"] == 1
    assert list(train["target_model"]) == ["B"]
    assert list(test["target_model"]) == ["C"]
    assert validation.empty
    assert isinstance(train.iloc[0]["target_visual_embedding"], (list, np.ndarray))
    np.testing.assert_allclose(np.linalg.norm(train.iloc[0]["target_visual_embedding"]), 1.0, rtol=1e-5)


def test_pre_2019_purchases_are_dropped_from_history() -> None:
    embedding = _embedding([1.0, 0.0, 0.0, 0.0])
    purchases = [
        _purchase_row(1, 1, "OLD", "2018-06-01"),
        _purchase_row(2, 1, "A", "2019-06-01"),
        _purchase_row(3, 1, "B", "2020-06-01"),
        _purchase_row(4, 2, "A", "2017-01-01"),
    ]
    models = [_model_row(code, embedding) for code in ("OLD", "A", "B")]
    examples, _counts = _examples_from_rows(purchases, models, [_customer_row(1, join_date="2010-01-01")])
    assert list(examples["target_model"]) == ["B"]
    assert examples.iloc[0]["history_model_ids"] == ["A"]
    behavior = json.loads(examples.iloc[0]["customer_behavior_features"])
    static = json.loads(examples.iloc[0]["customer_static_features"])
    assert behavior["purchase_count_before"] == 1
    assert static["tenure_days"] == pytest.approx(366, abs=2)


def test_single_post_2019_purchase_is_not_a_training_example() -> None:
    embedding = _embedding([0.0, 1.0, 0.0, 0.0])
    purchases = [
        _purchase_row(1, 9, "A", "2018-01-01"),
        _purchase_row(2, 9, "B", "2020-01-01"),
    ]
    models = [_model_row("A", embedding), _model_row("B", embedding)]
    examples, _counts = _examples_from_rows(purchases, models)
    assert examples.empty


def test_anonymous_and_zero_quantity_are_dropped() -> None:
    embedding = _embedding([0.0, 0.0, 1.0, 0.0])
    purchases = [
        _purchase_row(1, "99999999", "A", "2020-01-01"),
        _purchase_row(2, 3, "A", "2020-01-01", quantity=0),
        _purchase_row(3, 3, "B", "2021-01-01"),
        _purchase_row(4, 3, "C", "2022-01-01"),
    ]
    models = [_model_row(code, embedding) for code in ("A", "B", "C")]
    examples, _counts = _examples_from_rows(purchases, models)
    assert list(examples["customer_id"].unique()) == ["3"]
    assert examples.iloc[0]["history_model_ids"] == ["B"]

