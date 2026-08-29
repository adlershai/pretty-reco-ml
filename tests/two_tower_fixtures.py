"""Shared tiny two-tower fixtures for unit tests."""

from __future__ import annotations

import numpy as np

from training.config import MODEL_CATEGORICAL_FIELDS, TrainConfig
from training.dataset import collate_batch
from training.feature_encoders import FeatureEncoders
from training.two_tower import TwoTowerModel

VISUAL_DIM = 8


def unit_visual(seed: int, dim: int = VISUAL_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def base_features(**overrides: object) -> dict[str, object]:
    row = {field: "unknown" for field in MODEL_CATEGORICAL_FIELDS}
    row.update(
        {
            "main_category": "WOMEN",
            "sub_category": "Shoes",
            "material": "Leather",
            "material_1": "nappa",
            "material_2": "",
            "theme": "core",
            "color": "black",
            "season": "W26",
            "last_type": "angelis",
        }
    )
    row.update(overrides)
    return row


def make_encoders(dim: int = VISUAL_DIM) -> FeatureEncoders:
    encoders = FeatureEncoders(visual_input_dim=dim)
    encoders.fit_models(
        [
            base_features(),
            base_features(color="red", last_type="rosario", main_category="KIDS"),
        ]
    )
    encoders.fit_age_groups(["18-29", "30-39"])
    encoders.fit_numeric(
        [
            (
                {"age_group": "18-29", "tenure_days": 10},
                {
                    "purchase_count_before": 1,
                    "history_length": 1,
                    "recency_days": 2,
                    "full_price_ratio_before": 1.0,
                    "discount_ratio_before": 0.0,
                    "avg_discount_before": 0.0,
                },
            ),
            (
                {"age_group": "30-39", "tenure_days": 400},
                {
                    "purchase_count_before": 5,
                    "history_length": 5,
                    "recency_days": 40,
                    "full_price_ratio_before": 0.4,
                    "discount_ratio_before": 0.6,
                    "avg_discount_before": 0.2,
                },
            ),
        ]
    )
    return encoders


def make_config(embedding_dim: int = 8, **overrides: object) -> TrainConfig:
    values = {
        "hidden_dim": 16,
        "visual_input_dim": VISUAL_DIM,
        "max_history": 4,
        "batch_size": 4,
        "dropout": 0.0,
        "epochs": 1,
        "patience": 1,
    }
    values.update(overrides)
    return TrainConfig.from_embedding_dim(embedding_dim, **values)


def make_model(embedding_dim: int = 8) -> tuple[TwoTowerModel, FeatureEncoders, TrainConfig]:
    encoders = make_encoders()
    config = make_config(embedding_dim)
    return TwoTowerModel(config, encoders), encoders, config


def example_model(code: str, seed: int, **feature_overrides: object) -> dict[str, object]:
    features = base_features(**feature_overrides)
    features["model"] = code
    features["visual_embedding"] = unit_visual(seed)
    features["visual"] = features["visual_embedding"]
    return features


def customer_row(encoders: FeatureEncoders, history: list[dict], target: dict, age: str = "18-29") -> dict:
    return {
        "target_visual": np.asarray(target["visual_embedding"], dtype=np.float32),
        "target_cats": encoders.encode_model_categoricals(target),
        "history_visuals": [np.asarray(item["visual_embedding"], dtype=np.float32) for item in history],
        "history_cats": [encoders.encode_model_categoricals(item) for item in history],
        "numeric": encoders.encode_numeric(
            {"age_group": age, "tenure_days": 30},
            {
                "purchase_count_before": len(history),
                "history_length": len(history),
                "recency_days": 7,
                "full_price_ratio_before": 1.0,
                "discount_ratio_before": 0.0,
                "avg_discount_before": 0.0,
            },
        ),
        "age_id": encoders.encode_age(age),
        "target_code": str(target.get("model", "target")),
    }


def tiny_batch(encoders: FeatureEncoders, config: TrainConfig):
    a = example_model("A", 1)
    b = example_model("B", 2, color="red")
    c = example_model("C", 3, last_type="rosario")
    d = example_model("D", 4, main_category="KIDS")
    rows = [
        customer_row(encoders, [a], b),
        customer_row(encoders, [a, b], c),
        customer_row(encoders, [c], d),
        customer_row(encoders, [b], a),
    ]
    return collate_batch(rows, config)
