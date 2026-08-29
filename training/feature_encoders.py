"""Fit/persist categorical vocabularies and numeric scaling. Unseen → UNK."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from training.config import (
    MODEL_CATEGORICAL_FIELDS,
    NUMERIC_FEATURE_NAMES,
    PAD_TOKEN,
    UNKNOWN_TOKEN,
)


def categorical_string(value: Any) -> str:
    if value is None:
        return UNKNOWN_TOKEN
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return UNKNOWN_TOKEN
        if value.is_integer():
            return str(int(value))
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return UNKNOWN_TOKEN
        if number.is_integer():
            return str(int(number))
        return str(number)
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat", "null", "<na>"}:
        return UNKNOWN_TOKEN
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def parse_json_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        return json.loads(text)
    return {}


def numeric_features(static: Mapping[str, Any], behavior: Mapping[str, Any]) -> np.ndarray:
    purchase_count = max(_as_float(behavior.get("purchase_count_before")), 0.0)
    history_length = max(_as_float(behavior.get("history_length"), purchase_count), 0.0)
    recency = max(_as_float(behavior.get("recency_days")), 0.0)
    tenure_raw = static.get("tenure_days")
    tenure_missing = 1.0 if tenure_raw is None else 0.0
    tenure = max(_as_float(tenure_raw), 0.0)
    return np.array(
        [
            math.log1p(purchase_count),
            math.log1p(history_length),
            math.log1p(recency),
            math.log1p(tenure),
            tenure_missing,
            float(np.clip(_as_float(behavior.get("full_price_ratio_before")), 0.0, 1.0)),
            float(np.clip(_as_float(behavior.get("discount_ratio_before")), 0.0, 1.0)),
            float(np.clip(_as_float(behavior.get("avg_discount_before")), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


class Vocabulary:
    def __init__(self, token_to_index: dict[str, int] | None = None) -> None:
        self.token_to_index = token_to_index or {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}

    def fit(self, values: Iterable[Any]) -> None:
        for value in values:
            token = categorical_string(value)
            if token not in self.token_to_index:
                self.token_to_index[token] = len(self.token_to_index)

    def encode(self, value: Any) -> int:
        token = categorical_string(value)
        return self.token_to_index.get(token, self.token_to_index[UNKNOWN_TOKEN])

    @property
    def size(self) -> int:
        return len(self.token_to_index)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_index)

    @classmethod
    def from_dict(cls, payload: Mapping[str, int]) -> Vocabulary:
        return cls({str(key): int(value) for key, value in payload.items()})


class FeatureEncoders:
    def __init__(
        self,
        categoricals: dict[str, Vocabulary] | None = None,
        age_group: Vocabulary | None = None,
        numeric_mean: np.ndarray | None = None,
        numeric_std: np.ndarray | None = None,
        visual_input_dim: int = 768,
    ) -> None:
        self.categoricals = categoricals or {field: Vocabulary() for field in MODEL_CATEGORICAL_FIELDS}
        self.age_group = age_group or Vocabulary()
        self.numeric_mean = numeric_mean if numeric_mean is not None else np.zeros(len(NUMERIC_FEATURE_NAMES), dtype=np.float32)
        self.numeric_std = numeric_std if numeric_std is not None else np.ones(len(NUMERIC_FEATURE_NAMES), dtype=np.float32)
        self.visual_input_dim = int(visual_input_dim)

    def fit_models(self, feature_rows: Sequence[Mapping[str, Any]]) -> None:
        for field in MODEL_CATEGORICAL_FIELDS:
            self.categoricals[field].fit(row.get(field) for row in feature_rows)

    def fit_age_groups(self, values: Iterable[Any]) -> None:
        self.age_group.fit(values)

    def fit_numeric(self, rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
        matrix = np.stack([numeric_features(static, behavior) for static, behavior in rows], axis=0)
        self.numeric_mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        self.numeric_std = std

    def encode_model_categoricals(self, features: Mapping[str, Any]) -> dict[str, int]:
        return {field: self.categoricals[field].encode(features.get(field)) for field in MODEL_CATEGORICAL_FIELDS}

    def encode_age(self, value: Any) -> int:
        return self.age_group.encode(value)

    def encode_numeric(self, static: Mapping[str, Any], behavior: Mapping[str, Any]) -> np.ndarray:
        raw = numeric_features(static, behavior)
        return ((raw - self.numeric_mean) / self.numeric_std).astype(np.float32)

    def encode_visual(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != self.visual_input_dim or not np.isfinite(array).all():
            return np.zeros(self.visual_input_dim, dtype=np.float32)
        return array

    def to_json(self) -> dict[str, Any]:
        return {
            "categoricals": {field: vocab.to_dict() for field, vocab in self.categoricals.items()},
            "age_group": self.age_group.to_dict(),
            "numeric_mean": self.numeric_mean.tolist(),
            "numeric_std": self.numeric_std.tolist(),
            "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
            "visual_input_dim": self.visual_input_dim,
        }

    def dump(self, path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> FeatureEncoders:
        categoricals = {
            field: Vocabulary.from_dict(values)
            for field, values in payload["categoricals"].items()
        }
        return cls(
            categoricals=categoricals,
            age_group=Vocabulary.from_dict(payload["age_group"]),
            numeric_mean=np.asarray(payload["numeric_mean"], dtype=np.float32),
            numeric_std=np.asarray(payload["numeric_std"], dtype=np.float32),
            visual_input_dim=int(payload.get("visual_input_dim", 768)),
        )

    @classmethod
    def load(cls, path) -> FeatureEncoders:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
