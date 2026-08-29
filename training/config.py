"""Two-tower training configuration. Compact defaults for ~45K examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

MODEL_CATEGORICAL_FIELDS = (
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

NUMERIC_FEATURE_NAMES = (
    "log_purchase_count",
    "log_history_length",
    "log_recency_days",
    "log_tenure_days",
    "tenure_missing",
    "full_price_ratio",
    "discount_ratio",
    "avg_discount",
)

VISUAL_INPUT_DIM = 768
DEFAULT_EMBEDDING_DIMS = (64, 128)
EXPLORATORY_EMBEDDING_DIM = 256
RECALL_MARGIN = 0.01


def hidden_dim_for(embedding_dim: int) -> int:
    if embedding_dim <= 64:
        return 128
    return 256


def categorical_embedding_dim(vocab_size: int) -> int:
    return int(min(16, max(4, (vocab_size + 1) // 2)))


@dataclass
class TrainConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    visual_input_dim: int = VISUAL_INPUT_DIM
    categorical_fields: tuple[str, ...] = MODEL_CATEGORICAL_FIELDS
    numeric_feature_names: tuple[str, ...] = NUMERIC_FEATURE_NAMES
    max_history: int = 20
    dropout: float = 0.1
    temperature: float = 0.1
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 3
    seed: int = 42
    num_threads: int = 2
    age_embedding_dim: int = 8
    loss: str = "in_batch_sampled_softmax"
    normalize: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["categorical_fields"] = list(self.categorical_fields)
        payload["numeric_feature_names"] = list(self.numeric_feature_names)
        return payload

    @classmethod
    def from_embedding_dim(cls, embedding_dim: int, **overrides: Any) -> TrainConfig:
        values = dict(overrides)
        values.setdefault("hidden_dim", hidden_dim_for(embedding_dim))
        return cls(embedding_dim=int(embedding_dim), **values)
