"""Load snapshot parquet rows into tensors for two-tower training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset_builder import combine_visual_embeddings, parse_embedding
from data.schemas import DATASET_TEST, DATASET_TRAIN, DATASET_VALIDATION, MODEL_FEATURE_COLUMNS, SNAPSHOT_MODELS
from training.config import MODEL_CATEGORICAL_FIELDS, TrainConfig
from training.feature_encoders import FeatureEncoders, parse_json_map


@dataclass(frozen=True)
class CatalogModel:
    code: str
    features: dict[str, Any]
    visual: np.ndarray


def resolve_snapshot_paths(snapshot: Path) -> tuple[Path, Path]:
    snapshot = snapshot.resolve()
    if (snapshot / DATASET_TRAIN).is_file():
        dataset_dir = snapshot
        snapshot_dir = snapshot
        cursor = snapshot
        for _ in range(4):
            if (cursor / SNAPSHOT_MODELS).is_file():
                snapshot_dir = cursor
                break
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        return snapshot_dir, dataset_dir
    dataset_dir = snapshot / "dataset"
    if not (dataset_dir / DATASET_TRAIN).is_file():
        raise FileNotFoundError(f"training parquet not found under {snapshot}")
    return snapshot, dataset_dir


def load_catalog(snapshot_dir: Path) -> dict[str, CatalogModel]:
    frame = pd.read_parquet(snapshot_dir / SNAPSHOT_MODELS, engine="pyarrow")
    catalog: dict[str, CatalogModel] = {}
    keep = ("model", "model_id", "model_name", *MODEL_FEATURE_COLUMNS)
    present = [column for column in keep if column in frame.columns]
    for row in frame.to_dict("records"):
        code = str(row.get("model") or "").strip()
        if not code:
            continue
        visual = combine_visual_embeddings(
            parse_embedding(row.get("main_embedding")),
            parse_embedding(row.get("pers_embedding")),
            parse_embedding(row.get("side_embedding")),
        )
        if visual is None:
            continue
        features = {key: row.get(key) for key in present}
        catalog[code] = CatalogModel(code=code, features=features, visual=visual.astype(np.float32))
    return catalog


def load_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    names = {"train": DATASET_TRAIN, "validation": DATASET_VALIDATION, "test": DATASET_TEST}
    path = dataset_dir / names[split]
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, engine="pyarrow")


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        encoders: FeatureEncoders,
        catalog: Mapping[str, CatalogModel],
        max_history: int,
    ) -> None:
        self.encoders = encoders
        self.catalog = catalog
        self.max_history = int(max_history)
        self.rows = [self._prepare(row) for row in frame.to_dict("records") if str(row.get("target_model") or "") in catalog]

    def _prepare(self, row: Mapping[str, Any]) -> dict[str, Any]:
        target_code = str(row["target_model"])
        target = self.catalog[target_code]
        target_features = parse_json_map(row.get("target_model_features")) or target.features
        static = parse_json_map(row.get("customer_static_features"))
        behavior = parse_json_map(row.get("customer_behavior_features"))
        history_ids = _as_string_list(row.get("history_model_ids"))
        history_visuals: list[np.ndarray] = []
        history_cats: list[dict[str, int]] = []
        for code in history_ids[-self.max_history :]:
            item = self.catalog.get(code)
            if item is None:
                continue
            history_visuals.append(item.visual)
            history_cats.append(self.encoders.encode_model_categoricals(item.features))
        return {
            "target_visual": self.encoders.encode_visual(
                row.get("target_visual_embedding") if row.get("target_visual_embedding") is not None else target.visual
            ),
            "target_cats": self.encoders.encode_model_categoricals(target_features),
            "history_visuals": history_visuals,
            "history_cats": history_cats,
            "numeric": self.encoders.encode_numeric(static, behavior),
            "age_id": self.encoders.encode_age(static.get("age_group")),
            "target_code": target_code,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_batch(rows: Sequence[Mapping[str, Any]], config: TrainConfig) -> dict[str, torch.Tensor | list[str]]:
    batch_size = len(rows)
    max_len = min(config.max_history, max((len(row["history_visuals"]) for row in rows), default=1))
    max_len = max(max_len, 1)
    visual_dim = config.visual_input_dim
    history_visual = np.zeros((batch_size, max_len, visual_dim), dtype=np.float32)
    history_mask = np.zeros((batch_size, max_len), dtype=np.bool_)
    history_cats = {field: np.zeros((batch_size, max_len), dtype=np.int64) for field in MODEL_CATEGORICAL_FIELDS}
    target_visual = np.zeros((batch_size, visual_dim), dtype=np.float32)
    target_cats = {field: np.zeros(batch_size, dtype=np.int64) for field in MODEL_CATEGORICAL_FIELDS}
    numeric = np.zeros((batch_size, len(rows[0]["numeric"])), dtype=np.float32)
    age_id = np.zeros(batch_size, dtype=np.int64)
    target_codes: list[str] = []
    for index, row in enumerate(rows):
        length = min(len(row["history_visuals"]), max_len)
        if length:
            stacked = np.stack(row["history_visuals"][-length :], axis=0)
            history_visual[index, :length] = stacked
            history_mask[index, :length] = True
            for offset, cats in enumerate(row["history_cats"][-length :]):
                for field in MODEL_CATEGORICAL_FIELDS:
                    history_cats[field][index, offset] = cats[field]
        target_visual[index] = row["target_visual"]
        for field in MODEL_CATEGORICAL_FIELDS:
            target_cats[field][index] = row["target_cats"][field]
        numeric[index] = row["numeric"]
        age_id[index] = row["age_id"]
        target_codes.append(str(row["target_code"]))
    batch: dict[str, torch.Tensor | list[str]] = {
        "history_visual": torch.from_numpy(history_visual),
        "history_mask": torch.from_numpy(history_mask),
        "target_visual": torch.from_numpy(target_visual),
        "numeric": torch.from_numpy(numeric),
        "age_id": torch.from_numpy(age_id),
        "target_codes": target_codes,
    }
    for field in MODEL_CATEGORICAL_FIELDS:
        batch[f"history_cat_{field}"] = torch.from_numpy(history_cats[field])
        batch[f"target_cat_{field}"] = torch.from_numpy(target_cats[field])
    return batch


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in list(value) if str(item)]
