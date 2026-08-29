"""Save and reload the_pretty_model_v1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from training.config import TrainConfig
from training.feature_encoders import FeatureEncoders
from training.two_tower import TwoTowerModel


def save_artifact(
    *,
    artifact_dir: Path,
    model: TwoTowerModel,
    config: TrainConfig,
    encoders: FeatureEncoders,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    encoders.dump(artifact_dir / "feature_encoders.json")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_artifact(artifact_dir: Path, device: torch.device | None = None) -> TwoTowerModel:
    payload = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    config = TrainConfig.from_embedding_dim(
        int(payload["embedding_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        visual_input_dim=int(payload.get("visual_input_dim", 768)),
        max_history=int(payload.get("max_history", 20)),
        dropout=float(payload.get("dropout", 0.1)),
        temperature=float(payload.get("temperature", 0.1)),
        batch_size=int(payload.get("batch_size", 256)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        weight_decay=float(payload.get("weight_decay", 1e-4)),
        epochs=int(payload.get("epochs", 15)),
        patience=int(payload.get("patience", 3)),
        seed=int(payload.get("seed", 42)),
        num_threads=int(payload.get("num_threads", 2)),
        age_embedding_dim=int(payload.get("age_embedding_dim", 8)),
        loss=str(payload.get("loss", "in_batch_sampled_softmax")),
        normalize=bool(payload.get("normalize", True)),
    )
    encoders = FeatureEncoders.load(artifact_dir / "feature_encoders.json")
    model = TwoTowerModel(config, encoders)
    state = torch.load(artifact_dir / "model.pt", map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return model
