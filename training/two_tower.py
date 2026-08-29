"""Two-tower recommender: independently callable customer and model encoders."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import TrainConfig
from training.customer_tower import CustomerTower
from training.feature_encoders import FeatureEncoders, parse_json_map
from training.model_tower import ModelTower, categoricals_from_batch


class TwoTowerModel(nn.Module):
    def __init__(self, config: TrainConfig, encoders: FeatureEncoders) -> None:
        super().__init__()
        self.config = config
        self.encoders = encoders
        self.model_tower = ModelTower(config, encoders)
        self.customer_tower = CustomerTower(config, encoders, self.model_tower)

    def encode_models(self, batch: dict, device: torch.device | None = None) -> torch.Tensor:
        visual = batch["target_visual"]
        cats = categoricals_from_batch(batch, "target_cat_")
        if device is not None:
            visual = visual.to(device)
            cats = {field: tensor.to(device) for field, tensor in cats.items()}
        return self.model_tower(visual, cats)

    def encode_customers(self, batch: dict, device: torch.device | None = None) -> torch.Tensor:
        return self.customer_tower(batch, device=device)

    def forward(self, batch: dict, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        customers = self.encode_customers(batch, device=device)
        models = self.encode_models(batch, device=device)
        return customers, models

    def similarity(self, customer_vectors: torch.Tensor, model_vectors: torch.Tensor) -> torch.Tensor:
        return customer_vectors @ model_vectors.transpose(0, 1)

    def encode_model(self, model_input: Mapping[str, Any], device: torch.device | None = None) -> np.ndarray:
        self.eval()
        visual = torch.from_numpy(self.encoders.encode_visual(_visual_from_input(model_input))).unsqueeze(0)
        encoded_cats = self.encoders.encode_model_categoricals(model_input)
        cats = {field: torch.tensor([encoded_cats[field]], dtype=torch.long) for field in encoded_cats}
        if device is not None:
            visual = visual.to(device)
            cats = {field: tensor.to(device) for field, tensor in cats.items()}
            tower = self.model_tower.to(device)
        else:
            tower = self.model_tower
        with torch.inference_mode():
            vector = tower(visual, cats)
        return vector.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def encode_customer(self, customer_input: Mapping[str, Any], device: torch.device | None = None) -> np.ndarray:
        from training.dataset import collate_batch

        self.eval()
        row = _customer_row_from_input(customer_input, self.encoders, self.config.max_history)
        batch = collate_batch([row], self.config)
        with torch.inference_mode():
            vector = self.encode_customers(batch, device=device)
        return vector.squeeze(0).detach().cpu().numpy().astype(np.float32)


def in_batch_softmax_loss(
    customer_vectors: torch.Tensor,
    model_vectors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = (customer_vectors @ model_vectors.transpose(0, 1)) / temperature
    labels = torch.arange(customer_vectors.size(0), device=customer_vectors.device)
    return F.cross_entropy(logits, labels)


def _visual_from_input(model_input: Mapping[str, Any]) -> Any:
    if model_input.get("visual_embedding") is not None:
        return model_input["visual_embedding"]
    if model_input.get("target_visual_embedding") is not None:
        return model_input["target_visual_embedding"]
    return model_input.get("visual")


def _customer_row_from_input(
    customer_input: Mapping[str, Any],
    encoders: FeatureEncoders,
    max_history: int,
) -> dict[str, Any]:
    static = parse_json_map(customer_input.get("customer_static_features")) or dict(customer_input.get("static") or {})
    behavior = parse_json_map(customer_input.get("customer_behavior_features")) or dict(customer_input.get("behavior") or {})
    history_models = list(customer_input.get("history_models") or [])[-max_history:]
    history_visuals = []
    history_cats = []
    for item in history_models:
        history_visuals.append(encoders.encode_visual(_visual_from_input(item)))
        history_cats.append(encoders.encode_model_categoricals(item))
    placeholder_visual = (
        history_visuals[0]
        if history_visuals
        else np.zeros(encoders.visual_input_dim, dtype=np.float32)
    )
    return {
        "target_visual": placeholder_visual,
        "target_cats": history_cats[0] if history_cats else encoders.encode_model_categoricals({}),
        "history_visuals": history_visuals,
        "history_cats": history_cats,
        "numeric": encoders.encode_numeric(static, behavior),
        "age_id": encoders.encode_age(static.get("age_group") or customer_input.get("age_group")),
        "target_code": "",
    }
