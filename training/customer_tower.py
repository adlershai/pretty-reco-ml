"""Customer tower: history of model vectors + behavioral/static features → C."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import TrainConfig
from training.feature_encoders import FeatureEncoders
from training.model_tower import ModelTower, categoricals_from_batch


class CustomerTower(nn.Module):
    def __init__(self, config: TrainConfig, encoders: FeatureEncoders, model_tower: ModelTower) -> None:
        super().__init__()
        self.model_tower = model_tower
        self.embedding_dim = int(config.embedding_dim)
        self.age_embedding = nn.Embedding(encoders.age_group.size, config.age_embedding_dim, padding_idx=0)
        self.attention = nn.Linear(config.embedding_dim, 1)
        numeric_dim = len(config.numeric_feature_names)
        fused = config.embedding_dim + config.age_embedding_dim + numeric_dim
        self.fused = nn.Sequential(
            nn.Linear(fused, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def encode_history(self, batch: dict, device: torch.device | None = None) -> torch.Tensor:
        visual = batch["history_visual"]
        mask = batch["history_mask"]
        if device is not None:
            visual = visual.to(device)
            mask = mask.to(device)
        batch_size, length, _dim = visual.shape
        flat_visual = visual.reshape(batch_size * length, -1)
        cats = categoricals_from_batch(batch, "history_cat_")
        flat_cats = {
            field: tensor.to(visual.device).reshape(batch_size * length)
            for field, tensor in cats.items()
        }
        encoded = self.model_tower(flat_visual, flat_cats).reshape(batch_size, length, self.embedding_dim)
        mask_bool = mask.to(dtype=torch.bool)
        scores = self.attention(encoded).squeeze(-1)
        all_masked = ~mask_bool.any(dim=-1)
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_bool, fill_value)
        scores = scores.masked_fill(all_masked.unsqueeze(-1), 0.0)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask_bool, 0.0)
        weights = weights.masked_fill(all_masked.unsqueeze(-1), 0.0)
        return (weights.unsqueeze(-1) * encoded).sum(dim=1)

    def forward(self, batch: dict, device: torch.device | None = None) -> torch.Tensor:
        pooled = self.encode_history(batch, device=device)
        numeric = batch["numeric"]
        age_id = batch["age_id"]
        if device is not None:
            numeric = numeric.to(device)
            age_id = age_id.to(device)
        age = self.age_embedding(age_id)
        fused = torch.cat([pooled, age, numeric], dim=-1)
        output = self.fused(fused)
        return F.normalize(output, p=2, dim=-1)
