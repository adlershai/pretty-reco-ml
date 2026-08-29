"""Model tower: visual vector + categorical embeddings → normalized M."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import MODEL_CATEGORICAL_FIELDS, TrainConfig, categorical_embedding_dim
from training.feature_encoders import FeatureEncoders


class ModelTower(nn.Module):
    def __init__(self, config: TrainConfig, encoders: FeatureEncoders) -> None:
        super().__init__()
        self.fields = tuple(config.categorical_fields)
        self.embedding_dim = int(config.embedding_dim)
        self.embeddings = nn.ModuleDict()
        cat_dims: list[int] = []
        for field in self.fields:
            vocab_size = encoders.categoricals[field].size
            dim = categorical_embedding_dim(vocab_size)
            self.embeddings[field] = nn.Embedding(vocab_size, dim, padding_idx=0)
            cat_dims.append(dim)
        self.categorical_dim = int(sum(cat_dims))
        self.visual_proj = nn.Linear(config.visual_input_dim, config.hidden_dim)
        fused = config.hidden_dim + self.categorical_dim
        self.fused = nn.Sequential(
            nn.Linear(fused, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(
        self,
        visual: torch.Tensor,
        categoricals: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        visual_h = F.relu(self.visual_proj(visual))
        pieces = [self.embeddings[field](categoricals[field]) for field in self.fields]
        cats = torch.cat(pieces, dim=-1)
        fused = torch.cat([visual_h, cats], dim=-1)
        output = self.fused(fused)
        return F.normalize(output, p=2, dim=-1)


def categoricals_from_batch(batch: dict, prefix: str) -> dict[str, torch.Tensor]:
    return {field: batch[f"{prefix}{field}"] for field in MODEL_CATEGORICAL_FIELDS}
