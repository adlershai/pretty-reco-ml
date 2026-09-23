"""Generic multilingual text embeddings and cosine similarity.

This module is intentionally domain-agnostic. Callers own persistence and business
semantics; pretty-reco-ml owns vector generation and ranking.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_TEXT_MODEL = os.environ.get(
    "TEXT_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)


class TextEncoder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_TEXT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        self.embedding_dimension = int(hidden_size or 0)

    @staticmethod
    def _average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask[..., None].bool()
        masked = last_hidden_state.masked_fill(~mask, 0.0)
        denom = attention_mask.sum(dim=1)[..., None].clamp(min=1)
        return masked.sum(dim=1) / denom

    def encode(self, texts: list[str], *, prefix: str = "passage") -> np.ndarray:
        clean = [str(text or "").strip() for text in texts]
        if not clean or any(not text for text in clean):
            raise ValueError("TEXT_REQUIRED")
        if prefix not in {"passage", "query"}:
            raise ValueError("INVALID_TEXT_PREFIX")

        # E5: stored documents use passage:, search strings use query:.
        prepared = [f"{prefix}: {text}" for text in clean]
        batch = self.tokenizer(
            prepared,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)
            pooled = self._average_pool(outputs.last_hidden_state, batch["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return normalized.detach().cpu().numpy().astype(np.float32)


def cosine_similarity(query: list[float], candidate: list[float]) -> float:
    q = np.asarray(query, dtype=np.float32)
    c = np.asarray(candidate, dtype=np.float32)
    if q.ndim != 1 or c.ndim != 1 or q.shape[0] == 0 or q.shape != c.shape:
        raise ValueError("VECTOR_DIMENSION_MISMATCH")
    q_norm = float(np.linalg.norm(q))
    c_norm = float(np.linalg.norm(c))
    if q_norm == 0.0 or c_norm == 0.0:
        raise ValueError("ZERO_VECTOR")
    return float(np.dot(q, c) / (q_norm * c_norm))


def rank_candidates(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
    top: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for candidate in candidates:
        candidate_id = str(candidate.get("id", "")).strip()
        if not candidate_id:
            raise ValueError("CANDIDATE_ID_REQUIRED")
        if candidate_id in seen_ids:
            raise ValueError("DUPLICATE_CANDIDATE_ID")
        seen_ids.add(candidate_id)

        vector = candidate.get("embedding")
        if not isinstance(vector, list):
            raise ValueError("CANDIDATE_EMBEDDING_REQUIRED")

        ranked.append(
            {
                "id": candidate_id,
                "score": cosine_similarity(query_embedding, vector),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    selected = ranked[:top]
    return [
        {
            "id": item["id"],
            "rank": index,
            "score": item["score"],
        }
        for index, item in enumerate(selected, start=1)
    ]
