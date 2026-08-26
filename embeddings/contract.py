"""HTTP request/response models for POST /embeddings/models.

Node supplies model identity and explicit image URLs. Python returns embeddings
and per-image errors. This module does not access any database.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelImages(BaseModel):
    """Optional packshot URLs. Missing views are skipped, not fatal."""

    model_config = ConfigDict(extra="ignore")

    main: str | None = None
    pers: str | None = None
    side: str | None = None


class ModelEmbeddingInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_id: int | str
    model: str
    images: ModelImages


class EmbeddingsRequest(BaseModel):
    models: list[ModelEmbeddingInput]


class EmbeddingResult(BaseModel):
    model_id: Any
    model: str
    image_type: str
    embedding_model: str
    embedding_dimension: int
    embedding: list[float]
    image_hash: str = Field(min_length=64, max_length=64)


class EmbeddingError(BaseModel):
    model_id: Any = None
    model: str | None = None
    image_type: str | None = None
    error: str


class EmbeddingsResponse(BaseModel):
    results: list[EmbeddingResult]
    errors: list[EmbeddingError]
