"""HTTP request/response models for POST /embeddings/models.

Node supplies model identity and explicit image URLs. Python returns embeddings
and per-image errors. This module does not access any database.
"""

from __future__ import annotations

from typing import Any, Literal

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


class QueryEmbeddingRequest(BaseModel):
    image_base64: str = Field(min_length=8)


class QueryEmbeddingResponse(BaseModel):
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int
    relevance: Literal["footwear", "irrelevant"]
    scores: dict[str, float]


class ImageMatchRequest(BaseModel):
    image_base64: str = Field(min_length=8)
    top: int | None = Field(default=None, ge=1, le=50)
    verify_top: int | None = Field(default=None, ge=1, le=50)


class ImageMatchCandidate(BaseModel):
    model: str
    score: float | None = None
    best_image_type: str | None = None
    model_id: int | None = None


class OrderExtract(BaseModel):
    order_number: str | None = None
    model: str | None = None
    size: str | None = None


class DeliveryExtract(BaseModel):
    tracking_number: str | None = None
    status: str | None = None


class ImageMatchPreprocessing(BaseModel):
    shoe_isolated: bool
    crop: list[int]
    reason: str
    image_size: list[int]
    crop_jpeg_base64: str | None = None


class ImageMatchResponse(BaseModel):
    status: Literal[
        "needs_verification",
        "irrelevant",
        "match",
        "uncertain",
        "order",
        "delivery_notice",
        "garbage",
    ]
    match: ImageMatchCandidate | None = None
    candidates: list[ImageMatchCandidate]
    preprocessing: ImageMatchPreprocessing
    verifier: dict[str, Any] | None = None
    relevance: Literal["footwear", "irrelevant"]
    scores: dict[str, float]
    embedding_model: str
    image_kind: Literal["shoe", "order", "delivery_notice", "garbage"] = "shoe"
    order: OrderExtract | None = None
    delivery: DeliveryExtract | None = None


class ImageMatchDecideRequest(BaseModel):
    candidates: list[ImageMatchCandidate]
    openai_output: Any
    verify_top: int | None = Field(default=None, ge=1, le=50)


class ImageMatchDecideResponse(BaseModel):
    status: Literal["match", "uncertain"]
    match: ImageMatchCandidate | None = None
    confidence: float | None = None
    reason: str
    verification: list[dict[str, Any]]
    candidates: list[ImageMatchCandidate]


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class TextEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class TextEmbeddingsResponse(BaseModel):
    embedding_model: str
    embedding_dimension: int
    results: list[TextEmbeddingItem]


class TextSimilarityCandidate(BaseModel):
    id: str
    embedding: list[float] = Field(min_length=1)


class TextSimilarityRequest(BaseModel):
    query_text: str = Field(min_length=1)
    candidates: list[TextSimilarityCandidate] = Field(min_length=1, max_length=5000)
    top: int = Field(default=5, ge=1, le=100)


class TextSimilarityResult(BaseModel):
    id: str
    score: float


class TextSimilarityResponse(BaseModel):
    embedding_model: str
    embedding_dimension: int
    results: list[TextSimilarityResult]
