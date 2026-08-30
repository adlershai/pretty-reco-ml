"""HTTP entry point for pretty-reco-ml.

Binds to 127.0.0.1:8000 in production. Nginx proxies https://ai.adler-backend.com.
Reuses embeddings.worker — no vision or database logic lives here.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from embeddings.contract import EmbeddingsRequest, EmbeddingsResponse
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import DEFAULT_BATCH_SIZE, run
from inference.recommender import (
    MODEL_VERSION,
    ModelNotEncodableError,
    ModelNotFoundError,
    RecommenderService,
    normalize_limit,
)

logger = logging.getLogger("pretty-reco-ml.app")


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs without overwriting existing environment values."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent / ".env")


def _expected_api_key() -> str:
    key = os.environ.get("RECO_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="RECO_API_KEY is not configured")
    return key


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = _expected_api_key()
    provided = x_api_key or ""
    if len(provided) != len(expected) or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("loading vision encoder")
    app.state.encoder = VisionEncoder()
    logger.info("loading recommender")
    app.state.recommender = RecommenderService.load()
    yield


app = FastAPI(title="pretty-reco-ml", lifespan=lifespan)


def _recommender(request: Request) -> RecommenderService:
    recommender: RecommenderService | None = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=503, detail="recommender is not loaded")
    return recommender


@app.exception_handler(RequestValidationError)
async def invalid_payload(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "invalid request/payload"})


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    recommender: RecommenderService | None = getattr(request.app.state, "recommender", None)
    payload: dict[str, Any] = {"status": "ok"}
    if recommender is not None:
        payload["model"] = MODEL_VERSION
        payload["dimension"] = recommender.dimension
    return payload


@app.exception_handler(ModelNotFoundError)
async def model_not_found(_request: Request, exc: ModelNotFoundError) -> JSONResponse:
    model = str(exc.args[0]) if exc.args else ""
    return JSONResponse(status_code=404, content={"error": "model_not_found", "model": model})


@app.exception_handler(ModelNotEncodableError)
async def model_not_encodable(_request: Request, exc: ModelNotEncodableError) -> JSONResponse:
    model = str(exc.args[0]) if exc.args else ""
    return JSONResponse(status_code=422, content={"error": "model_not_encodable", "model": model})


@app.get("/recommend/customers/{model}")
def recommend_customers(
    model: str,
    limit: int | None = Query(default=None),
    recommender: RecommenderService = Depends(_recommender),
) -> list[dict[str, Any]]:
    try:
        effective_limit = normalize_limit(limit)
    except ValueError:
        raise HTTPException(status_code=400, detail="limit must be >= 1") from None
    return recommender.recommend(model, limit=effective_limit)


@app.post("/embeddings/models", response_model=EmbeddingsResponse)
def embeddings_models(
    payload: EmbeddingsRequest,
    request: Request,
    _: None = Depends(require_api_key),
) -> EmbeddingsResponse:
    batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    if batch_size < 1:
        raise HTTPException(status_code=500, detail="EMBEDDING_BATCH_SIZE must be >= 1")

    encoder: VisionEncoder = request.app.state.encoder
    try:
        raw = run(payload.model_dump(exclude_none=True), encoder, batch_size)
        return EmbeddingsResponse.model_validate(raw)
    except HTTPException:
        raise
    except Exception:
        logger.exception("encoder/service-level failure")
        raise HTTPException(status_code=500, detail="encoder/service-level failure") from None
