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

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from embeddings.contract import EmbeddingsRequest, EmbeddingsResponse
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import DEFAULT_BATCH_SIZE, run

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
    yield


app = FastAPI(title="pretty-reco-ml", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_payload(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "invalid request/payload"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
