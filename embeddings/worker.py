"""JSON CLI worker: download model packshots, hash them, and emit embeddings.

Stdout is machine-readable JSON only. Logs go to stderr.

Usage:
    python -m embeddings.worker examples/40724_001.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from embeddings.vision_encoder import VisionEncoder

__version__ = "0.1.0"

logger = logging.getLogger("embeddings.worker")

IMAGE_TYPES = ("main", "pers", "side")
DEFAULT_BATCH_SIZE = 12
DOWNLOAD_TIMEOUT = (5, 20)
MAX_DOWNLOAD_ATTEMPTS = 2
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
USER_AGENT = f"pretty-reco-ml/{__version__}"

EXIT_OK = 0
EXIT_FATAL = 1


class PayloadError(ValueError):
    """Fatal input error: malformed JSON or unsupported payload structure."""


@dataclass(frozen=True)
class ImageJob:
    model_id: Any
    model: str
    image_type: str
    url: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_image_bytes(url: str, session: requests.Session) -> bytes:
    """Download image bytes. Retries transient failures once (max 2 attempts)."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                timeout=DOWNLOAD_TIMEOUT,
                allow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("download attempt %s failed for %s: %s", attempt, url, exc)
            continue

        if response.status_code in {403, 404}:
            raise FileNotFoundError("IMAGE_NOT_FOUND")
        if response.status_code in RETRYABLE_STATUS and attempt < MAX_DOWNLOAD_ATTEMPTS:
            logger.warning(
                "download attempt %s HTTP %s for %s",
                attempt,
                response.status_code,
                url,
            )
            continue
        if not response.ok:
            raise requests.HTTPError(
                f"HTTP {response.status_code}",
                response=response,
            )
        return response.content

    raise requests.RequestException(str(last_error) if last_error else "download failed")


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("INVALID_IMAGE") from exc
    except OSError as exc:
        raise ValueError("INVALID_IMAGE") from exc


def load_payload(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PayloadError(f"cannot read input file: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PayloadError(f"malformed input JSON: {exc}") from exc
    return parse_payload_object(data)


def parse_payload_object(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise PayloadError('unsupported payload structure: expected {"models": [...]}')
    return data


def collect_jobs(payload: dict[str, Any]) -> tuple[list[ImageJob], list[dict[str, Any]]]:
    jobs: list[ImageJob] = []
    errors: list[dict[str, Any]] = []

    for item in payload["models"]:
        if not isinstance(item, dict):
            errors.append(
                {
                    "model_id": None,
                    "model": None,
                    "image_type": None,
                    "error": "INVALID_MODEL_ENTRY",
                }
            )
            continue

        model_id = item.get("model_id")
        model = item.get("model")
        images = item.get("images")
        if model is None or not isinstance(images, dict):
            errors.append(
                {
                    "model_id": model_id,
                    "model": model,
                    "image_type": None,
                    "error": "INVALID_MODEL_ENTRY",
                }
            )
            continue

        for image_type in IMAGE_TYPES:
            if image_type not in images:
                continue
            url = images[image_type]
            if not isinstance(url, str) or not url.strip():
                errors.append(
                    {
                        "model_id": model_id,
                        "model": model,
                        "image_type": image_type,
                        "error": "INVALID_URL",
                    }
                )
                continue
            jobs.append(
                ImageJob(
                    model_id=model_id,
                    model=str(model),
                    image_type=image_type,
                    url=url.strip(),
                )
            )

        unknown = set(images) - set(IMAGE_TYPES)
        if unknown:
            logger.warning(
                "ignoring unsupported image types for %s: %s",
                model,
                ", ".join(sorted(unknown)),
            )

        if not any(image_type in images for image_type in IMAGE_TYPES):
            logger.info("model %s has no main/pers/side images", model)

    return jobs, errors


def error_code_for_download(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "IMAGE_NOT_FOUND"
    if isinstance(exc, ValueError) and str(exc) == "INVALID_IMAGE":
        return "INVALID_IMAGE"
    return "DOWNLOAD_FAILED"


def process_jobs(
    jobs: list[ImageJob],
    encoder: VisionEncoder,
    batch_size: int,
    session: requests.Session,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: list[tuple[ImageJob, str, Image.Image]] = []

    def flush() -> None:
        if not pending:
            return
        batch_jobs = [item[0] for item in pending]
        hashes = [item[1] for item in pending]
        images = [item[2] for item in pending]
        try:
            vectors = encoder.encode_batch(images)
        except Exception:
            logger.exception("batch encode failed; retrying images individually")
            vectors = None

        if vectors is not None:
            if vectors.shape[0] != len(batch_jobs):
                raise RuntimeError("encoder returned unexpected batch size")
            for job, image_hash, vector in zip(batch_jobs, hashes, vectors, strict=True):
                results.append(
                    {
                        "model_id": job.model_id,
                        "model": job.model,
                        "image_type": job.image_type,
                        "embedding_model": encoder.embedding_model,
                        "embedding_dimension": int(vector.shape[0]),
                        "embedding": vector.astype(float).tolist(),
                        "image_hash": image_hash,
                    }
                )
            pending.clear()
            return

        for job, image_hash, image in zip(batch_jobs, hashes, images, strict=True):
            try:
                vector = encoder.encode(image)
            except Exception as exc:
                logger.exception("encode failed for %s %s", job.model, job.image_type)
                errors.append(
                    {
                        "model_id": job.model_id,
                        "model": job.model,
                        "image_type": job.image_type,
                        "error": "ENCODE_FAILED",
                    }
                )
                continue
            results.append(
                {
                    "model_id": job.model_id,
                    "model": job.model,
                    "image_type": job.image_type,
                    "embedding_model": encoder.embedding_model,
                    "embedding_dimension": int(vector.shape[0]),
                    "embedding": vector.astype(float).tolist(),
                    "image_hash": image_hash,
                }
            )
        pending.clear()

    for job in jobs:
        try:
            image_bytes = download_image_bytes(job.url, session)
            image_hash = sha256_hex(image_bytes)
            image = decode_image(image_bytes)
        except Exception as exc:
            logger.warning("%s %s %s: %s", job.model, job.image_type, job.url, exc)
            errors.append(
                {
                    "model_id": job.model_id,
                    "model": job.model,
                    "image_type": job.image_type,
                    "error": error_code_for_download(exc),
                }
            )
            continue

        pending.append((job, image_hash, image))
        if len(pending) >= batch_size:
            flush()

    flush()
    return results, errors


def run(payload: dict[str, Any], encoder: VisionEncoder, batch_size: int) -> dict[str, Any]:
    jobs, errors = collect_jobs(payload)
    with requests.Session() as session:
        results, download_errors = process_jobs(jobs, encoder, batch_size, session)
    errors.extend(download_errors)
    return {"results": results, "errors": errors}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretty Reco ML embeddings worker")
    parser.add_argument(
        "input_json",
        nargs="?",
        help="Path to JSON payload with a models array",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        help="Encoder batch size (default: 12, or EMBEDDING_BATCH_SIZE)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print worker and Python versions, then exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    if args.version:
        print(f"pretty-reco-ml {__version__}")
        print(f"Python {sys.version.split()[0]}")
        return EXIT_OK

    if not args.input_json:
        logger.error("input JSON path is required")
        return EXIT_FATAL

    if args.batch_size < 1:
        logger.error("batch size must be >= 1")
        return EXIT_FATAL

    try:
        payload = load_payload(Path(args.input_json))
    except PayloadError as exc:
        logger.error("%s", exc)
        return EXIT_FATAL

    try:
        from embeddings.vision_encoder import VisionEncoder

        encoder = VisionEncoder()
    except Exception as exc:
        logger.exception("encoder cannot initialize: %s", exc)
        return EXIT_FATAL

    output = run(payload, encoder, args.batch_size)
    sys.stdout.write(json.dumps(output, ensure_ascii=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
