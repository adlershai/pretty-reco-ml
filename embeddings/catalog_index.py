"""In-memory catalog of per-view SigLIP embeddings for visual product ID."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.config import LOCAL_ROOT, load_dotenv
from data.db_client import DbApiClient
from data.schemas import MODEL_VIEW
from embeddings.vision_encoder import EMBEDDING_DIMENSION, SIGLIP_MODEL_ID
from evaluation.image_match import (
    CatalogImage,
    ImageHit,
    ModelHit,
    aggregate_models_max,
    catalog_from_model_view,
    catalog_from_records,
    rank_catalog_images,
    stack_embeddings,
)

logger = logging.getLogger("embeddings.catalog_index")

LOAD_CHUNK = 80
CACHE_DIR = LOCAL_ROOT / "cache"
CACHE_META = "catalog_image_index.json"
CACHE_VECTORS = "catalog_image_index.npz"


@dataclass
class CatalogIndex:
    rows: list[CatalogImage]
    matrix: np.ndarray
    embedding_model: str

    @classmethod
    def from_rows(
        cls,
        rows: list[CatalogImage],
        embedding_model: str = SIGLIP_MODEL_ID,
    ) -> CatalogIndex:
        if rows:
            matrix = stack_embeddings(rows)
        else:
            matrix = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        return cls(rows=list(rows), matrix=matrix, embedding_model=embedding_model)

    def search_images(self, query: np.ndarray) -> list[ImageHit]:
        return rank_catalog_images(query, self.rows)

    def search_models(self, query: np.ndarray, top: int = 10) -> list[ModelHit]:
        return aggregate_models_max(self.search_images(query))[: max(0, int(top))]


def _cache_paths() -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / CACHE_META, CACHE_DIR / CACHE_VECTORS


def save_catalog_cache(rows: list[CatalogImage], source: str, embedding_model: str) -> None:
    meta_path, vector_path = _cache_paths()
    payload = {
        "source": source,
        "embedding_model": embedding_model,
        "rows": [
            {
                "id": row.row_id,
                "model_id": row.model_id,
                "model": row.model,
                "image_type": row.image_type,
                "embedding_model": row.embedding_model,
                "embedding_dimension": row.embedding_dimension,
                "image_hash": row.image_hash,
                "source_url": row.source_url,
            }
            for row in rows
        ],
    }
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(vector_path, embeddings=stack_embeddings(rows))


def load_catalog_cache(source: str, embedding_model: str) -> list[CatalogImage] | None:
    meta_path, vector_path = _cache_paths()
    if not meta_path.is_file() or not vector_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        matrix = np.load(vector_path)["embeddings"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    if payload.get("source") != source or payload.get("embedding_model") != embedding_model:
        return None
    records = payload.get("rows")
    if not isinstance(records, list) or len(records) != int(matrix.shape[0]):
        return None
    rows: list[CatalogImage] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return None
        model_id = record.get("model_id")
        try:
            model_id_int = int(model_id) if model_id is not None else None
        except (TypeError, ValueError):
            model_id_int = None
        row_id = record.get("id")
        try:
            row_id_int = int(row_id) if row_id is not None else None
        except (TypeError, ValueError):
            row_id_int = None
        rows.append(
            CatalogImage(
                model=str(record.get("model") or ""),
                image_type=str(record.get("image_type") or ""),
                embedding=np.asarray(matrix[index], dtype=np.float32),
                model_id=model_id_int,
                row_id=row_id_int,
                embedding_model=str(record.get("embedding_model") or embedding_model),
                embedding_dimension=int(record.get("embedding_dimension") or matrix.shape[1]),
                image_hash=record.get("image_hash"),
                source_url=record.get("source_url"),
            )
        )
    return rows


def load_catalog_from_table(client: DbApiClient, embedding_model: str) -> list[CatalogImage]:
    records: list[dict[str, Any]] = []
    after_id = 0
    while True:
        batch = client.all(
            f"""SELECT e.id, e.model_id, e.image_type, e.embedding, e.embedding_model,
                       e.embedding_dimension, e.image_hash, m.model AS model
                FROM reco_model_image_embeddings e
                INNER JOIN models m ON m.id = e.model_id
                WHERE e.embedding_model = ?
                  AND e.id > ?
                ORDER BY e.id ASC
                LIMIT {LOAD_CHUNK}""",
            [embedding_model, after_id],
        )
        if not batch:
            break
        records.extend(batch)
        max_id = after_id
        for row in batch:
            try:
                row_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            if row_id > max_id:
                max_id = row_id
        logger.info("catalog table rows loaded: %s", len(records))
        if len(batch) < LOAD_CHUNK or max_id <= after_id:
            break
        after_id = max_id
    return catalog_from_records(records)


def load_catalog_from_view(client: DbApiClient) -> list[CatalogImage]:
    logger.info("fetching %s", MODEL_VIEW)
    return catalog_from_model_view(client.get_view(MODEL_VIEW))


def load_catalog_rows(
    *,
    client: DbApiClient | None = None,
    source: str = "table",
    embedding_model: str = SIGLIP_MODEL_ID,
    refresh: bool = False,
) -> list[CatalogImage]:
    load_dotenv()
    if not refresh:
        cached = load_catalog_cache(source, embedding_model)
        if cached:
            logger.info("catalog cache hit: %s images", len(cached))
            return cached
    db = client or DbApiClient()
    if source == "view":
        rows = load_catalog_from_view(db)
    else:
        rows = load_catalog_from_table(db, embedding_model)
    if not rows:
        raise RuntimeError("catalog is empty")
    save_catalog_cache(rows, source, embedding_model)
    logger.info("catalog loaded: %s images", len(rows))
    return rows


def load_catalog_index(**kwargs: Any) -> CatalogIndex:
    embedding_model = str(kwargs.get("embedding_model") or SIGLIP_MODEL_ID)
    rows = load_catalog_rows(**kwargs)
    return CatalogIndex.from_rows(rows, embedding_model=embedding_model)
