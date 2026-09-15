"""Visual product identification: per-image SigLIP cosine, then MAX per model.

This is not the two-tower recommender. Catalog views stay independent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from data.dataset_builder import parse_embedding
from embeddings.vision_encoder import EMBEDDING_DIMENSION, SIGLIP_MODEL_ID

IMAGE_TYPES = ("main", "pers", "side")
IMAGE_BASE = "https://media.adler.co.il/app/products"
NORM_ATOL = 1e-3


@dataclass(frozen=True)
class CatalogImage:
    model: str
    image_type: str
    embedding: np.ndarray
    model_id: int | None = None
    row_id: int | None = None
    embedding_model: str = ""
    embedding_dimension: int = 0
    image_hash: str | None = None
    source_url: str | None = None

    @property
    def key(self) -> str:
        return f"{self.model} / {self.image_type}"

    @property
    def vector_norm(self) -> float:
        return float(np.linalg.norm(self.embedding))


@dataclass(frozen=True)
class ImageHit:
    rank: int
    model: str
    image_type: str
    similarity: float
    model_id: int | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class ModelHit:
    rank: int
    model: str
    model_score: float
    best_image_type: str
    best_image_similarity: float
    model_id: int | None = None


@dataclass
class IntegrityIssue:
    model: str
    image_type: str
    code: str
    detail: str


@dataclass
class IntegrityReport:
    embedding_model: str
    dimension: int
    expected_model: str = SIGLIP_MODEL_ID
    expected_dimension: int = EMBEDDING_DIMENSION
    issues: list[IntegrityIssue] = field(default_factory=list)
    image_count: int = 0
    model_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def catalog_image_url(model: str, image_type: str) -> str:
    code = str(model or "").strip()
    view = str(image_type or "").strip().lower()
    if view == "main":
        return f"{IMAGE_BASE}/{code}.jpg"
    return f"{IMAGE_BASE}/{code}_{view}.jpg"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot L2-normalize an empty or non-finite vector")
    return (array / norm).astype(np.float32, copy=False)


def stack_embeddings(rows: Sequence[CatalogImage]) -> np.ndarray:
    if not rows:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    return np.stack([row.embedding for row in rows], axis=0)


def image_similarities(query: np.ndarray, catalog: np.ndarray) -> np.ndarray:
    """Cosine similarity when both sides are L2-normalized: query @ catalog.T."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(catalog, dtype=np.float32)
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("catalog embeddings must be a 2-D matrix")
    if int(matrix.shape[1]) != int(q.shape[0]):
        raise ValueError(
            f"query dim {q.shape[0]} does not match catalog dim {matrix.shape[1]}"
        )
    return matrix @ q


def rank_catalog_images(
    query: np.ndarray,
    rows: Sequence[CatalogImage],
    *,
    top: int | None = None,
) -> list[ImageHit]:
    scores = image_similarities(query, stack_embeddings(rows))
    order = np.argsort(-scores, kind="stable")
    if top is not None:
        order = order[: max(0, int(top))]
    hits: list[ImageHit] = []
    for rank, index in enumerate(order, start=1):
        row = rows[int(index)]
        hits.append(
            ImageHit(
                rank=rank,
                model=row.model,
                image_type=row.image_type,
                similarity=float(scores[int(index)]),
                model_id=row.model_id,
                source_url=row.source_url,
            )
        )
    return hits


def aggregate_models_max(image_hits: Sequence[ImageHit]) -> list[ModelHit]:
    """One score per model: MAX(image similarity). Do not average views."""
    best: dict[str, ImageHit] = {}
    for hit in image_hits:
        current = best.get(hit.model)
        if current is None or hit.similarity > current.similarity:
            best[hit.model] = hit
    ranked = sorted(
        best.values(),
        key=lambda hit: (-hit.similarity, hit.model, hit.image_type),
    )
    return [
        ModelHit(
            rank=rank,
            model=hit.model,
            model_score=hit.similarity,
            best_image_type=hit.image_type,
            best_image_similarity=hit.similarity,
            model_id=hit.model_id,
        )
        for rank, hit in enumerate(ranked, start=1)
    ]


def classify_failure(
    *,
    expected: str | None,
    compare: str | None,
    expected_side: ImageHit | None,
    compare_hits: Sequence[ImageHit],
    production: ModelHit | None,
    expected_self_test_ok: bool | None = None,
) -> str:
    """A = search bug, B = embedding consistency, C = retrieval quality, ok = match."""
    if expected_self_test_ok is False:
        return "B"
    compare_best = max((hit.similarity for hit in compare_hits), default=float("-inf"))
    expected_best = expected_side.similarity if expected_side is not None else float("-inf")
    if (
        expected
        and compare
        and expected_side is not None
        and expected_best > compare_best
        and production is not None
        and production.model == compare
    ):
        return "A"
    if expected and production is not None and production.model == expected:
        return "ok"
    if expected and production is not None and production.model != expected:
        return "C"
    return "C"


def find_image_hit(hits: Sequence[ImageHit], model: str, image_type: str) -> ImageHit | None:
    needle_model = str(model or "").strip()
    needle_type = str(image_type or "").strip().lower()
    for hit in hits:
        if hit.model == needle_model and hit.image_type == needle_type:
            return hit
    return None


def hits_for_model(hits: Sequence[ImageHit], model: str) -> list[ImageHit]:
    needle = str(model or "").strip()
    return [hit for hit in hits if hit.model == needle]


def inspect_vector(
    vector: np.ndarray,
    *,
    embedding_model: str,
    expected_model: str = SIGLIP_MODEL_ID,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> list[str]:
    flags: list[str] = []
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if embedding_model and embedding_model != expected_model:
        flags.append(f"embedding_model={embedding_model} (expected {expected_model})")
    if int(array.size) != int(expected_dimension):
        flags.append(f"dimension={array.size} (expected {expected_dimension})")
    if array.size == 0:
        flags.append("empty vector")
        return flags
    if not np.isfinite(array).all():
        flags.append("non-finite values")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or abs(norm - 1.0) > NORM_ATOL:
        flags.append(f"norm={norm:.6f} (expected ~ 1.0)")
    return flags


def inspect_catalog_integrity(
    rows: Sequence[CatalogImage],
    *,
    expected_model: str = SIGLIP_MODEL_ID,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> IntegrityReport:
    issues: list[IntegrityIssue] = []
    seen_keys: dict[str, int] = {}
    vector_groups: dict[bytes, list[str]] = {}
    models: set[str] = set()

    for row in rows:
        models.add(row.model)
        key = row.key
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if seen_keys[key] == 2:
            issues.append(
                IntegrityIssue(row.model, row.image_type, "duplicate_key", f"{key} appears more than once")
            )
        if row.embedding_model and row.embedding_model != expected_model:
            issues.append(
                IntegrityIssue(
                    row.model,
                    row.image_type,
                    "embedding_model",
                    f"{row.embedding_model} (expected {expected_model})",
                )
            )
        dim = int(row.embedding.size)
        stored_dim = int(row.embedding_dimension or dim)
        if dim != expected_dimension or stored_dim != expected_dimension:
            issues.append(
                IntegrityIssue(
                    row.model,
                    row.image_type,
                    "dimension",
                    f"vector={dim} stored={stored_dim} (expected {expected_dimension})",
                )
            )
        if dim == 0:
            issues.append(IntegrityIssue(row.model, row.image_type, "empty", "missing vector"))
            continue
        if not np.isfinite(row.embedding).all():
            issues.append(IntegrityIssue(row.model, row.image_type, "non_finite", "NaN/Inf in vector"))
        norm = row.vector_norm
        if not np.isfinite(norm) or abs(norm - 1.0) > NORM_ATOL:
            issues.append(
                IntegrityIssue(row.model, row.image_type, "norm", f"{norm:.6f} (expected ~ 1.0)")
            )
        vector_groups.setdefault(row.embedding.tobytes(), []).append(key)

    for keys in vector_groups.values():
        unique_models = {item.split(" / ", 1)[0] for item in keys}
        if len(keys) > 1 and len(unique_models) > 1:
            issues.append(
                IntegrityIssue(
                    keys[0].split(" / ", 1)[0],
                    keys[0].split(" / ", 1)[1],
                    "duplicate_vector",
                    "identical vector shared by " + ", ".join(keys),
                )
            )

    model_names = {row.embedding_model for row in rows if row.embedding_model}
    report_model = next(iter(model_names), "") if len(model_names) == 1 else ",".join(sorted(model_names))
    dims = {int(row.embedding.size) for row in rows}
    report_dim = next(iter(dims)) if len(dims) == 1 else 0
    return IntegrityReport(
        embedding_model=report_model,
        dimension=report_dim,
        expected_model=expected_model,
        expected_dimension=expected_dimension,
        issues=issues,
        image_count=len(rows),
        model_count=len(models),
    )


def catalog_from_records(records: Iterable[dict[str, Any]]) -> list[CatalogImage]:
    rows: list[CatalogImage] = []
    for record in records:
        model = str(record.get("model") or "").strip()
        image_type = str(record.get("image_type") or "").strip().lower()
        embedding = parse_embedding(record.get("embedding"))
        if not model or not image_type or embedding is None:
            continue
        embedding_model = str(record.get("embedding_model") or "")
        stored_dim = record.get("embedding_dimension")
        try:
            dimension = int(stored_dim) if stored_dim is not None else int(embedding.size)
        except (TypeError, ValueError):
            dimension = int(embedding.size)
        model_id_raw = record.get("model_id")
        try:
            model_id = int(model_id_raw) if model_id_raw is not None else None
        except (TypeError, ValueError):
            model_id = None
        row_id_raw = record.get("id") or record.get("row_id")
        try:
            row_id = int(row_id_raw) if row_id_raw is not None else None
        except (TypeError, ValueError):
            row_id = None
        image_hash = record.get("image_hash")
        hash_text = str(image_hash).strip() if image_hash else None
        source = record.get("source_url") or record.get("image_url")
        source_url = str(source).strip() if source else catalog_image_url(model, image_type)
        rows.append(
            CatalogImage(
                model=model,
                image_type=image_type,
                embedding=embedding.astype(np.float32, copy=False),
                model_id=model_id,
                row_id=row_id,
                embedding_model=embedding_model,
                embedding_dimension=dimension,
                image_hash=hash_text or None,
                source_url=source_url,
            )
        )
    return rows


def catalog_from_model_view(records: Iterable[dict[str, Any]]) -> list[CatalogImage]:
    """Expand vw_reco_model_representation_v1 into per-image rows."""
    rows: list[CatalogImage] = []
    for record in records:
        model = str(record.get("model") or "").strip()
        if not model:
            continue
        embedding_model = str(record.get("embedding_model") or "")
        stored_dim = record.get("embedding_dimension")
        model_id_raw = record.get("model_id")
        try:
            model_id = int(model_id_raw) if model_id_raw is not None else None
        except (TypeError, ValueError):
            model_id = None
        for image_type, column in (
            ("main", "main_embedding"),
            ("pers", "pers_embedding"),
            ("side", "side_embedding"),
        ):
            embedding = parse_embedding(record.get(column))
            if embedding is None:
                continue
            try:
                dimension = int(stored_dim) if stored_dim is not None else int(embedding.size)
            except (TypeError, ValueError):
                dimension = int(embedding.size)
            rows.append(
                CatalogImage(
                    model=model,
                    image_type=image_type,
                    embedding=embedding.astype(np.float32, copy=False),
                    model_id=model_id,
                    embedding_model=embedding_model,
                    embedding_dimension=dimension,
                    source_url=catalog_image_url(model, image_type),
                )
            )
    return rows


def format_image_table(hits: Sequence[ImageHit]) -> str:
    lines = ["rank | model | image_type | similarity"]
    for hit in hits:
        lines.append(f"{hit.rank} | {hit.model} | {hit.image_type} | {hit.similarity:.6f}")
    return "\n".join(lines)


def format_model_table(hits: Sequence[ModelHit]) -> str:
    lines = ["rank | model | model_score | best_image_type | best_image_similarity"]
    for hit in hits:
        lines.append(
            f"{hit.rank} | {hit.model} | {hit.model_score:.6f} | "
            f"{hit.best_image_type} | {hit.best_image_similarity:.6f}"
        )
    return "\n".join(lines)


def dump_hits_json(image_hits: Sequence[ImageHit], model_hits: Sequence[ModelHit]) -> str:
    payload = {
        "images": [
            {
                "rank": hit.rank,
                "model": hit.model,
                "image_type": hit.image_type,
                "similarity": hit.similarity,
            }
            for hit in image_hits
        ],
        "models": [
            {
                "rank": hit.rank,
                "model": hit.model,
                "model_score": hit.model_score,
                "best_image_type": hit.best_image_type,
                "best_image_similarity": hit.best_image_similarity,
            }
            for hit in model_hits
        ],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)
