"""Load production vectors and rank customers for a catalog model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.config import DEFAULT_SNAPSHOT_ROOT, REPO_ROOT
from data.dataset_builder import find_latest_snapshot, load_snapshot_frames
from data.db_client import DbApiClient
from training.artifact import load_artifact
from training.dataset import CatalogModel, load_catalog
from training.trainer import select_device
from training.two_tower import TwoTowerModel

from inference.customers import build_current_customers, encode_customer_vectors
from inference.like_score import LikeCalibrator, load_calibrator
from inference.models import catalog_from_rows, encode_catalog_models, fetch_model_representation
from inference.ranking import score_matrix
from inference.recency import rank_customers_with_recency
from inference.sanity import check_vectors

logger = logging.getLogger("pretty-reco-ml.recommender")

DEFAULT_ARTIFACT = REPO_ROOT / "artifacts" / "the_pretty_model_v1"
MODEL_VERSION = "the_pretty_model_v1"
DEFAULT_LIMIT = 100
MAX_LIMIT = 200


class ModelNotFoundError(LookupError):
    """Requested model code is absent from the live catalog view."""


class ModelNotEncodableError(ValueError):
    """Model row exists but cannot be encoded (missing visual vector)."""


@dataclass
class RecommenderService:
    tower: TwoTowerModel
    catalog: dict[str, CatalogModel]
    customer_ids: list[str]
    customer_vectors: np.ndarray
    last_purchase_dates: list[str | None]
    db_client: DbApiClient
    device: torch.device
    like_calibrator: LikeCalibrator
    model_version: str = MODEL_VERSION

    @property
    def dimension(self) -> int:
        return int(self.tower.config.embedding_dim)

    @classmethod
    def load(
        cls,
        *,
        artifact_dir: Path | None = None,
        snapshot_dir: Path | None = None,
        snapshot_root: Path | None = None,
        device: torch.device | None = None,
        db_client: DbApiClient | None = None,
    ) -> RecommenderService:
        artifact_path = (artifact_dir or DEFAULT_ARTIFACT).resolve()
        metadata_path = artifact_path / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("model_version") or "") != MODEL_VERSION:
                logger.warning(
                    "artifact metadata model_version=%s, expected %s",
                    metadata.get("model_version"),
                    MODEL_VERSION,
                )

        resolved_snapshot = snapshot_dir or find_latest_snapshot(snapshot_root or DEFAULT_SNAPSHOT_ROOT)
        resolved_device = device or select_device()
        tower = load_artifact(artifact_path, device=resolved_device)
        dim = int(tower.config.embedding_dim)

        purchases, customers, _models = load_snapshot_frames(resolved_snapshot.resolve())
        catalog = load_catalog(resolved_snapshot.resolve())
        records, stats = build_current_customers(
            purchases,
            customers,
            catalog,
            new_model_codes=set(),
        )
        logger.info(
            "recommender snapshot=%s customers=%s catalog=%s",
            resolved_snapshot.name,
            stats["encoded_customers"],
            len(catalog),
        )
        customer_vectors = encode_customer_vectors(tower, records, catalog, device=resolved_device)
        check_vectors(customer_vectors, dim=dim, name="customer_vectors")
        customer_ids = [str(record["customer_id"]) for record in records]
        last_purchase_dates = [record.get("last_purchase_date") for record in records]
        like_calibrator = load_calibrator(artifact_path)
        return cls(
            tower=tower,
            catalog=catalog,
            customer_ids=customer_ids,
            customer_vectors=customer_vectors,
            last_purchase_dates=last_purchase_dates,
            db_client=db_client or DbApiClient(),
            device=resolved_device,
            like_calibrator=like_calibrator,
        )

    def _resolve_model_catalog(self, model_code: str) -> CatalogModel:
        code = str(model_code or "").strip()
        if not code:
            raise ModelNotFoundError(code)
        rows = fetch_model_representation(self.db_client, [code])
        if not rows:
            raise ModelNotFoundError(code)
        live_catalog = catalog_from_rows(rows)
        if code not in live_catalog:
            raise ModelNotEncodableError(code)
        return live_catalog[code]

    def score_model(self, model_code: str) -> np.ndarray:
        code = str(model_code or "").strip()
        item = self._resolve_model_catalog(code)
        working_catalog = dict(self.catalog)
        working_catalog[code] = item
        model_vectors, encoded_codes = encode_catalog_models(
            self.tower,
            working_catalog,
            [code],
            device=self.device,
        )
        if not encoded_codes:
            raise ModelNotEncodableError(code)
        check_vectors(model_vectors, dim=self.dimension, name="model_vectors")
        return score_matrix(self.customer_vectors, model_vectors)[:, 0]

    def recommend(self, model_code: str, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
        scores = self.score_model(model_code)
        like_scores = self.like_calibrator.transform(scores)
        ranked, diagnostics = rank_customers_with_recency(
            scores,
            self.customer_ids,
            self.last_purchase_dates,
            like_scores=like_scores,
            top_k=limit,
        )
        logger.info(
            "recommend model=%s scored=%s excluded_lt_60d=%s eligible=%s returned=%s dist=%s",
            str(model_code).strip(),
            diagnostics["total_scored"],
            diagnostics["excluded_lt_60d"],
            diagnostics["eligible"],
            diagnostics["returned"],
            diagnostics["returned_distribution"],
        )
        return [
            {
                "customer_id": _customer_id_value(row["customer_id"]),
                "like_score": round(float(row["like_score"]), 4),
                "rank": int(row["rank"]),
            }
            for row in ranked
        ]


def normalize_limit(limit: int | None) -> int:
    """Default 100, cap at 200. Values below 1 raise ValueError."""
    if limit is None:
        return DEFAULT_LIMIT
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return min(int(limit), MAX_LIMIT)


def _customer_id_value(raw: str) -> int:
    text = str(raw).strip()
    if not text.isdigit():
        raise ValueError(f"customer_id is not numeric: {raw!r}")
    return int(text)
