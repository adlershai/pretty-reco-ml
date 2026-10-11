"""Fetch new catalog rows through the DB API and encode with the Model Tower."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from data.dataset_builder import combine_visual_embeddings, parse_embedding
from data.db_client import DbApiClient
from data.schemas import MODEL_FEATURE_COLUMNS, VISUAL_EMBEDDING_COLUMNS
from training.config import TrainConfig, UNKNOWN_TOKEN
from training.dataset import CatalogModel, collate_batch
from training.feature_encoders import FeatureEncoders, categorical_string
from training.two_tower import TwoTowerModel

REQUIRED_MODEL_FIELDS = MODEL_FEATURE_COLUMNS
MODEL_REPRESENTATION_PROC = "get_reco_model_representation"
PROC_CODE_CHUNK = 200


def fetch_model_representation(client: DbApiClient, codes: list[str]) -> list[dict[str, Any]]:
    wanted = [str(code).strip() for code in codes if str(code or "").strip()]
    if not wanted:
        return []
    rows: list[dict[str, Any]] = []
    for index in range(0, len(wanted), PROC_CODE_CHUNK):
        chunk = wanted[index : index + PROC_CODE_CHUNK]
        rows.extend(client.proc(MODEL_REPRESENTATION_PROC, {"models": chunk}))
    return rows


def catalog_from_rows(rows: list[dict[str, Any]]) -> dict[str, CatalogModel]:
    catalog: dict[str, CatalogModel] = {}
    keep = ("model", "model_id", "model_name", *MODEL_FEATURE_COLUMNS)
    for row in rows:
        code = str(row.get("model") or "").strip()
        if not code:
            continue
        visual = combine_visual_embeddings(
            *(parse_embedding(row.get(column)) for column in VISUAL_EMBEDDING_COLUMNS)
        )
        if visual is None:
            continue
        features = {key: row.get(key) for key in keep}
        catalog[code] = CatalogModel(code=code, features=features, visual=visual.astype(np.float32))
    return catalog


def missing_attributes(features: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_MODEL_FIELDS:
        token = categorical_string(features.get(field))
        if token == UNKNOWN_TOKEN:
            missing.append(field)
    return missing


def model_status_rows(codes: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_code = {str(row.get("model") or "").strip(): row for row in rows}
    reports = []
    for code in codes:
        row = by_code.get(code)
        if row is None:
            reports.append(
                {
                    "model": code,
                    "vector_generated": False,
                    "visual_available": False,
                    "missing_attributes": ["not_in_catalog"],
                }
            )
            continue
        visual = combine_visual_embeddings(
            *(parse_embedding(row.get(column)) for column in VISUAL_EMBEDDING_COLUMNS)
        )
        reports.append(
            {
                "model": code,
                "vector_generated": visual is not None,
                "visual_available": visual is not None,
                "missing_attributes": missing_attributes(row),
            }
        )
    return reports


def encode_catalog_models(
    model: TwoTowerModel,
    catalog: dict[str, CatalogModel],
    codes: list[str],
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[str]]:
    encoders: FeatureEncoders = model.encoders
    config: TrainConfig = model.config
    rows = []
    kept: list[str] = []
    for code in codes:
        item = catalog.get(code)
        if item is None:
            continue
        rows.append(
            {
                "target_visual": item.visual,
                "target_cats": encoders.encode_model_categoricals(item.features),
                "history_visuals": [],
                "history_cats": [],
                "numeric": np.zeros(len(encoders.numeric_mean), dtype=np.float32),
                "age_id": 1,
                "target_code": code,
            }
        )
        kept.append(code)
    if not rows:
        return np.zeros((0, config.embedding_dim), dtype=np.float32), []
    batch = collate_batch(rows, config)
    model.eval()
    with torch.inference_mode():
        encoded = model.encode_models(batch, device=device)
    return encoded.detach().cpu().numpy().astype(np.float32), kept
