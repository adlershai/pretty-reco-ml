"""Phase 3 vector and ranking sanity checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from inference.csv_models import ANONYMOUS_CUSTOMER_ID


def check_vectors(vectors: np.ndarray, *, dim: int, name: str, atol: float = 1e-3) -> None:
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {vectors.shape}")
    if vectors.shape[1] != dim:
        raise ValueError(f"{name} dim {vectors.shape[1]} != {dim}")
    if vectors.size and not np.isfinite(vectors).all():
        raise ValueError(f"{name} contains non-finite values")
    if vectors.shape[0] == 0:
        return
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError(f"{name} L2 norms are not ~1 (min={norms.min():.6f} max={norms.max():.6f})")


def check_customer_ranking(rows: Sequence[dict[str, Any]], *, new_model_codes: set[str]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    for model, group in by_model.items():
        ordered = sorted(group, key=lambda item: int(item["customer_rank"]))
        ranks = [int(item["customer_rank"]) for item in ordered]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{model}: customer_rank must be 1..n without gaps")
        scores = [float(item["similarity_score"]) for item in ordered]
        if scores != sorted(scores, reverse=True):
            raise ValueError(f"{model}: similarity_score is not descending")
        ids = [str(item["customer_id"]) for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{model}: duplicate customer in ranking")
        if ANONYMOUS_CUSTOMER_ID in ids:
            raise ValueError(f"{model}: anonymous customer {ANONYMOUS_CUSTOMER_ID} in ranking")
    missing = set(new_model_codes) - set(by_model)
    if missing:
        raise ValueError(f"ranking missing models: {sorted(missing)}")


def check_history_excludes_new_models(records: Sequence[dict[str, Any]], new_model_codes: set[str]) -> None:
    for record in records:
        history = [str(code) for code in record.get("history_model_ids") or []]
        leaked = [code for code in history if code in new_model_codes]
        if leaked:
            raise ValueError(f"customer {record.get('customer_id')} history contains new models {leaked}")
