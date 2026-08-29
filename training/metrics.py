"""Ranking metrics for held-out purchases against the full catalog."""

from __future__ import annotations

import numpy as np


def _rank_of_target(scores: np.ndarray, target_index: int) -> int:
    order = np.argsort(-scores, kind="mergesort")
    matches = np.where(order == target_index)[0]
    if matches.size == 0:
        return int(scores.size)
    return int(matches[0]) + 1


def ranking_metrics(ranks: list[int], ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    if not ranks:
        return {**{f"recall@{k}": 0.0 for k in ks}, "ndcg@10": 0.0, "mrr": 0.0, "hit_rate@10": 0.0}
    array = np.asarray(ranks, dtype=np.float64)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = float((array <= k).mean())
    ndcg = np.where(array <= 10, 1.0 / np.log2(array + 1.0), 0.0)
    metrics["ndcg@10"] = float(ndcg.mean())
    metrics["mrr"] = float((1.0 / array).mean())
    metrics["hit_rate@10"] = metrics["recall@10"]
    metrics["examples"] = float(len(ranks))
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    keys = ["recall@5", "recall@10", "recall@20", "ndcg@10", "mrr", "hit_rate@10"]
    parts = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
    return " ".join(parts)
