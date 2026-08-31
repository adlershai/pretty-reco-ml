"""Monotonic mapping from raw tower cosine to a 0–1 like_score.

like_score is taste affinity, not a purchase probability. Recency is not an input.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CALIBRATOR_FILENAME = "like_calibrator.json"


def pool_adjacent_violators(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Increasing PAV. `values` must already be ordered by the covariate."""
    means = np.asarray(values, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = int(means.shape[0])
    if n == 0:
        return means
    blocks: list[list[float]] = []
    for mean, weight in zip(means, mass, strict=True):
        blocks.append([float(mean) * float(weight), float(weight), 1.0])
        while len(blocks) >= 2:
            left_sum, left_w, left_n = blocks[-2]
            right_sum, right_w, right_n = blocks[-1]
            if left_sum / left_w <= right_sum / right_w + 1e-15:
                break
            blocks[-2] = [left_sum + right_sum, left_w + right_w, left_n + right_n]
            blocks.pop()
    fitted = np.empty(n, dtype=np.float64)
    cursor = 0
    for total, weight, width in blocks:
        count = int(width)
        fitted[cursor : cursor + count] = total / weight
        cursor += count
    return fitted


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binary labels y in {0,1} vs raw similarity x → unique increasing knots."""
    sim = np.asarray(x, dtype=np.float64).reshape(-1)
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    if sim.size == 0:
        raise ValueError("no calibration pairs")
    order = np.argsort(sim, kind="mergesort")
    sorted_x = sim[order]
    fitted = pool_adjacent_violators(labels[order], np.ones(sim.size, dtype=np.float64))
    knots_x: list[float] = []
    knots_y: list[float] = []
    for value_x, value_y in zip(sorted_x, fitted, strict=True):
        clipped = float(np.clip(value_y, 0.0, 1.0))
        if knots_x and abs(value_x - knots_x[-1]) < 1e-12:
            knots_y[-1] = max(knots_y[-1], clipped)
            continue
        if knots_y and clipped + 1e-12 < knots_y[-1]:
            clipped = knots_y[-1]
        if knots_y and abs(clipped - knots_y[-1]) < 1e-6 and abs(value_x - knots_x[-1]) < 1e-4:
            knots_x[-1] = float(value_x)
            knots_y[-1] = clipped
            continue
        knots_x.append(float(value_x))
        knots_y.append(clipped)
    if len(knots_x) >= 3:
        kept_x = [knots_x[0]]
        kept_y = [knots_y[0]]
        for value_x, value_y in zip(knots_x[1:-1], knots_y[1:-1], strict=True):
            if abs(value_y - kept_y[-1]) > 1e-5:
                kept_x.append(value_x)
                kept_y.append(value_y)
        kept_x.append(knots_x[-1])
        kept_y.append(knots_y[-1])
        knots_x, knots_y = kept_x, kept_y
    return np.asarray(knots_x, dtype=np.float64), np.asarray(knots_y, dtype=np.float64)


class LikeCalibrator:
    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        xs = np.asarray(x, dtype=np.float64).reshape(-1)
        ys = np.asarray(y, dtype=np.float64).reshape(-1)
        if xs.size < 2 or xs.size != ys.size:
            raise ValueError("calibrator needs at least two (similarity, like_score) knots")
        if np.any(np.diff(xs) < 0):
            raise ValueError("calibrator x must be non-decreasing")
        if np.any(np.diff(ys) < -1e-12):
            raise ValueError("calibrator must be monotonic non-decreasing")
        self.x = xs
        self.y = np.clip(ys, 0.0, 1.0)
        self.metadata = dict(metadata or {})

    def transform(self, similarities: np.ndarray | Sequence[float]) -> np.ndarray:
        raw = np.asarray(similarities, dtype=np.float64)
        mapped = np.interp(raw, self.x, self.y)
        return np.clip(mapped, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["x"] = [float(value) for value in self.x]
        payload["y"] = [float(value) for value in self.y]
        return payload

    def dump(self, path: Path | str) -> None:
        destination = Path(path)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> LikeCalibrator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        x = payload["x"]
        y = payload["y"]
        metadata = {key: value for key, value in payload.items() if key not in {"x", "y"}}
        return cls(x, y, metadata=metadata)

    @classmethod
    def fit_pairs(
        cls,
        positive_similarities: Sequence[float],
        negative_similarities: Sequence[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> LikeCalibrator:
        positives = np.asarray(positive_similarities, dtype=np.float64).reshape(-1)
        negatives = np.asarray(negative_similarities, dtype=np.float64).reshape(-1)
        x = np.concatenate([positives, negatives])
        y = np.concatenate([np.ones(positives.size), np.zeros(negatives.size)])
        knots_x, knots_y = fit_isotonic(x, y)
        extra = {
            "calibration_method": "isotonic_pav",
            "positive_pair_count": int(positives.size),
            "negative_pair_count": int(negatives.size),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        extra.update(metadata or {})
        return cls(knots_x, knots_y, metadata=extra)


def load_calibrator(artifact_dir: Path | str) -> LikeCalibrator:
    path = Path(artifact_dir) / CALIBRATOR_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"like-score calibrator missing: {path}")
    return LikeCalibrator.load(path)
