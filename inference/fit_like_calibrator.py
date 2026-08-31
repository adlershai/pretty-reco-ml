"""Fit like_score isotonic calibrator from historical purchase pairs.

Usage:
    python -m inference.fit_like_calibrator --latest
    python -m inference.fit_like_calibrator --snapshot local/snapshots/<timestamp>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.config import DEFAULT_SNAPSHOT_ROOT, load_dotenv
from data.dataset_builder import find_latest_snapshot
from inference.like_score import CALIBRATOR_FILENAME, LikeCalibrator
from inference.recommender import DEFAULT_ARTIFACT
from training.artifact import load_artifact
from training.dataset import TwoTowerDataset, collate_batch, load_catalog, load_split, resolve_snapshot_paths
from training.trainer import catalog_eval_rows, encode_catalog, select_device

logger = logging.getLogger("pretty-reco-ml.like-score")

NEGATIVES_PER_POSITIVE = 8


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit like_score calibrator from historical purchases.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--negatives", type=int, default=NEGATIVES_PER_POSITIVE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_snapshot(args: argparse.Namespace) -> Path:
    if args.snapshot:
        return args.snapshot.resolve()
    return find_latest_snapshot(args.snapshot_root)


def collect_pairs(
    *,
    model,
    dataset: TwoTowerDataset,
    catalog_matrix: np.ndarray,
    catalog_codes: list[str],
    negatives_per_positive: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    code_index = {str(code): index for index, code in enumerate(catalog_codes)}
    loader = DataLoader(
        dataset,
        batch_size=model.config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows, cfg=model.config: collate_batch(rows, cfg),
    )
    catalog_t = torch.from_numpy(catalog_matrix).to(device)
    rng = np.random.default_rng(seed)
    n_catalog = int(catalog_matrix.shape[0])
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            customers = model.encode_customers(batch, device=device)
            models = model.encode_models(batch, device=device)
            pos = (customers * models).sum(dim=-1).detach().cpu().numpy()
            positives.append(pos.astype(np.float64))
            scores = (customers @ catalog_t.transpose(0, 1)).detach().cpu().numpy()
            for row_index, code in enumerate(batch["target_codes"]):
                blocked = code_index.get(str(code))
                drawn: list[int] = []
                attempts = 0
                while len(drawn) < negatives_per_positive and attempts < negatives_per_positive * 8:
                    attempts += 1
                    candidate = int(rng.integers(0, n_catalog))
                    if candidate == blocked or candidate in drawn:
                        continue
                    drawn.append(candidate)
                if drawn:
                    negatives.append(scores[row_index, drawn].astype(np.float64))
    if not positives:
        raise ValueError("no historical positive pairs to calibrate")
    return np.concatenate(positives), np.concatenate(negatives) if negatives else np.zeros(0, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_dotenv()
    args = parse_args(argv)
    snapshot_dir, dataset_dir = resolve_snapshot_paths(resolve_snapshot(args))
    device = select_device()
    tower = load_artifact(args.artifact, device=device)
    catalog = load_catalog(snapshot_dir)
    train_frame = load_split(dataset_dir, "train")
    dataset = TwoTowerDataset(train_frame, tower.encoders, catalog, tower.config.max_history)
    catalog_matrix, catalog_codes = encode_catalog(
        tower,
        catalog_eval_rows(catalog, tower.encoders),
        tower.config,
        device,
    )
    positives, negatives = collect_pairs(
        model=tower,
        dataset=dataset,
        catalog_matrix=catalog_matrix,
        catalog_codes=catalog_codes,
        negatives_per_positive=max(1, int(args.negatives)),
        seed=int(args.seed),
        device=device,
    )
    period = snapshot_dir.name
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.is_file():
        period = f"{snapshot_dir.name}/{metadata_path.name}"
    calibrator = LikeCalibrator.fit_pairs(
        positives,
        negatives,
        metadata={
            "calibration_dataset_period": period,
            "candidate_sampling_method": "uniform_catalog_excluding_target",
            "negatives_per_positive": int(args.negatives),
            "artifact": str(args.artifact),
        },
    )
    output = args.output or (Path(args.artifact) / CALIBRATOR_FILENAME)
    calibrator.dump(output)
    logger.info(
        "wrote %s positives=%s negatives=%s knots=%s",
        output,
        calibrator.metadata.get("positive_pair_count"),
        calibrator.metadata.get("negative_pair_count"),
        int(calibrator.x.size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
