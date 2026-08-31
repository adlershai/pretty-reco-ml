"""Train and compare compact two-tower recommenders.

Usage:
    python -m training.train --snapshot local/snapshots/<timestamp>
    python -m training.train --snapshot local/snapshots/<timestamp>/dataset --embedding-dim 128
    python -m training.train --snapshot local/snapshots/<timestamp> --compare 64,128
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.config import DEFAULT_SNAPSHOT_ROOT, LEGACY_SNAPSHOT_ROOT, REPO_ROOT, load_dotenv
from data.schemas import OBSERVATION_START
from training.config import (
    DEFAULT_EMBEDDING_DIMS,
    EXPLORATORY_EMBEDDING_DIM,
    RECALL_MARGIN,
    TrainConfig,
)
from training.dataset import TwoTowerDataset, collate_batch, load_catalog, load_split, resolve_snapshot_paths
from training.feature_encoders import FeatureEncoders, parse_json_map
from training.artifact import load_artifact, save_artifact
from training.metrics import format_metrics
from training.trainer import catalog_eval_rows, encode_catalog, evaluate, select_device, train_model
from training.two_tower import TwoTowerModel

logger = logging.getLogger("pretty-reco-ml.training")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def configure_threads(num_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(num_threads, 2)))
    except RuntimeError:
        pass


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def fit_encoders(train_frame, catalog) -> FeatureEncoders:
    encoders = FeatureEncoders()
    encoders.fit_models([item.features for item in catalog.values()])
    target_features = [parse_json_map(row) for row in train_frame["target_model_features"].tolist()]
    encoders.fit_models(target_features)
    static_rows = [parse_json_map(row) for row in train_frame["customer_static_features"].tolist()]
    behavior_rows = [parse_json_map(row) for row in train_frame["customer_behavior_features"].tolist()]
    encoders.fit_age_groups(row.get("age_group") for row in static_rows)
    encoders.fit_numeric(list(zip(static_rows, behavior_rows, strict=True)))
    return encoders


def select_dimension(results: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(results, key=lambda row: (-float(row["validation"]["recall@10"]), int(row["embedding_dim"])))
    best = ranked[0]
    threshold = float(best["validation"]["recall@10"]) - RECALL_MARGIN
    competitive = [row for row in ranked if float(row["validation"]["recall@10"]) >= threshold]
    return min(competitive, key=lambda row: int(row["embedding_dim"]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pretty two-tower recommender V1.")
    parser.add_argument("--snapshot", type=Path, help="Snapshot directory or dataset directory")
    parser.add_argument("--latest", action="store_true", help="Use the newest snapshot under local/snapshots/")
    parser.add_argument("--embedding-dim", type=int, help="Train a single embedding dimension")
    parser.add_argument("--compare", default="", help="Comma-separated dims to compare, e.g. 64,128")
    parser.add_argument("--include-256", action="store_true", help="Also train the exploratory 256D model")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "the_pretty_model_v1",
    )
    return parser.parse_args(argv)


def dims_from_args(args: argparse.Namespace) -> list[int]:
    if args.embedding_dim and not args.compare:
        dims = [int(args.embedding_dim)]
    elif args.compare:
        dims = [int(part.strip()) for part in args.compare.split(",") if part.strip()]
    else:
        dims = list(DEFAULT_EMBEDDING_DIMS)
    if args.include_256 and EXPLORATORY_EMBEDDING_DIM not in dims:
        dims.append(EXPLORATORY_EMBEDDING_DIM)
    return dims


def latest_snapshot(root: Path) -> Path:
    roots = [root]
    if Path(root).resolve() == DEFAULT_SNAPSHOT_ROOT.resolve():
        roots.append(LEGACY_SNAPSHOT_ROOT)
    candidates: list[Path] = []
    for folder in roots:
        if not folder.is_dir():
            continue
        candidates.extend(
            path for path in folder.iterdir() if path.is_dir() and (path / "dataset" / "train.parquet").is_file()
        )
    if not candidates:
        raise FileNotFoundError(f"no dataset snapshots under {root}")
    return max(candidates, key=lambda path: path.name)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)
    num_threads = args.num_threads or int(os.environ.get("TORCH_NUM_THREADS", "2"))
    configure_threads(max(1, num_threads))
    seed_everything(args.seed)
    device = select_device()

    if args.latest and args.snapshot:
        logger.error("use either --snapshot or --latest, not both")
        return 1
    if not args.snapshot and not args.latest:
        args.latest = True
    snapshot_arg = args.snapshot if args.snapshot else latest_snapshot(DEFAULT_SNAPSHOT_ROOT)
    snapshot_dir, dataset_dir = resolve_snapshot_paths(snapshot_arg)

    train_frame = load_split(dataset_dir, "train")
    val_frame = load_split(dataset_dir, "validation")
    test_frame = load_split(dataset_dir, "test")
    catalog = load_catalog(snapshot_dir)
    encoders = fit_encoders(train_frame, catalog)

    dims = dims_from_args(args)
    logger.info(
        "training dims=%s snapshot=%s train=%s val=%s test=%s catalog=%s",
        dims,
        snapshot_dir,
        len(train_frame),
        len(val_frame),
        len(test_frame),
        len(catalog),
    )
    logger.info("loss=in_batch_sampled_softmax (in-batch negatives, not true business rejects)")

    comparison: list[dict[str, Any]] = []
    trained: dict[int, dict[str, Any]] = {}
    for dim in dims:
        config = TrainConfig.from_embedding_dim(
            dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            num_threads=num_threads,
            max_history=args.max_history,
            patience=args.patience,
        )
        train_dataset = TwoTowerDataset(train_frame, encoders, catalog, config.max_history)
        val_dataset = TwoTowerDataset(val_frame, encoders, catalog, config.max_history)
        model = TwoTowerModel(config, encoders)
        result = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            catalog_rows=catalog_eval_rows(catalog, encoders),
            config=config,
            device=device,
        )
        model.load_state_dict(result["state_dict"])
        test_dataset = TwoTowerDataset(test_frame, encoders, catalog, config.max_history)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda rows, cfg=config: collate_batch(rows, cfg),
        )
        catalog_matrix, _codes = encode_catalog(
            model, catalog_eval_rows(catalog, encoders), config, device
        )
        catalog_index = {code: index for index, code in enumerate(_codes)}
        test_metrics = evaluate(model, test_loader, catalog_matrix, catalog_index, device)
        logger.info("dim=%s test %s", dim, format_metrics(test_metrics))
        summary = {
            "embedding_dim": dim,
            "hidden_dim": config.hidden_dim,
            "validation": result["validation"],
            "test": test_metrics,
            "train_loss_last": result["history"][-1]["train_loss"] if result["history"] else None,
            "epochs_ran": len(result["history"]),
            "elapsed_sec": result["elapsed_sec"],
        }
        comparison.append(summary)
        trained[dim] = {"model": model, "config": config, "result": result, "test": test_metrics}

    selected = select_dimension(comparison) if len(comparison) > 1 else comparison[0]
    chosen = trained[int(selected["embedding_dim"])]
    artifact_dir = args.artifact_dir
    metadata = {
        "model_version": artifact_dir.name,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(snapshot_dir),
        "dataset": str(dataset_dir),
        "train_row_count": int(len(train_frame)),
        "validation_row_count": int(len(val_frame)),
        "test_row_count": int(len(test_frame)),
        "catalog_models": int(len(catalog)),
        "selected_embedding_dim": int(selected["embedding_dim"]),
        "observation_start": OBSERVATION_START,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device.type,
        "loss": "in_batch_sampled_softmax",
        "lifecycle": {
            "new_or_updated_customer": "re-run Customer Tower only",
            "new_model": "run Model Tower only",
            "end_of_season": "retrain two towers on updated history, promote if better, regenerate all vectors",
            "retrain_trigger": "~3,000–4,000 new purchases / end of season",
        },
    }
    metrics_payload = {
        "selected": selected,
        "comparison": comparison,
        "validation": chosen["result"]["validation"],
        "test": chosen["test"],
        "history": chosen["result"]["history"],
    }
    save_artifact(
        artifact_dir=artifact_dir,
        model=chosen["model"],
        config=chosen["config"],
        encoders=encoders,
        metrics=metrics_payload,
        metadata=metadata,
    )
    (artifact_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print("=== Two-tower comparison ===")
    print(f"{'dim':>6}  {'Recall@10':>10}  {'Recall@20':>10}  {'NDCG@10':>10}  {'MRR':>8}  {'Hit@10':>8}")
    for row in comparison:
        val = row["validation"]
        print(
            f"{row['embedding_dim']:6d}  {val['recall@10']:10.4f}  {val['recall@20']:10.4f}  "
            f"{val['ndcg@10']:10.4f}  {val['mrr']:8.4f}  {val['hit_rate@10']:8.4f}"
        )
    print(f"selected dim={selected['embedding_dim']} (smallest competitive on validation Recall@10, margin={RECALL_MARGIN})")
    print(f"artifact written to {artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
