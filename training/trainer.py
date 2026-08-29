"""Train a two-tower model with in-batch negatives and early stopping."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import TwoTowerDataset, collate_batch
from training.feature_encoders import FeatureEncoders
from training.metrics import format_metrics, ranking_metrics
from training.two_tower import TwoTowerModel, in_batch_softmax_loss

logger = logging.getLogger("pretty-reco-ml.training")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_catalog(
    model: TwoTowerModel,
    catalog_rows: list[dict[str, Any]],
    config: TrainConfig,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    codes = [str(row["target_code"]) for row in catalog_rows]
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(catalog_rows), batch_size):
            chunk = catalog_rows[start : start + batch_size]
            batch = collate_batch(chunk, config)
            encoded = model.encode_models(batch, device=device)
            vectors.append(encoded.detach().cpu().numpy())
    matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, config.embedding_dim), dtype=np.float32)
    return matrix.astype(np.float32), codes


def evaluate(
    model: TwoTowerModel,
    loader: DataLoader,
    catalog_matrix: np.ndarray,
    catalog_index: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ranks: list[int] = []
    catalog_t = torch.from_numpy(catalog_matrix).to(device)
    with torch.inference_mode():
        for batch in loader:
            customers = model.encode_customers(batch, device=device)
            scores = customers @ catalog_t.transpose(0, 1)
            score_np = scores.detach().cpu().numpy()
            for row_scores, code in zip(score_np, batch["target_codes"], strict=True):
                index = catalog_index.get(str(code))
                if index is None:
                    continue
                order = np.argsort(-row_scores, kind="mergesort")
                rank_hits = np.where(order == index)[0]
                ranks.append(int(rank_hits[0]) + 1 if rank_hits.size else int(row_scores.size))
    return ranking_metrics(ranks)


def train_model(
    *,
    model: TwoTowerModel,
    train_dataset: TwoTowerDataset,
    val_dataset: TwoTowerDataset,
    catalog_rows: list[dict[str, Any]],
    config: TrainConfig,
    device: torch.device | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    device = device or select_device()
    emit = log or (lambda message: logger.info("%s", message))
    model.to(device)
    collate = lambda rows: collate_batch(rows, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=len(train_dataset) > config.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics: dict[str, float] = {}
    best_recall = -1.0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    catalog_index = {str(row["target_code"]): index for index, row in enumerate(catalog_rows)}

    emit(f"device={device.type} train_rows={len(train_dataset)} val_rows={len(val_dataset)} dim={config.embedding_dim}")
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        epoch_start = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            customers, models = model(batch, device=device)
            loss = in_batch_softmax_loss(customers, models, config.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        mean_loss = epoch_loss / max(steps, 1)
        catalog_matrix, _codes = encode_catalog(model, catalog_rows, config, device)
        val_metrics = evaluate(model, val_loader, catalog_matrix, catalog_index, device)
        elapsed = time.perf_counter() - epoch_start
        row = {"epoch": epoch, "train_loss": mean_loss, "elapsed_sec": elapsed, **val_metrics}
        history.append(row)
        emit(
            f"epoch={epoch} train_loss={mean_loss:.4f} {format_metrics(val_metrics)} "
            f"elapsed={elapsed:.1f}s device={device.type}"
        )
        recall = val_metrics.get("recall@10", 0.0)
        if recall > best_recall + 1e-6:
            best_recall = recall
            best_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                emit(f"early stopping at epoch {epoch} (best recall@10={best_recall:.4f})")
                break
        if mean_loss < 0.05 and val_metrics.get("recall@10", 0.0) < best_recall - 0.02:
            emit("validation lagging training loss; continuing with early-stop rule")

    model.load_state_dict(best_state)
    model.to(device)
    catalog_matrix, _codes = encode_catalog(model, catalog_rows, config, device)
    final_val = evaluate(model, val_loader, catalog_matrix, catalog_index, device)
    return {
        "history": history,
        "best_validation": best_metrics or final_val,
        "validation": final_val,
        "device": device.type,
        "elapsed_sec": time.perf_counter() - started,
        "state_dict": best_state,
    }


def catalog_eval_rows(catalog, encoders: FeatureEncoders) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code, item in catalog.items():
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
    return rows
