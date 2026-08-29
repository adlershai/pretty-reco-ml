from __future__ import annotations

import torch

from tests.two_tower_fixtures import make_model, tiny_batch
from training.artifact import load_artifact, save_artifact
from training.metrics import ranking_metrics
from training.two_tower import in_batch_softmax_loss


def test_one_batch_forward_backward() -> None:
    model, _encoders, config = make_model(8)
    batch = tiny_batch(model.encoders, config)
    model.train()
    customers, models = model(batch)
    loss = in_batch_softmax_loss(customers, models, config.temperature)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_ranking_metrics_known_ranks() -> None:
    metrics = ranking_metrics([1, 5, 11, 20])
    assert metrics["recall@5"] == 0.5
    assert metrics["recall@10"] == 0.5
    assert metrics["recall@20"] == 1.0
    assert metrics["hit_rate@10"] == metrics["recall@10"]
    assert 0 < metrics["mrr"] < 1
    assert 0 < metrics["ndcg@10"] < 1


def test_load_artifact_accepts_string_path(tmp_path) -> None:
    model, encoders, config = make_model(8)
    save_artifact(
        artifact_dir=tmp_path / "v1",
        model=model,
        config=config,
        encoders=encoders,
        metrics={},
        metadata={"selected_embedding_dim": 8},
    )
    loaded = load_artifact(str(tmp_path / "v1"))
    assert loaded.config.embedding_dim == 8
