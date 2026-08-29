from __future__ import annotations

import numpy as np
import torch

from tests.two_tower_fixtures import make_model, tiny_batch
from training.model_tower import categoricals_from_batch


def test_model_tower_output_shape_and_norm() -> None:
    model, _encoders, config = make_model(8)
    batch = tiny_batch(model.encoders, config)
    visual = batch["target_visual"]
    cats = categoricals_from_batch(batch, "target_cat_")
    encoded = model.model_tower(visual, cats)
    assert encoded.shape == (4, 8)
    norms = torch.linalg.vector_norm(encoded, dim=-1)
    np.testing.assert_allclose(norms.detach().numpy(), np.ones(4), rtol=1e-5, atol=1e-5)


def test_unseen_model_still_encodes() -> None:
    model, _encoders, _config = make_model(8)
    vector = model.encode_model(
        {
            "main_category": "brand_new_category",
            "sub_category": "never_seen",
            "material": "silk",
            "color": "chartreuse",
            "season": "S99",
            "last_type": "nope",
            "visual_embedding": np.linspace(0.1, 1.0, 8).astype(np.float32),
        }
    )
    assert vector.shape == (8,)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)
    assert np.isfinite(vector).all()
