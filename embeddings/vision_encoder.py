"""SigLIP vision encoder: load once, preprocess, return L2-normalized embeddings."""

from __future__ import annotations

import logging
import os
from typing import Sequence

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, SiglipImageProcessorPil

from embeddings.relevance import (
    DELIVERY_PROMPTS,
    FOOTWEAR_PROMPTS,
    GARBAGE_PROMPTS,
    JUNK_PROMPTS,
    ORDER_PROMPTS,
    classify_non_shoe,
    classify_relevance,
    classify_scene,
)

logger = logging.getLogger(__name__)

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
EMBEDDING_DIMENSION = 768


def select_device() -> torch.device:
    """Use CUDA when a GPU is available; otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VisionEncoder:
    """Reusable image encoder. The Hugging Face model is loaded once per instance."""

    def __init__(
        self,
        model_id: str = SIGLIP_MODEL_ID,
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or select_device()
        logger.info("vision device: %s", self.device.type)

        self._processor = SiglipImageProcessorPil.from_pretrained(model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

        hidden = getattr(self._model.config, "vision_config", None)
        projection = getattr(self._model.config, "projection_dim", None)
        if projection is not None:
            self.embedding_dimension = int(projection)
        elif hidden is not None and getattr(hidden, "hidden_size", None):
            self.embedding_dimension = int(hidden.hidden_size)
        else:
            self.embedding_dimension = EMBEDDING_DIMENSION

        self._footwear_text = self.encode_text(FOOTWEAR_PROMPTS)
        self._junk_text = self.encode_text(JUNK_PROMPTS)
        self._order_text = self.encode_text(ORDER_PROMPTS)
        self._delivery_text = self.encode_text(DELIVERY_PROMPTS)
        self._garbage_text = self.encode_text(GARBAGE_PROMPTS)

    @property
    def embedding_model(self) -> str:
        return self.model_id

    def encode(self, image: Image.Image) -> np.ndarray:
        """Encode one RGB image to a 1-D float32 unit vector."""
        return self.encode_batch([image])[0]

    def encode_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode images to an (N, D) float32 array of L2-normalized vectors."""
        if not images:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        rgb = [image.convert("RGB") for image in images]
        processed = self._processor(images=rgb, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)

        with torch.inference_mode():
            vision_outputs = self._model.vision_model(pixel_values=pixel_values)
            features = vision_outputs.pooler_output
            features = F.normalize(features, p=2, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode prompts to an (N, D) float32 array of L2-normalized vectors."""
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        encoded = self._tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device) if "attention_mask" in encoded else None

        with torch.inference_mode():
            if attention_mask is not None:
                outputs = self._model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self._model.get_text_features(input_ids=input_ids)
            features = getattr(outputs, "pooler_output", outputs)
            features = F.normalize(features, p=2, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def classify_relevance(self, image_vector: np.ndarray) -> tuple[str, dict[str, float]]:
        """Zero-shot footwear vs junk on the same image vector used for catalog NN."""
        return classify_relevance(image_vector, self._footwear_text, self._junk_text)

    def classify_scene(self, image_vector: np.ndarray) -> tuple[str, dict[str, float]]:
        """Full-frame shoe vs order screenshot vs garbage (diagnostics)."""
        return classify_scene(
            image_vector,
            self._footwear_text,
            self._garbage_text,
            self._order_text,
        )

    def classify_non_shoe(self, image_vector: np.ndarray) -> tuple[str, dict[str, float]]:
        """Order vs delivery notice vs garbage after shoe detection has failed."""
        return classify_non_shoe(
            image_vector,
            self._garbage_text,
            self._order_text,
            self._delivery_text,
        )
