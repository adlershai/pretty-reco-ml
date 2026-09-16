"""Visual product identification: isolate shoe, encode, MAX-view catalog search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from embeddings.catalog_index import CatalogIndex
from embeddings.isolate import CropBox, crop_image, isolation_applied, propose_crops
from embeddings.order_extract import extract_order as default_extract_order, order_payload
from embeddings.relevance import (
    IMAGE_KIND_GARBAGE,
    IMAGE_KIND_ORDER,
    IMAGE_KIND_SHOE,
    RELEVANCE_FOOTWEAR,
    RELEVANCE_IRRELEVANT,
)
from embeddings.verifier import (
    build_verifier_request,
    crop_jpeg_base64,
    decide_identity,
    verify_top_value,
)
from embeddings.vision_encoder import VisionEncoder
from evaluation.image_match import ModelHit

DEFAULT_TOP = 10
SCORE_TIE = 1e-6


def _prefer_crop(score: float, box: CropBox, best_score: float, best_box: CropBox) -> bool:
    """Highest catalog similarity wins; ties prefer an isolated (non-full) crop."""
    if not np.isfinite(score):
        return False
    if not np.isfinite(best_score) or score > best_score + SCORE_TIE:
        return True
    if score < best_score - SCORE_TIE:
        return False
    isolated = box.reason != "full"
    best_isolated = best_box.reason != "full"
    if isolated != best_isolated:
        return isolated
    return box.area < best_box.area


@dataclass(frozen=True)
class ImageMatchResult:
    match: ModelHit | None
    candidates: list[ModelHit]
    crop: CropBox
    shoe_isolated: bool
    relevance: str
    scores: dict[str, float]
    embedding_model: str
    crop_scores: list[tuple[CropBox, float]]
    image_kind: str = IMAGE_KIND_SHOE
    order: dict[str, str | None] | None = None


def _model_payload(hit: ModelHit) -> dict[str, Any]:
    return {
        "model": hit.model,
        "score": hit.model_score,
        "best_image_type": hit.best_image_type,
        "model_id": hit.model_id,
    }


def _scene_kind(encoder: VisionEncoder, vector: np.ndarray) -> tuple[str, dict[str, float]]:
    classify_scene = getattr(encoder, "classify_scene", None)
    if callable(classify_scene):
        return classify_scene(vector)
    relevance, scores = encoder.classify_relevance(vector)
    if relevance == RELEVANCE_FOOTWEAR:
        return IMAGE_KIND_SHOE, scores
    return IMAGE_KIND_GARBAGE, scores


def match_image(
    image: Image.Image,
    encoder: VisionEncoder,
    catalog: CatalogIndex,
    *,
    top: int = DEFAULT_TOP,
    extract_order_fn: Any = None,
) -> ImageMatchResult:
    """Route shoe / order / garbage, then rank catalog models only for shoes."""
    rgb = image.convert("RGB")
    boxes = propose_crops(rgb)
    crops = [crop_image(rgb, box) for box in boxes]
    vectors = encoder.encode_batch(crops)
    scene, scene_scores = _scene_kind(encoder, vectors[0])

    if scene == IMAGE_KIND_ORDER:
        extract = extract_order_fn or default_extract_order
        return ImageMatchResult(
            match=None,
            candidates=[],
            crop=boxes[0],
            shoe_isolated=False,
            relevance=RELEVANCE_IRRELEVANT,
            scores=scene_scores,
            embedding_model=encoder.embedding_model,
            crop_scores=[],
            image_kind=IMAGE_KIND_ORDER,
            order=order_payload(extract(rgb)),
        )

    if scene == IMAGE_KIND_GARBAGE:
        return ImageMatchResult(
            match=None,
            candidates=[],
            crop=boxes[0],
            shoe_isolated=False,
            relevance=RELEVANCE_IRRELEVANT,
            scores=scene_scores,
            embedding_model=encoder.embedding_model,
            crop_scores=[],
            image_kind=IMAGE_KIND_GARBAGE,
            order=None,
        )

    best_index: int | None = None
    best_score = float("-inf")
    crop_scores: list[tuple[CropBox, float]] = []
    gate_scores = {"footwear": 0.0, "irrelevant": 0.0}
    for index, vector in enumerate(vectors):
        relevance, scores = encoder.classify_relevance(vector)
        models = catalog.search_models(vector, top=1)
        score = models[0].model_score if models else float("-inf")
        crop_scores.append((boxes[index], float(score) if np.isfinite(score) else float("-inf")))
        if relevance != RELEVANCE_FOOTWEAR:
            continue
        if best_index is None or _prefer_crop(score, boxes[index], best_score, boxes[best_index]):
            best_index = index
            best_score = score
            gate_scores = scores

    if best_index is None:
        chosen = boxes[0]
        isolated = isolation_applied(chosen, rgb)
        _relevance, scores = encoder.classify_relevance(vectors[0])
        return ImageMatchResult(
            match=None,
            candidates=[],
            crop=chosen,
            shoe_isolated=isolated,
            relevance=RELEVANCE_IRRELEVANT,
            scores=scores,
            embedding_model=encoder.embedding_model,
            crop_scores=crop_scores,
            image_kind=IMAGE_KIND_GARBAGE,
            order=None,
        )

    chosen = boxes[best_index]
    query = vectors[best_index]
    isolated = isolation_applied(chosen, rgb)
    candidates = catalog.search_models(query, top=top)
    return ImageMatchResult(
        match=candidates[0] if candidates else None,
        candidates=candidates,
        crop=chosen,
        shoe_isolated=isolated,
        relevance=RELEVANCE_FOOTWEAR,
        scores=gate_scores,
        embedding_model=encoder.embedding_model,
        crop_scores=crop_scores,
        image_kind=IMAGE_KIND_SHOE,
        order=None,
    )


def match_result_to_dict(
    result: ImageMatchResult,
    *,
    image_size: tuple[int, int],
    image: Image.Image | None = None,
    catalog: CatalogIndex | None = None,
    verify_top: int | None = None,
) -> dict[str, Any]:
    width, height = image_size
    candidates = [_model_payload(hit) for hit in result.candidates]
    footwear = result.relevance == RELEVANCE_FOOTWEAR and bool(candidates)
    preprocessing = {
        "shoe_isolated": result.shoe_isolated,
        "crop": result.crop.as_list(),
        "reason": result.crop.reason,
        "image_size": [int(width), int(height)],
        "crop_jpeg_base64": None,
    }
    if image is not None:
        preprocessing["crop_jpeg_base64"] = crop_jpeg_base64(image, result.crop)
    verifier = None
    if footwear:
        verifier = build_verifier_request(
            candidates,
            catalog_rows=list(catalog.rows) if catalog is not None else [],
            verify_top=verify_top,
        )
    kind = result.image_kind or (IMAGE_KIND_SHOE if footwear else IMAGE_KIND_GARBAGE)
    if kind == IMAGE_KIND_ORDER:
        status = "order"
    elif kind == IMAGE_KIND_GARBAGE or not footwear:
        status = "garbage"
    else:
        status = "needs_verification"
    return {
        "status": status,
        "match": None,
        "candidates": candidates,
        "preprocessing": preprocessing,
        "verifier": verifier,
        "relevance": result.relevance,
        "scores": result.scores,
        "embedding_model": result.embedding_model,
        "image_kind": kind,
        "order": result.order,
    }


def decide_match(
    openai_output: Any,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = decide_identity(openai_output, candidates)
    chosen = None
    if decision.status == "match" and decision.model:
        for hit in candidates:
            if str(hit.get("model") or "") == decision.model:
                chosen = hit
                break
        if chosen is None:
            chosen = {
                "model": decision.model,
                "score": None,
                "best_image_type": None,
                "model_id": None,
            }
    return {
        "status": decision.status,
        "match": chosen if decision.status == "match" else None,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "verification": decision.verification,
        "candidates": candidates,
    }


def run_match(
    image_bytes: bytes,
    encoder: VisionEncoder,
    catalog: CatalogIndex,
    *,
    top: int = DEFAULT_TOP,
    verify_top: int | None = None,
) -> dict[str, Any]:
    from embeddings.worker import decode_image

    retrieve_top = max(int(top), verify_top_value(verify_top))
    image = decode_image(image_bytes)
    result = match_image(image, encoder, catalog, top=retrieve_top)
    return match_result_to_dict(
        result,
        image_size=image.size,
        image=image,
        catalog=catalog,
        verify_top=verify_top,
    )
