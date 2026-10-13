"""Visual product identification: isolate shoe, encode, MAX-view catalog search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from embeddings.catalog_index import CatalogIndex
from embeddings.delivery_extract import (
    delivery_payload,
    extract_delivery as default_extract_delivery,
)
from embeddings.document_kind import (
    classify_document_kind,
    has_strong_delivery_evidence,
    has_strong_order_evidence,
)
from embeddings.isolate import MIN_PANEL_AREA, CropBox, crop_image, isolation_applied, propose_crops
from embeddings.order_extract import (
    extract_order as default_extract_order,
    ocr_image_text,
    order_payload,
)
from embeddings.relevance import (
    IMAGE_KIND_DELIVERY,
    IMAGE_KIND_GARBAGE,
    IMAGE_KIND_ORDER,
    IMAGE_KIND_SHOE,
    IMAGE_KIND_UNKNOWN,
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
# Issue #4 WATI set, 2026-09-21: 753/736 max catalog 0.572; 1070 PDP min 0.655.
CATALOG_SHOE_MIN = 0.62
# classify_relevance sends equality to footwear. On a relevance-only/unknown scene,
# junk must beat footwear by this margin. classify_scene garbage is enough on its own.
GARBAGE_MARGIN = 0.05


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
    delivery: dict[str, str | None] | None = None
    unknown_fallback: bool = False


def _model_payload(hit: ModelHit) -> dict[str, Any]:
    return {
        "model": hit.model,
        "score": hit.model_score,
        "best_image_type": hit.best_image_type,
        "model_id": hit.model_id,
    }


def _crop_fraction(box: CropBox, image: Image.Image) -> float:
    width, height = image.size
    return box.area / float(max(1, width * height))


def _usable_shoe_crop(box: CropBox, image: Image.Image) -> bool:
    """Tiny thumbnails inside an order table must not count as the whole image."""
    if box.reason == "full":
        return True
    return _crop_fraction(box, image) >= MIN_PANEL_AREA


def _fallback_crop_index(
    crop_scores: list[tuple[CropBox, float]],
    boxes: list[CropBox],
) -> int:
    """Highest catalog score among proposed crops; low similarity is allowed."""
    best_index = 0
    best_score = float("-inf")
    best_box = boxes[0]
    for index, (box, score) in enumerate(crop_scores):
        if index == 0:
            if np.isfinite(score):
                best_score = score
            continue
        if _prefer_crop(score, box, best_score, best_box):
            best_index = index
            best_score = score
            best_box = box
    return best_index


def _catalog_score(catalog: CatalogIndex, vector: np.ndarray) -> float:
    hits = catalog.search_models(vector, top=1)
    if not hits:
        return float("-inf")
    score = float(hits[0].model_score)
    return score if np.isfinite(score) else float("-inf")


def _is_identifiable_shoe(
    relevance: str,
    catalog_score: float,
    box: CropBox,
    image: Image.Image,
) -> bool:
    """Step 1: footwear prompt or strong PB catalog evidence on a usable crop."""
    if not _usable_shoe_crop(box, image):
        return False
    if relevance == RELEVANCE_FOOTWEAR:
        return True
    return catalog_score >= CATALOG_SHOE_MIN


def _junk_score(scores: dict[str, float]) -> float:
    return max(
        float(scores.get(IMAGE_KIND_GARBAGE) or 0.0),
        float(scores.get(RELEVANCE_IRRELEVANT) or 0.0),
    )


def _confident_garbage(scene_kind: str, scores: dict[str, float]) -> bool:
    """Positive irrelevance, not a leftover after a failed shoe label.

    classify_scene garbage means garbage prompts beat footwear. That is enough.
    A relevance-only miss (unknown scene) needs junk to beat footwear by
    GARBAGE_MARGIN. Shoe/order/delivery scene labels are never garbage here.
    """
    if scene_kind in {IMAGE_KIND_SHOE, IMAGE_KIND_ORDER, IMAGE_KIND_DELIVERY}:
        return False
    if scene_kind == IMAGE_KIND_GARBAGE:
        return True
    footwear = float(scores.get(RELEVANCE_FOOTWEAR) or 0.0)
    junk = _junk_score(scores)
    if not np.isfinite(footwear) or not np.isfinite(junk):
        return False
    return junk >= footwear + GARBAGE_MARGIN


def _scene_kind(encoder: VisionEncoder, vector: np.ndarray) -> tuple[str, dict[str, float]]:
    classify_scene = getattr(encoder, "classify_scene", None)
    if callable(classify_scene):
        return classify_scene(vector)
    relevance, scores = encoder.classify_relevance(vector)
    if relevance == RELEVANCE_FOOTWEAR:
        return IMAGE_KIND_SHOE, scores
    return IMAGE_KIND_UNKNOWN, scores


def _non_shoe_kind(encoder: VisionEncoder, vector: np.ndarray) -> tuple[str, dict[str, float]]:
    classify_non_shoe = getattr(encoder, "classify_non_shoe", None)
    if callable(classify_non_shoe):
        return classify_non_shoe(vector)
    scene, scores = _scene_kind(encoder, vector)
    if scene == IMAGE_KIND_SHOE:
        return IMAGE_KIND_GARBAGE, scores
    return scene, scores


def _document_result(
    *,
    boxes: list[CropBox],
    crop_scores: list[tuple[CropBox, float]],
    scores: dict[str, float],
    embedding_model: str,
    image_kind: str,
    order: dict[str, str | None] | None = None,
    delivery: dict[str, str | None] | None = None,
) -> ImageMatchResult:
    return ImageMatchResult(
        match=None,
        candidates=[],
        crop=boxes[0],
        shoe_isolated=False,
        relevance=RELEVANCE_IRRELEVANT,
        scores=scores,
        embedding_model=embedding_model,
        crop_scores=crop_scores,
        image_kind=image_kind,
        order=order,
        delivery=delivery,
    )


def match_image(
    image: Image.Image,
    encoder: VisionEncoder,
    catalog: CatalogIndex,
    *,
    top: int = DEFAULT_TOP,
    extract_order_fn: Any = None,
    extract_delivery_fn: Any = None,
    ocr_fn: Any = None,
    include_hero: bool | None = None,
) -> ImageMatchResult:
    """Whole-image document intent first; shoe only if the frame is not an order/delivery."""
    rgb = image.convert("RGB")
    text_fn = ocr_fn if ocr_fn is not None else ocr_image_text
    text = text_fn(rgb)
    if extract_order_fn:
        order_fields = extract_order_fn(rgb)
    else:
        order_fields = default_extract_order(rgb, text=text)
    if extract_delivery_fn:
        delivery_fields = extract_delivery_fn(rgb)
    else:
        delivery_fields = default_extract_delivery(rgb, text=text)

    boxes = propose_crops(rgb, include_hero=include_hero)
    crops = [crop_image(rgb, box) for box in boxes]
    vectors = encoder.encode_batch(crops)
    scene, scene_scores = _non_shoe_kind(encoder, vectors[0])

    best_index: int | None = None
    best_score = float("-inf")
    crop_scores: list[tuple[CropBox, float]] = []
    gate_scores = {"footwear": 0.0, "irrelevant": 0.0, "catalog": float("-inf")}
    for index, vector in enumerate(vectors):
        relevance, scores = encoder.classify_relevance(vector)
        catalog_score = _catalog_score(catalog, vector)
        crop_scores.append((boxes[index], catalog_score))
        if not _is_identifiable_shoe(relevance, catalog_score, boxes[index], rgb):
            continue
        if best_index is None or _prefer_crop(catalog_score, boxes[index], best_score, boxes[best_index]):
            best_index = index
            best_score = catalog_score
            gate_scores = {**scores, "catalog": catalog_score}

    if has_strong_order_evidence(text, order_fields):
        return _document_result(
            boxes=boxes,
            crop_scores=crop_scores,
            scores=scene_scores,
            embedding_model=encoder.embedding_model,
            image_kind=IMAGE_KIND_ORDER,
            order=order_payload(order_fields),
        )

    if has_strong_delivery_evidence(text, delivery_fields):
        return _document_result(
            boxes=boxes,
            crop_scores=crop_scores,
            scores=scene_scores,
            embedding_model=encoder.embedding_model,
            image_kind=IMAGE_KIND_DELIVERY,
            delivery=delivery_payload(delivery_fields),
        )

    if best_index is not None:
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
            delivery=None,
        )

    kind = classify_document_kind(text) or scene
    if kind == IMAGE_KIND_ORDER:
        return _document_result(
            boxes=boxes,
            crop_scores=crop_scores,
            scores=scene_scores,
            embedding_model=encoder.embedding_model,
            image_kind=IMAGE_KIND_ORDER,
            order=order_payload(order_fields),
        )

    if kind == IMAGE_KIND_DELIVERY:
        return _document_result(
            boxes=boxes,
            crop_scores=crop_scores,
            scores=scene_scores,
            embedding_model=encoder.embedding_model,
            image_kind=IMAGE_KIND_DELIVERY,
            delivery=delivery_payload(delivery_fields),
        )

    scene_kind, confident_scores = _scene_kind(encoder, vectors[0])
    if _confident_garbage(scene_kind, confident_scores):
        chosen = boxes[0]
        isolated = isolation_applied(chosen, rgb)
        return ImageMatchResult(
            match=None,
            candidates=[],
            crop=chosen,
            shoe_isolated=isolated,
            relevance=RELEVANCE_IRRELEVANT,
            scores=confident_scores,
            embedding_model=encoder.embedding_model,
            crop_scores=crop_scores,
            image_kind=IMAGE_KIND_GARBAGE,
            order=None,
            delivery=None,
        )

    fallback_index = _fallback_crop_index(crop_scores, boxes)
    chosen = boxes[fallback_index]
    query = vectors[fallback_index]
    isolated = isolation_applied(chosen, rgb)
    candidates = catalog.search_models(query, top=top)
    catalog_score = crop_scores[fallback_index][1]
    scores = dict(confident_scores)
    scores["catalog"] = float(catalog_score) if np.isfinite(catalog_score) else 0.0
    return ImageMatchResult(
        match=candidates[0] if candidates else None,
        candidates=candidates,
        crop=chosen,
        shoe_isolated=isolated,
        relevance=RELEVANCE_IRRELEVANT,
        scores=scores,
        embedding_model=encoder.embedding_model,
        crop_scores=crop_scores,
        image_kind=IMAGE_KIND_UNKNOWN,
        order=None,
        delivery=None,
        unknown_fallback=True,
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
    kind = result.image_kind or (IMAGE_KIND_SHOE if result.relevance == RELEVANCE_FOOTWEAR else IMAGE_KIND_GARBAGE)
    fallback = bool(result.unknown_fallback or kind == IMAGE_KIND_UNKNOWN)
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
    if (footwear or fallback) and candidates:
        verifier = build_verifier_request(
            candidates,
            catalog_rows=list(catalog.rows) if catalog is not None else [],
            verify_top=verify_top,
        )
    if kind == IMAGE_KIND_ORDER:
        status = "order"
    elif kind == IMAGE_KIND_DELIVERY:
        status = "delivery_notice"
    elif kind == IMAGE_KIND_GARBAGE:
        status = "garbage"
    elif kind == IMAGE_KIND_UNKNOWN:
        status = "needs_verification" if candidates else "uncertain"
    elif footwear:
        status = "needs_verification"
    else:
        status = "garbage"
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
        "unknown_fallback": fallback,
        "order": result.order,
        "delivery": result.delivery,
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
