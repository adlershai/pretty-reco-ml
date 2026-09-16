"""Fine-grained product identity verification. OpenAI is called by pretty-crm-api."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

from embeddings.isolate import CropBox, crop_image
from evaluation.image_match import IMAGE_TYPES, catalog_image_url

DEFAULT_VERIFY_TOP = 10
VERIFY_TOP_ENV = "MATCH_VERIFY_TOP"
CROP_MAX_EDGE = 768
REQUEST_TYPE = "watiImageIdentity"
SHAPE_EXCLUDE = 0.5
PATTERN_EXCLUDE = 0.5
MATERIAL_EXCLUDE = 0.55
DETAILS_EXCLUDE = 0.55

DEVELOPER_PROMPT = """You verify Pretty Ballerinas catalog identity.

The first image is a customer photo (shoe isolated). Each later group is one catalog
candidate with every available packshot view (main, pers, side).

Question: is the customer shoe THE SAME catalog SKU as that candidate?
Not similar style, not a close sister model, not similar color. Exact product identity.

Inspect distinctive SKU-level evidence before global similarity:
1. decorative hardware and stones — shape (square/diamond vs round), setting, size, spacing, repetition
2. strap construction (width, count, crossing, attachments)
3. bow construction and placement
4. material / surface (suede vs smooth leather vs patent vs velvet vs textile)
5. seams, piping, trims, toe cap
6. toe / vamp geometry, heel, proportions
7. pattern type, geometry, repetition, scale
8. where colors sit on the shoe — last, and never enough to override a detail contradiction

A clear contradiction in stone/hardware geometry, material, bow, strap, toe/vamp, or construction
means DIFFERENT even when silhouette, pink lining, and overall black/ballet look match.
Close sister models (example: studded Mary-Jane 49452 vs 54101) are DIFFERENT.

If two candidates could both be the same model, mark extras uncertain rather than forcing same.

Return one object per candidate, using the exact model codes from the labels."""


JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "model",
                    "verdict",
                    "shape",
                    "pattern",
                    "material",
                    "details",
                    "color_placement",
                    "contradictions",
                    "reason",
                ],
                "properties": {
                    "model": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["same", "different", "uncertain"],
                    },
                    "shape": {"type": "number"},
                    "pattern": {"type": "number"},
                    "material": {"type": "number"},
                    "details": {"type": "number"},
                    "color_placement": {"type": "number"},
                    "contradictions": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class VerifierDecision:
    status: str
    model: str | None
    confidence: float | None
    reason: str
    verification: list[dict[str, Any]]


def verify_top_value(explicit: int | None = None) -> int:
    """How many SigLIP candidates CRM should send to the vision verifier."""
    if explicit is not None:
        return max(1, min(50, int(explicit)))
    raw = os.environ.get(VERIFY_TOP_ENV, "").strip()
    if not raw:
        return DEFAULT_VERIFY_TOP
    try:
        return max(1, min(50, int(raw)))
    except ValueError:
        return DEFAULT_VERIFY_TOP


def crop_jpeg_base64(image: Image.Image, box: CropBox, *, max_edge: int = CROP_MAX_EDGE) -> str:
    import base64

    cropped = crop_image(image, box)
    width, height = cropped.size
    longest = max(width, height)
    if longest > max_edge and longest > 0:
        scale = max_edge / float(longest)
        cropped = cropped.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    cropped.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def views_for_model(catalog_rows: list[Any], model: str) -> list[str]:
    found: list[str] = []
    for row in catalog_rows:
        view = str(getattr(row, "image_type", "") or "")
        if getattr(row, "model", None) == model and view and view not in found:
            found.append(view)
    order = {name: index for index, name in enumerate(IMAGE_TYPES)}
    return sorted(found, key=lambda view: order.get(view, 99))


def build_verifier_request(
    candidates: list[dict[str, Any]],
    *,
    catalog_rows: list[Any],
    verify_top: int | None = None,
) -> dict[str, Any]:
    """Prompt + schema + image slots. CRM fills bytes and calls OpenAI."""
    limit = verify_top_value(verify_top)
    chosen = list(candidates[:limit])
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": "Customer image (isolated shoe):"},
        {"type": "input_image", "slot": "customer"},
    ]
    images: list[dict[str, Any]] = [{"slot": "customer", "kind": "crop"}]
    for hit in chosen:
        model = str(hit.get("model") or "")
        views = views_for_model(catalog_rows, model) or ["main"]
        content.append(
            {
                "type": "input_text",
                "text": f"Candidate {model}. Views follow. Same catalog model as the customer shoe?",
            }
        )
        for view in views:
            slot = f"{model}_{view}"
            content.append({"type": "input_text", "text": f"{model} / {view}"})
            content.append({"type": "input_image", "slot": slot})
            images.append(
                {
                    "slot": slot,
                    "kind": "packshot",
                    "model": model,
                    "view": view,
                    "url": catalog_image_url(model, view),
                }
            )
    return {
        "requestType": REQUEST_TYPE,
        "verify_top": limit,
        "jsonSchema": JSON_SCHEMA,
        "prompt": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": content},
        ],
        "images": images,
    }


def _clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    verdict = str(raw.get("verdict") or "uncertain").strip().lower()
    if verdict not in {"same", "different", "uncertain"}:
        verdict = "uncertain"
    contradictions = raw.get("contradictions") or []
    if not isinstance(contradictions, list):
        contradictions = [str(contradictions)]
    return {
        "model": str(raw.get("model") or "").strip(),
        "verdict": verdict,
        "shape": _clamp_unit(raw.get("shape")),
        "pattern": _clamp_unit(raw.get("pattern")),
        "material": _clamp_unit(raw.get("material")),
        "details": _clamp_unit(raw.get("details")),
        "color_placement": _clamp_unit(raw.get("color_placement")),
        "contradictions": [str(item) for item in contradictions if str(item).strip()],
        "reason": str(raw.get("reason") or ""),
    }


def _content_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        text = item.get("text")
        if isinstance(text, str):
            return text
    return ""


def parse_openai_output(openai_output: Any) -> dict[str, Any] | None:
    """Accept parsed JSON, Responses API output array, or a failed CRM wrapper."""
    if openai_output is None:
        return None
    if isinstance(openai_output, dict):
        if openai_output.get("result") == "failed":
            return None
        if isinstance(openai_output.get("candidates"), list):
            return openai_output
        for key in ("text", "output_text"):
            value = openai_output.get(key)
            if isinstance(value, str) and value.strip():
                return parse_openai_output(value)
    if isinstance(openai_output, list):
        chunks: list[str] = []
        for item in openai_output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                chunks.extend(_content_text(part) for part in content)
            chunks.append(_content_text(item))
        return parse_openai_output("".join(chunks))
    if isinstance(openai_output, str):
        text = openai_output.strip()
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _exclusionary(row: dict[str, Any]) -> bool:
    distinctive_low = (
        row["shape"] < SHAPE_EXCLUDE
        or row["pattern"] < PATTERN_EXCLUDE
        or row["material"] < MATERIAL_EXCLUDE
        or row["details"] < DETAILS_EXCLUDE
    )
    if row["contradictions"] and distinctive_low:
        return True
    if row["verdict"] == "same" and (
        row["material"] < MATERIAL_EXCLUDE or row["details"] < DETAILS_EXCLUDE
    ):
        return True
    return False


def _confidence(row: dict[str, Any]) -> float:
    return min(row["shape"], row["pattern"], row["details"])


def decide_identity(
    openai_output: Any,
    retrieval_candidates: list[dict[str, Any]],
) -> VerifierDecision:
    """Turn verifier JSON into MATCH or UNCERTAIN. SigLIP cosine is not used."""
    parsed = parse_openai_output(openai_output)
    if parsed is None:
        return VerifierDecision(
            status="uncertain",
            model=None,
            confidence=None,
            reason="verifier_failed",
            verification=[],
        )
    raw_rows = parsed.get("candidates")
    if not isinstance(raw_rows, list) or not raw_rows:
        return VerifierDecision(
            status="uncertain",
            model=None,
            confidence=None,
            reason="verifier_empty",
            verification=[],
        )
    rows = [_normalize_row(item) for item in raw_rows if isinstance(item, dict)]
    known = {str(hit.get("model") or "") for hit in retrieval_candidates}
    rows = [row for row in rows if row["model"] and (not known or row["model"] in known)]
    sames = [
        row
        for row in rows
        if row["verdict"] == "same" and not _exclusionary(row)
    ]
    if len(sames) == 1:
        winner = sames[0]
        return VerifierDecision(
            status="match",
            model=winner["model"],
            confidence=_confidence(winner),
            reason=winner["reason"] or "unique_same_model",
            verification=rows,
        )
    if len(sames) > 1:
        reason = "ambiguous_same_models:" + ",".join(row["model"] for row in sames)
    else:
        reason = "no_unique_same_model"
    return VerifierDecision(
        status="uncertain",
        model=None,
        confidence=None,
        reason=reason,
        verification=rows,
    )
