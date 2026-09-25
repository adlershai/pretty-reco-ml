"""Semantic document type from OCR text. Empty text means no decision."""

from __future__ import annotations

import re
from typing import Any

from embeddings.relevance import IMAGE_KIND_DELIVERY, IMAGE_KIND_ORDER

ORDER_LANG_RE = re.compile(
    r'thank you for your order|new order|order confirmation|'
    r'your order number|אישור\s*הזמנה|מספר\s*הזמנה|הזמנתך(?:\s*התקבלה)?',
    re.IGNORECASE,
)
DELIVERY_LANG_RE = re.compile(
    r'on the way|מספר משלוח|מעקב משלוח|track shipment',
    re.IGNORECASE,
)
CS_ORDER_RE = re.compile(r'^CS\d{8,12}$', re.IGNORECASE)
LABELED_ORDER_RE = re.compile(
    r'(?:new\s*order|order\s*(?:number|confirmation)|אישור\s*הזמנה|מספר\s*הזמנה|הזמנה)'
    r'\s*[:#\s]*([A-Z]{0,3}\d{5,12})\b',
    re.IGNORECASE,
)


def classify_document_kind(text: str) -> str | None:
    """Return order or delivery_notice from wording, or None if text is unhelpful.

    A tracking/order number inside a shipment notice must not make it ORDER.
    """
    raw = str(text or '')
    if not raw.strip():
        return None
    delivery = bool(DELIVERY_LANG_RE.search(raw))
    order = bool(ORDER_LANG_RE.search(raw))
    if delivery:
        return IMAGE_KIND_DELIVERY
    if order:
        return IMAGE_KIND_ORDER
    return None


def has_strong_order_evidence(text: str, fields: dict[str, Any] | None) -> bool:
    """Whole-image ORDER intent: terminology plus an extractable order number.

    Weak cosine-to-order must not steal a shoe photo. A recognizable catalog
    shoe inside a confirmation must not steal a genuine order.
    """
    number = str((fields or {}).get('order_number') or '').strip()
    if not number:
        return False
    raw = str(text or '')
    if classify_document_kind(raw) == IMAGE_KIND_ORDER:
        return True
    if LABELED_ORDER_RE.search(raw):
        return True
    return bool(CS_ORDER_RE.fullmatch(number))


def has_strong_delivery_evidence(text: str, fields: dict[str, Any] | None) -> bool:
    tracking = str((fields or {}).get('tracking_number') or '').strip()
    if not tracking:
        return False
    return classify_document_kind(text) == IMAGE_KIND_DELIVERY
