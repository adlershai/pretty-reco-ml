"""Semantic document type from OCR text. Empty text means no decision."""

from __future__ import annotations

import re

from embeddings.relevance import IMAGE_KIND_DELIVERY, IMAGE_KIND_ORDER

ORDER_LANG_RE = re.compile(
    r'thank you for your order|new order|order confirmation|your order number|מספר הזמנה|הזמנתך',
    re.IGNORECASE,
)
DELIVERY_LANG_RE = re.compile(
    r'on the way|מספר משלוח|מעקב משלוח|track shipment',
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
