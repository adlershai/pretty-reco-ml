"""Parse tracking / shipment fields from a delivery-notice screenshot."""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

from embeddings.order_extract import ocr_image_text

TRACKING_LABEL_RE = re.compile(
    r'(?:מספר\s*משלוח|tracking(?:\s*(?:number|no\.?))?|shipment(?:\s*number)?)\s*[:#\s]*(\d{8,14})',
    re.IGNORECASE,
)
ON_THE_WAY_RE = re.compile(
    r'on the way|בדרך אליך|בדרך|packed|ארוז',
    re.IGNORECASE,
)
TRACKING_BARE_RE = re.compile(r'(?<!\d)(\d{9,12})(?!\d)')


def parse_delivery_fields(text: str) -> dict[str, str | None]:
    raw = str(text or '')
    tracking = None
    labeled = TRACKING_LABEL_RE.search(raw)
    if labeled:
        tracking = labeled.group(1)
    elif ON_THE_WAY_RE.search(raw):
        bare = TRACKING_BARE_RE.search(raw)
        if bare:
            tracking = bare.group(1)

    status = None
    if ON_THE_WAY_RE.search(raw):
        status = 'on_the_way'

    return {
        'tracking_number': tracking,
        'status': status,
    }


def extract_delivery(image: Image.Image, *, text: str | None = None) -> dict[str, str | None]:
    raw = text if text is not None else ocr_image_text(image)
    return parse_delivery_fields(raw)


def delivery_payload(fields: dict[str, Any] | None) -> dict[str, str | None] | None:
    if not fields:
        return None
    tracking = fields.get('tracking_number') or None
    status = fields.get('status') or None
    if not tracking and not status:
        return {'tracking_number': None, 'status': None}
    return {
        'tracking_number': str(tracking) if tracking else None,
        'status': str(status) if status else None,
    }
