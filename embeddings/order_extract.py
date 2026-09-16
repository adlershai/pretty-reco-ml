"""Parse order number, model, and size from OCR text on an order screenshot."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image

ORDER_NUMBER_RE = re.compile(
    r'(?:order\s*number|הזמנה[#\s:]*)\s*#?\s*(\d{5,6})\b',
    re.IGNORECASE,
)
HASH_ORDER_RE = re.compile(r'#(\d{5,6})\b')
MODEL_RE = re.compile(r'(?<!\d)(\d{5}_\d{3})(?:\d{2,3})?(?!\d)')
SIZE_RE = re.compile(
    r'(?:[-–]|size|מידה)\s*(3[5-9](?:\.5)?|4[0-2](?:\.5)?)',
    re.IGNORECASE,
)


def parse_order_fields(text: str) -> dict[str, str | None]:
    raw = str(text or '')
    order_number = None
    numbered = ORDER_NUMBER_RE.search(raw)
    if numbered:
        order_number = numbered.group(1)
    else:
        hashed = HASH_ORDER_RE.search(raw)
        if hashed:
            order_number = hashed.group(1)

    model = None
    modeled = MODEL_RE.search(raw)
    if modeled:
        model = modeled.group(1)

    size = None
    sized = SIZE_RE.search(raw)
    if sized:
        size = sized.group(1)

    return {
        'order_number': order_number,
        'model': model,
        'size': size,
    }


def ocr_image_text(image: Image.Image) -> str:
    """Best-effort Tesseract OCR. Empty when tesseract is not installed."""
    binary = shutil.which('tesseract')
    if not binary:
        return ''
    rgb = image.convert('RGB')
    handle, path = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    try:
        rgb.save(path, format='PNG')
        completed = subprocess.run(
            [binary, path, 'stdout', '--psm', '6'],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if completed.returncode != 0:
        return ''
    return str(completed.stdout or '')


def extract_order(image: Image.Image, *, text: str | None = None) -> dict[str, str | None]:
    raw = text if text is not None else ocr_image_text(image)
    return parse_order_fields(raw)


def order_payload(fields: dict[str, Any] | None) -> dict[str, str | None] | None:
    if not fields:
        return None
    order_number = fields.get('order_number') or None
    model = fields.get('model') or None
    size = fields.get('size') or None
    if not order_number and not model and not size:
        return {'order_number': None, 'model': None, 'size': None}
    return {
        'order_number': str(order_number) if order_number else None,
        'model': str(model) if model else None,
        'size': str(size) if size else None,
    }
