"""Parse order number, model, and size from OCR text on an order screenshot."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image, ImageOps

ORDER_NUMBER_RE = re.compile(
    r'(?:new\s*order|order\s*(?:number|confirmation)|אישור\s*הזמנה|מספר\s*הזמנה|הזמנה)'
    r'\s*[:#\s]*([A-Z]{0,3}\d{5,12})\b',
    re.IGNORECASE,
)
HASH_ORDER_RE = re.compile(r'#(\d{5,6})\b')
CS_ORDER_RE = re.compile(r'\b(CS\d{8,12})\b', re.IGNORECASE)
BS_ORDER_RE = re.compile(r'\b(BS\d{8,12})\b', re.IGNORECASE)
OCR_APP_PREFIX_RE = re.compile(r'\b([CB][5S]|8S)(\d{8,12})\b', re.IGNORECASE)
MODEL_RE = re.compile(r'(?<!\d)(\d{5}_\d{3})(?:\d{2,3})?(?!\d)')
SIZE_RE = re.compile(
    r'(?:[-–—−]|size|מידה)\s*(3[5-9]|4[0-2])([.,][05])?',
    re.IGNORECASE,
)
SIZE_DECIMAL_RE = re.compile(r'(?<!\d)((?:3[5-9]|4[0-2])[.,][05])(?!\d)')


def recover_app_order_number(text: str, captured: str | None) -> str | None:
    """Prefer a readable BS/CS token; repair OCR that turns S into 5 when the letter prefix remains."""
    raw = str(text or '')
    tokens = [match.group(1).upper() for match in CS_ORDER_RE.finditer(raw)]
    tokens.extend(match.group(1).upper() for match in BS_ORDER_RE.finditer(raw))
    if tokens:
        return max(tokens, key=len)
    fuzzy = OCR_APP_PREFIX_RE.search(raw)
    if fuzzy:
        prefix = fuzzy.group(1).upper()
        digits = fuzzy.group(2)
        if prefix in {'CS', 'C5'}:
            return 'CS' + digits
        return 'BS' + digits
    cand = str(captured or '').strip().upper()
    if re.fullmatch(r'(?:BS|CS)\d{8,12}', cand):
        return cand
    return captured


def _normalize_size(value: str) -> str:
    return str(value or '').replace(',', '.')


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
        else:
            cs_order = CS_ORDER_RE.search(raw) or BS_ORDER_RE.search(raw)
            if cs_order:
                order_number = cs_order.group(1).upper()
    order_number = recover_app_order_number(raw, order_number)

    model = None
    modeled = MODEL_RE.search(raw)
    if modeled:
        model = modeled.group(1)

    size = None
    sized = SIZE_RE.search(raw)
    if sized:
        size = sized.group(1)
        if sized.group(2):
            size = _normalize_size(size + sized.group(2))
    else:
        bare = SIZE_DECIMAL_RE.search(raw)
        if bare:
            size = _normalize_size(bare.group(1))

    return {
        'order_number': order_number,
        'model': model,
        'size': size,
    }


def _tesseract_langs(binary: str) -> list[str]:
    listed = subprocess.run(
        [binary, '--list-langs'],
        check=False,
        capture_output=True,
        text=True,
    )
    langs = str(listed.stdout or '').lower()
    if 'heb' in langs:
        return ['eng', 'heb+eng']
    return ['eng']


def _run_tesseract(
    binary: str,
    path: str,
    lang: str,
    psm: str = '6',
    extra: list[str] | None = None,
) -> str:
    command = [binary, path, 'stdout', '-l', lang, '--psm', psm]
    if extra:
        command.extend(extra)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return ''
    return str(completed.stdout or '')


def _ocr_latin_title(binary: str, rgb: Image.Image) -> str:
    """High-contrast English pass on screenshot bands that carry CS/BS ids."""
    width, height = rgb.size
    whitelist = ['-c', 'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']
    chunks: list[str] = []
    for top_frac, bot_frac in ((0.0, 0.22), (0.28, 0.52)):
        box = (0, int(height * top_frac), width, max(int(height * bot_frac), int(height * top_frac) + 80))
        band = rgb.crop(box)
        band = ImageOps.autocontrast(band.convert('L')).convert('RGB')
        band = band.resize((band.width * 3, band.height * 3), Image.LANCZOS)
        handle, path = tempfile.mkstemp(suffix='.png')
        os.close(handle)
        try:
            band.save(path, format='PNG')
            for psm in ('6', '7', '11'):
                text = _run_tesseract(binary, path, 'eng', psm, whitelist)
                if text.strip():
                    chunks.append(text)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return '\n'.join(chunks)


def ocr_image_text(image: Image.Image) -> str:
    """Best-effort Tesseract OCR. Empty when tesseract is not installed.

    English is run first so Latin CS/BS order numbers survive Hebrew screenshots.
    """
    binary = shutil.which('tesseract')
    if not binary:
        return ''
    rgb = image.convert('RGB')
    if max(rgb.size) < 1600:
        rgb = rgb.resize((rgb.width * 2, rgb.height * 2), Image.LANCZOS)
    handle, path = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    chunks: list[str] = []
    try:
        rgb.save(path, format='PNG')
        for lang in _tesseract_langs(binary):
            text = _run_tesseract(binary, path, lang)
            if text.strip():
                chunks.append(text)
        latin = _ocr_latin_title(binary, rgb)
        if latin.strip():
            chunks.insert(0, latin)
        if not chunks:
            completed = subprocess.run(
                [binary, path, 'stdout', '--psm', '6'],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                chunks.append(str(completed.stdout or ''))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return '\n'.join(chunks)


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
