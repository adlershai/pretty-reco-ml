"""Read new-model codes from a headerless CSV (column A)."""

from __future__ import annotations

from pathlib import Path


ANONYMOUS_CUSTOMER_ID = "99999999"


def read_model_codes(csv_path: Path) -> list[str]:
    text = csv_path.read_text(encoding="utf-8-sig")
    codes: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        code = line.split(",", 1)[0].strip().strip('"')
        if not code or code.lower() in {"model", "sku", "model_code"}:
            continue
        if code not in seen:
            seen.add(code)
            codes.append(code)
    if not codes:
        raise ValueError(f"no model codes in {csv_path}")
    return codes
