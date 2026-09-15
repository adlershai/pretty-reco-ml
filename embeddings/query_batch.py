"""Batch query embeddings from a JSON file of base64 images. Finite CLI; no HTTP.

Usage:
    python -m embeddings.query_batch --in images.json --out query.json

Input: { "images": [ { "id": 1, "image_base64": "..." } ] }
Output: { "results": [ { "id": 1, ...QueryEmbeddingResponse } ], "errors": [...] }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from embeddings.query_image import QueryImageError, decode_image_base64, run_query
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch query-image embeddings")
    parser.add_argument("--in", dest="input_json", required=True)
    parser.add_argument("--out", dest="output_json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    items = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        print("expected {\"images\": [...]}", file=sys.stderr)
        return 1

    encoder = VisionEncoder()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            errors.append({"id": None, "error": "INVALID_ENTRY"})
            continue
        item_id = item.get("id")
        try:
            image_bytes = decode_image_base64(str(item.get("image_base64") or ""))
            out = run_query(image_bytes, encoder)
            out["id"] = item_id
            results.append(out)
        except (QueryImageError, ValueError) as exc:
            errors.append({"id": item_id, "error": str(exc) or "INVALID_IMAGE"})
        except Exception as exc:
            errors.append({"id": item_id, "error": str(exc)})

    Path(args.output_json).write_text(
        json.dumps({"results": results, "errors": errors}, ensure_ascii=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
