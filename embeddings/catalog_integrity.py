"""Audit and refresh catalog image embeddings when source bytes change.

Usage:
    python -m embeddings.catalog_integrity --audit --models 52160_001,52792_006
    python -m embeddings.catalog_integrity --refresh-stale --models 52160_001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

import requests

from data.config import REPO_ROOT, load_dotenv
from data.db_client import DbApiClient
from embeddings.catalog_index import load_catalog_rows
from embeddings.vision_encoder import SIGLIP_MODEL_ID, VisionEncoder
from embeddings.worker import configure_logging, decode_image, download_image_bytes, sha256_hex
from evaluation.image_match import CatalogImage, catalog_image_url

logger = logging.getLogger("embeddings.catalog_integrity")

TABLE = "reco_model_image_embeddings"
UPDATE_FIELDS = ["embedding_dimension", "embedding", "image_hash"]
INSERT_FIELDS = [
    "model_id",
    "image_type",
    "embedding_model",
    "embedding_dimension",
    "embedding",
    "image_hash",
]


@dataclass
class AuditItem:
    model: str
    image_type: str
    row_id: int | None
    model_id: int | None
    source_url: str
    stored_hash: str | None
    current_hash: str | None
    stale: bool
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit/refresh catalog image embeddings")
    parser.add_argument("--audit", action="store_true", help="Compare CDN hashes to stored image_hash")
    parser.add_argument("--refresh-stale", action="store_true", help="Re-encode and write stale rows")
    parser.add_argument("--models", help="Comma-separated model codes (default: all cached/loaded rows)")
    parser.add_argument("--refresh-catalog", action="store_true", help="Reload catalog from DB, ignore cache")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to audit (0 = no limit)")
    return parser.parse_args(argv)


def _selected_rows(rows: list[CatalogImage], models: str | None, limit: int) -> list[CatalogImage]:
    wanted = {item.strip() for item in str(models or "").split(",") if item.strip()}
    selected = [row for row in rows if not wanted or row.model in wanted]
    if limit > 0:
        selected = selected[:limit]
    return selected


def audit_row(row: CatalogImage, session: requests.Session) -> AuditItem:
    url = row.source_url or catalog_image_url(row.model, row.image_type)
    try:
        image_bytes = download_image_bytes(url, session)
    except Exception as exc:
        return AuditItem(
            model=row.model,
            image_type=row.image_type,
            row_id=row.row_id,
            model_id=row.model_id,
            source_url=url,
            stored_hash=row.image_hash,
            current_hash=None,
            stale=True,
            error=str(exc),
        )
    current = sha256_hex(image_bytes)
    stored = row.image_hash
    return AuditItem(
        model=row.model,
        image_type=row.image_type,
        row_id=row.row_id,
        model_id=row.model_id,
        source_url=url,
        stored_hash=stored,
        current_hash=current,
        stale=bool(stored and current != stored) or not stored,
        error=None,
    )


def write_embedding(client: DbApiClient, row: CatalogImage, vector: Any, image_hash: str) -> None:
    embedding = [float(value) for value in vector.tolist()]
    if row.row_id:
        client.update(
            TABLE,
            int(row.row_id),
            UPDATE_FIELDS,
            [int(vector.shape[0]), json.dumps(embedding), image_hash],
        )
        return
    if row.model_id is None:
        raise RuntimeError(f"cannot upsert {row.key}: missing model_id")
    client.insert(
        TABLE,
        INSERT_FIELDS,
        [
            int(row.model_id),
            row.image_type,
            row.embedding_model or SIGLIP_MODEL_ID,
            int(vector.shape[0]),
            json.dumps(embedding),
            image_hash,
        ],
        on_duplicate_update=UPDATE_FIELDS,
    )


def refresh_row(
    row: CatalogImage,
    session: requests.Session,
    encoder: VisionEncoder,
    client: DbApiClient,
) -> AuditItem:
    url = row.source_url or catalog_image_url(row.model, row.image_type)
    image_bytes = download_image_bytes(url, session)
    image_hash = sha256_hex(image_bytes)
    vector = encoder.encode(decode_image(image_bytes))
    write_embedding(client, row, vector, image_hash)
    return AuditItem(
        model=row.model,
        image_type=row.image_type,
        row_id=row.row_id,
        model_id=row.model_id,
        source_url=url,
        stored_hash=row.image_hash,
        current_hash=image_hash,
        stale=False,
        error=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    load_dotenv(REPO_ROOT / ".env")
    if not args.audit and not args.refresh_stale:
        logger.error("provide --audit and/or --refresh-stale")
        return 1

    rows = load_catalog_rows(refresh=args.refresh_catalog)
    selected = _selected_rows(rows, args.models, args.limit)
    if not selected:
        logger.error("no catalog rows selected")
        return 1

    session = requests.Session()
    audits = [audit_row(row, session) for row in selected]
    stale_items = [item for item in audits if item.stale]
    print(f"audited={len(audits)} stale={len(stale_items)}")
    for item in audits:
        status = "ERROR" if item.error else ("STALE" if item.stale else "ok")
        print(
            f"  {status} {item.model}/{item.image_type} stored={item.stored_hash or '-'} "
            f"current={item.current_hash or '-'} {item.error or ''}"
        )

    if args.refresh_stale and stale_items:
        encoder = VisionEncoder()
        client = DbApiClient()
        by_key = {(row.model, row.image_type): row for row in selected}
        for item in stale_items:
            row = by_key.get((item.model, item.image_type))
            if row is None:
                continue
            try:
                refreshed = refresh_row(row, session, encoder, client)
                print(f"  refreshed {refreshed.model}/{refreshed.image_type} hash={refreshed.current_hash}")
            except Exception as exc:
                logger.exception("refresh failed for %s/%s", item.model, item.image_type)
                print(f"  refresh-failed {item.model}/{item.image_type} {exc}")
                return 1
        from embeddings.catalog_index import load_catalog_rows as reload

        reload(refresh=True)
        print("catalog cache refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
