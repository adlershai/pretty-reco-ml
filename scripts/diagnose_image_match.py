"""Diagnose visual catalog matching for a customer-supplied shoe image.

Usage:
    python scripts/diagnose_image_match.py \\
      --image-url "<url>" \\
      --expected-model "52792_006" \\
      --compare-model "52160_001" \\
      --top 20

    python scripts/diagnose_image_match.py --self-test 52792_006/side --top 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import requests

from data.config import REPO_ROOT, load_dotenv
from embeddings.catalog_index import load_catalog_rows
from embeddings.query_image import run_query
from embeddings.vision_encoder import SIGLIP_MODEL_ID, VisionEncoder
from embeddings.worker import decode_image, download_image_bytes, sha256_hex
from evaluation.image_match import (
    IMAGE_TYPES,
    CatalogImage,
    ImageHit,
    IntegrityReport,
    ModelHit,
    aggregate_models_max,
    catalog_image_url,
    classify_failure,
    find_image_hit,
    format_image_table,
    format_model_table,
    hits_for_model,
    inspect_catalog_integrity,
    inspect_vector,
    rank_catalog_images,
)

logger = logging.getLogger("diagnose_image_match")

PRODUCTION_NN = (
    "pretty-crm-api processWatiImageMatch: download WATI bytes -> S3 -> "
    "POST /match/image (isolate shoe, VisionEncoder 768D L2-normalized SigLIP, "
    "MAX view cosine over reco_model_image_embeddings) -> "
    "write wati_image_matches if score >= WATI_IMAGE_MATCH_MIN_SCORE (default 0.2)."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SigLIP catalog image matching")
    parser.add_argument("--image-url", help="Query image URL")
    parser.add_argument("--image-path", help="Local query image path")
    parser.add_argument("--expected-model", help="Model that should match the query")
    parser.add_argument("--compare-model", help="Model currently returned in production")
    parser.add_argument("--top", type=int, default=20, help="Top-N to print (default 20)")
    parser.add_argument(
        "--self-test",
        action="append",
        default=[],
        metavar="MODEL/TYPE",
        help="Re-encode a catalog image and search. Repeatable, e.g. 52792_006/side",
    )
    parser.add_argument(
        "--catalog-source",
        choices=("table", "view"),
        default="table",
        help="Load per-image vectors from reco_model_image_embeddings (table) or the training view",
    )
    parser.add_argument("--refresh-catalog", action="store_true", help="Ignore local catalog cache")
    parser.add_argument("--json-out", help="Optional path to write the full ranking JSON")
    parser.add_argument(
        "--embedding-model",
        default=SIGLIP_MODEL_ID,
        help=f"Catalog embedding_model filter (default {SIGLIP_MODEL_ID})",
    )
    return parser.parse_args(argv)


def parse_self_test(spec: str) -> tuple[str, str]:
    text = str(spec or "").strip()
    if "/" not in text:
        raise ValueError(f"self-test must be MODEL/TYPE, got {spec!r}")
    model, image_type = text.split("/", 1)
    model = model.strip()
    image_type = image_type.strip().lower()
    if not model or image_type not in IMAGE_TYPES:
        raise ValueError(f"self-test must be MODEL/main|pers|side, got {spec!r}")
    return model, image_type


def load_query_bytes(args: argparse.Namespace, session: requests.Session) -> tuple[bytes, str]:
    if args.image_path:
        path = Path(args.image_path)
        return path.read_bytes(), str(path)
    if args.image_url:
        return download_image_bytes(str(args.image_url), session), str(args.image_url)
    raise ValueError("provide --image-url or --image-path")


def encode_query(image_bytes: bytes, encoder: VisionEncoder) -> dict[str, Any]:
    result = run_query(image_bytes, encoder)
    vector = np.asarray(result["embedding"], dtype=np.float32)
    result["embedding_array"] = vector
    result["vector_norm"] = float(np.linalg.norm(vector))
    result["image_hash"] = sha256_hex(image_bytes)
    return result


def rows_for_model(rows: list[CatalogImage], model: str) -> list[CatalogImage]:
    needle = str(model or "").strip()
    return [row for row in rows if row.model == needle]


def print_model_views(label: str, model: str, hits: list[ImageHit], rows: list[CatalogImage]) -> None:
    print(f"\n{label} {model}")
    by_type = {row.image_type: row for row in rows}
    hit_by_type = {hit.image_type: hit for hit in hits}
    if not by_type:
        print("  MISSING: no catalog images for this model")
        return
    for image_type in IMAGE_TYPES:
        row = by_type.get(image_type)
        hit = hit_by_type.get(image_type)
        if row is None:
            print(f"  {image_type}: MISSING")
            continue
        similarity = f"{hit.similarity:.6f}" if hit else "n/a"
        rank = str(hit.rank) if hit else "n/a"
        print(
            f"  {image_type}: similarity={similarity} global_rank={rank} "
            f"dim={row.embedding.size} norm={row.vector_norm:.6f} "
            f"model={row.embedding_model or '?'} url={row.source_url}"
        )


def print_integrity(title: str, vector: np.ndarray, embedding_model: str, extra: str = "") -> None:
    flags = inspect_vector(vector, embedding_model=embedding_model)
    status = "ok" if not flags else "; ".join(flags)
    print(f"{title}: dim={vector.size} norm={float(np.linalg.norm(vector)):.6f} {status}{extra}")


def print_catalog_integrity(report: IntegrityReport, focus_models: list[str], rows: list[CatalogImage]) -> None:
    print("\nCatalog integrity")
    print(f"  images={report.image_count} models={report.model_count}")
    print(f"  embedding_model={report.embedding_model or '?'} (expected {report.expected_model})")
    print(f"  dimension={report.dimension or '?'} (expected {report.expected_dimension})")
    focus = {model for model in focus_models if model}
    relevant = [issue for issue in report.issues if not focus or issue.model in focus]
    other = len(report.issues) - len(relevant)
    if not report.issues:
        print("  no issues")
    else:
        shown = relevant or report.issues[:20]
        for issue in shown[:40]:
            print(f"  {issue.code}: {issue.model} / {issue.image_type} -- {issue.detail}")
        remaining = len(report.issues) - len(shown)
        if remaining > 0:
            print(f"  ... {remaining} more issue(s)")
        elif other > 0 and relevant:
            print(f"  ({other} issue(s) on other models)")
    for model in focus_models:
        if not model:
            continue
        for row in rows_for_model(rows, model):
            print(
                f"  {row.key}: url={row.source_url} hash={row.image_hash or '?'} "
                f"dim={row.embedding.size} norm={row.vector_norm:.6f} "
                f"embedding_model={row.embedding_model or '?'}"
            )


def production_top1(model_hits: list[ModelHit], min_score: float = 0.2) -> ModelHit | None:
    if not model_hits:
        return None
    best = model_hits[0]
    if best.model_score < min_score:
        return None
    return best


def run_self_test(
    spec: str,
    rows: list[CatalogImage],
    encoder: VisionEncoder,
    session: requests.Session,
    top: int,
) -> dict[str, Any]:
    model, image_type = parse_self_test(spec)
    stored = next((row for row in rows if row.model == model and row.image_type == image_type), None)
    url = stored.source_url if stored and stored.source_url else catalog_image_url(model, image_type)
    image_bytes = download_image_bytes(url, session)
    encoded = encode_query(image_bytes, encoder)
    hits = rank_catalog_images(encoded["embedding_array"], rows)
    model_hits = aggregate_models_max(hits)
    stored_sim = None
    if stored is not None:
        stored_sim = float(np.dot(encoded["embedding_array"], stored.embedding))
    target = find_image_hit(hits, model, image_type)
    hash_match = None
    if stored is not None and stored.image_hash:
        hash_match = stored.image_hash == encoded["image_hash"]
    print(f"\nSelf-test {model}/{image_type}")
    print(f"  source={url}")
    print(f"  query_norm={encoded['vector_norm']:.6f} relevance={encoded['relevance']}")
    if stored is None:
        print("  stored catalog row: MISSING")
    else:
        print(
            f"  stored vs re-encode cosine={stored_sim:.6f} "
            f"hash_match={hash_match} stored_hash={stored.image_hash or '?'}"
        )
    if target is None:
        print("  re-encoded image was not found in catalog ranking")
    else:
        print(f"  rank={target.rank} similarity={target.similarity:.6f}")
        if target.rank != 1:
            print("  FAIL: catalog image did not retrieve itself at rank #1")
        elif stored_sim is not None and stored_sim < 0.999:
            print("  WARN: rank #1 but cosine is not extremely close to 1.0")
        else:
            print("  ok: rank #1")
    print("  top images:")
    print(format_image_table(hits[:top]))
    return {
        "spec": spec,
        "rank": target.rank if target else None,
        "similarity": target.similarity if target else None,
        "stored_cosine": stored_sim,
        "hash_match": hash_match,
        "ok": bool(target and target.rank == 1),
        "top_images": hits[:top],
        "top_models": model_hits[:top],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    load_dotenv(REPO_ROOT / ".env")

    try:
        for spec in args.self_test:
            parse_self_test(spec)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    if not args.image_url and not args.image_path and not args.self_test:
        logger.error("provide --image-url / --image-path and/or --self-test")
        return 1
    if args.top < 1:
        logger.error("--top must be >= 1")
        return 1

    catalog = load_catalog_rows(
        source=args.catalog_source,
        embedding_model=args.embedding_model,
        refresh=args.refresh_catalog,
    )
    integrity = inspect_catalog_integrity(catalog, expected_model=args.embedding_model)

    print("Current production matching method")
    print(f"  {PRODUCTION_NN}")
    print(f"  diagnostic catalog source={args.catalog_source} images={len(catalog)}")
    print_catalog_integrity(
        integrity,
        [args.expected_model or "", args.compare_model or ""],
        catalog,
    )

    encoder = VisionEncoder()
    session = requests.Session()
    self_results: list[dict[str, Any]] = []
    query_hits: list[ImageHit] = []
    query_models: list[ModelHit] = []
    query_meta: dict[str, Any] | None = None

    if args.image_url or args.image_path:
        image_bytes, source = load_query_bytes(args, session)
        query_meta = encode_query(image_bytes, encoder)
        query_meta["source"] = source
        print("\nQuery")
        print(f"  source={source}")
        print(
            f"  embedding_model={query_meta['embedding_model']} "
            f"dimension={query_meta['embedding_dimension']} "
            f"norm={query_meta['vector_norm']:.6f} "
            f"relevance={query_meta['relevance']} "
            f"footwear={query_meta['scores'].get('footwear')} "
            f"irrelevant={query_meta['scores'].get('irrelevant')}"
        )
        print_integrity(
            "  query vector",
            query_meta["embedding_array"],
            str(query_meta["embedding_model"]),
        )
        query_image = decode_image(image_bytes)
        if query_image.height > int(query_image.width * 1.4):
            print(
                f"  note: query is tall ({query_image.width}x{query_image.height}); "
                "SigLIP embeds the full screenshot including browser chrome and thumbnails."
            )
        query_hits = rank_catalog_images(query_meta["embedding_array"], catalog)
        query_models = aggregate_models_max(query_hits)

        expected_hits = hits_for_model(query_hits, args.expected_model or "")
        compare_hits = hits_for_model(query_hits, args.compare_model or "")
        expected_side = find_image_hit(query_hits, args.expected_model or "", "side")
        print_model_views("Expected model", args.expected_model or "(none)", expected_hits, rows_for_model(catalog, args.expected_model or ""))
        print_model_views("Compare model", args.compare_model or "(none)", compare_hits, rows_for_model(catalog, args.compare_model or ""))

        print("\nGlobal rank of expected side view")
        if not args.expected_model:
            print("  skipped (no --expected-model)")
        elif expected_side is None:
            print(f"  {args.expected_model} / side not present in ranking")
        else:
            print(f"  {args.expected_model} / side rank={expected_side.rank} similarity={expected_side.similarity:.6f}")

        print(f"\nTop {args.top} individual catalog images")
        print(format_image_table(query_hits[: args.top]))
        print(f"\nTop {args.top} models using MAX(view similarity)")
        print(format_model_table(query_models[: args.top]))

        production = production_top1(query_models)
        print("\nDiagnostic search top-1 (same rule as production: best individual catalog image)")
        if production is None:
            print("  below threshold / empty")
        else:
            print(
                f"  model={production.model} image_type={production.best_image_type} "
                f"score={production.model_score:.6f}"
            )
            if args.compare_model and production.model == args.compare_model:
                print(f"  this matches the reported production result {args.compare_model}")
            elif args.compare_model:
                print(f"  this does NOT match the reported production result {args.compare_model}")

    for spec in args.self_test:
        self_results.append(run_self_test(spec, catalog, encoder, session, args.top))

    expected_self_ok: bool | None = None
    if self_results and args.expected_model:
        expected_prefix = f"{args.expected_model}/"
        expected_tests = [item for item in self_results if str(item["spec"]).startswith(expected_prefix)]
        if expected_tests:
            expected_self_ok = all(bool(item["ok"]) for item in expected_tests)
    stale = [item for item in self_results if item.get("hash_match") is False]
    if stale:
        print("\nStale catalog embeddings (CDN bytes != stored image_hash)")
        for item in stale:
            print(
                f"  {item['spec']}: re-encode rank={item['rank']} "
                f"stored_cosine={item['stored_cosine']:.6f} hash_match=False"
            )

    if query_models:
        expected_side = find_image_hit(query_hits, args.expected_model or "", "side")
        compare_hits = hits_for_model(query_hits, args.compare_model or "")
        production = production_top1(query_models)
        case = classify_failure(
            expected=args.expected_model,
            compare=args.compare_model,
            expected_side=expected_side,
            compare_hits=compare_hits,
            production=production,
            expected_self_test_ok=expected_self_ok,
        )
        print("\nFailure classification")
        labels = {
            "A": "A - implementation/search bug",
            "B": "B - embedding/storage consistency bug",
            "C": "C - SigLIP retrieval-quality limitation",
            "ok": "ok - diagnostic top-1 matches expected model",
        }
        print(f"  {labels.get(case, case)}")
        if case == "A":
            print("  Raw SigLIP prefers the expected view, so production ranking/aggregation is wrong.")
        elif case == "B":
            print("  The expected catalog image did not retrieve itself. Fix embedding/storage before retrieval strategy.")
        elif case == "C":
            print(
                "  Pipeline compares individual 768D SigLIP views with MAX aggregation; "
                "the full query embedding is genuinely closer to another catalog shoe."
            )
        print("  Recommended next change: do not implement in this diagnostic pass.")
        if case == "C":
            print("  Crop website screenshots to the product packshot before embedding; also re-embed when CDN image_hash drifts.")
        elif case == "A":
            print("  Fix the CRM search/aggregation path so it uses per-image MAX cosine on 768D SigLIP.")
        elif case == "B":
            print("  Re-embed catalog images with the same VisionEncoder/preprocessing as query.")

    if args.json_out:
        payload = {
            "query": None
            if query_meta is None
            else {
                "source": query_meta.get("source"),
                "embedding_model": query_meta["embedding_model"],
                "embedding_dimension": query_meta["embedding_dimension"],
                "vector_norm": query_meta["vector_norm"],
                "relevance": query_meta["relevance"],
                "scores": query_meta["scores"],
            },
            "images": [
                {
                    "rank": hit.rank,
                    "model": hit.model,
                    "image_type": hit.image_type,
                    "similarity": hit.similarity,
                }
                for hit in query_hits[: max(args.top, 50)]
            ],
            "models": [
                {
                    "rank": hit.rank,
                    "model": hit.model,
                    "model_score": hit.model_score,
                    "best_image_type": hit.best_image_type,
                    "best_image_similarity": hit.best_image_similarity,
                }
                for hit in query_models[: max(args.top, 50)]
            ],
            "self_tests": [
                {key: value for key, value in item.items() if key not in {"top_images", "top_models"}}
                for item in self_results
            ],
        }
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
