"""Encode current customers and new models, then rank who to contact.

Usage:
    python -m inference.run --new-models path/to/load.csv --latest
    python -m inference.run --new-models path/to/load.csv --snapshot snapshots/<timestamp>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from data.config import DEFAULT_SNAPSHOT_ROOT, REPO_ROOT, load_dotenv
from data.dataset_builder import find_latest_snapshot, load_snapshot_frames
from data.db_client import DbApiClient
from training.artifact import load_artifact
from training.dataset import load_catalog
from training.trainer import select_device

from inference.csv_models import read_model_codes
from inference.customers import build_current_customers, encode_customer_vectors
from inference.export import (
    MATCH_COLUMNS,
    RANKING_COLUMNS,
    write_csv,
    write_top_customers_markdown,
)
from inference.models import (
    catalog_from_rows,
    encode_catalog_models,
    fetch_model_representation,
    model_status_rows,
)
from inference.ranking import (
    CUSTOMER_TOP_MODELS,
    EXPORT_TOP_K,
    enrich_customer_ranking,
    overlap_stats,
    per_model_summary,
    rank_customers_for_models,
    rank_models_for_customers,
    score_matrix,
    unique_customers_in_top,
)
from inference.sanity import check_customer_ranking, check_history_excludes_new_models, check_vectors

logger = logging.getLogger("pretty-reco-ml.inference")

DEFAULT_ARTIFACT = REPO_ROOT / "artifacts" / "the_pretty_model_v1"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank existing customers for new catalog models.")
    parser.add_argument("--new-models", required=True, type=Path, help="Headerless CSV; column A is model code")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    return parser.parse_args(argv)


def resolve_snapshot(args: argparse.Namespace) -> Path:
    if args.snapshot:
        return args.snapshot.resolve()
    if args.latest:
        return find_latest_snapshot(args.snapshot_root)
    return find_latest_snapshot(args.snapshot_root)


def print_completion(
    *,
    encoded_customers: int,
    encoded_models: int,
    ranking_pairs: int,
    overlap25: dict[str, Any],
    output_files: list[Path],
) -> None:
    print("customers encoded:")
    print(encoded_customers)
    print("new models encoded:")
    print(encoded_models)
    print("ranking pairs computed:")
    print(ranking_pairs)
    print("unique customers in Top 25:")
    print(overlap25.get("unique_customers", 0))
    print("average Top-25 overlap:")
    print(f"{overlap25.get('average_models_per_selected_customer', 0):.4f}")
    print("highest repeated customer count:")
    print(overlap25.get("highest_repeated_customer_count", 0))
    print("output files:")
    for path in output_files:
        print(path)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_dotenv()
    args = parse_args(argv)
    snapshot_dir = resolve_snapshot(args)
    artifact_dir = args.artifact.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = read_model_codes(args.new_models)
    new_model_set = set(codes)
    logger.info("new model codes from CSV: %s", len(codes))

    device = select_device()
    model = load_artifact(artifact_dir, device=device)
    dim = int(model.config.embedding_dim)
    if dim != 64:
        logger.warning("artifact embedding_dim is %s, expected 64", dim)

    purchases, customers, _models = load_snapshot_frames(snapshot_dir)
    catalog = load_catalog(snapshot_dir)
    logger.info("snapshot %s catalog models: %s", snapshot_dir.name, len(catalog))

    client = DbApiClient()
    live_rows = fetch_model_representation(client, codes)
    live_catalog = catalog_from_rows(live_rows)
    catalog.update(live_catalog)
    status = model_status_rows(codes, live_rows)
    for row in status:
        logger.info(
            "model %s vector=%s visual=%s missing=%s",
            row["model"],
            row["vector_generated"],
            row["visual_available"],
            row["missing_attributes"] or [],
        )
    missing_live = [code for code in codes if code not in live_catalog]
    if missing_live:
        logger.warning("new models without a usable visual/catalog row: %s", missing_live)

    records, customer_stats = build_current_customers(
        purchases,
        customers,
        catalog,
        new_model_codes=new_model_set,
    )
    logger.info(
        "customers total=%s encoded=%s excluded=%s reasons=%s",
        customer_stats["total_customers"],
        customer_stats["encoded_customers"],
        customer_stats["excluded_customers"],
        customer_stats["reason_counts"],
    )
    check_history_excludes_new_models(records, new_model_set)

    customer_vectors = encode_customer_vectors(model, records, catalog, device=device)
    model_vectors, encoded_codes = encode_catalog_models(model, catalog, codes, device=device)
    check_vectors(customer_vectors, dim=dim, name="customer_vectors")
    check_vectors(model_vectors, dim=dim, name="model_vectors")
    if encoded_codes != [code for code in codes if code in live_catalog]:
        logger.warning("encoded model order/count differs from CSV: %s", encoded_codes)

    customer_ids = [record["customer_id"] for record in records]
    scores = score_matrix(customer_vectors, model_vectors)
    ranked_raw = rank_customers_for_models(scores, customer_ids, encoded_codes, top_k=EXPORT_TOP_K)
    ranked = enrich_customer_ranking(ranked_raw, records)
    matches = rank_models_for_customers(scores, customer_ids, encoded_codes, top_k=CUSTOMER_TOP_MODELS)
    check_customer_ranking(ranked, new_model_codes=set(encoded_codes))

    summaries = per_model_summary(ranked)
    overlap10 = overlap_stats(ranked, top_n=10)
    overlap25 = overlap_stats(ranked, top_n=25)

    ranking_path = output_dir / "new_models_customer_ranking.csv"
    matches_path = output_dir / "customer_new_model_matches.csv"
    report_path = output_dir / "new_models_top_customers.md"
    write_csv(ranking_path, ranked, RANKING_COLUMNS)
    write_csv(matches_path, matches, MATCH_COLUMNS)
    write_top_customers_markdown(
        report_path,
        ranked=ranked,
        records=records,
        summaries=summaries,
        overlap_top10=overlap10,
        overlap_top25=overlap25,
        model_status=status,
        encoded_customers=len(records),
        encoded_models=len(encoded_codes),
        new_model_count=len(codes),
    )

    print_completion(
        encoded_customers=len(records),
        encoded_models=len(encoded_codes),
        ranking_pairs=len(ranked),
        overlap25=overlap25,
        output_files=[ranking_path, matches_path, report_path],
    )
    logger.info("unique Top 10 customers: %s", unique_customers_in_top(ranked, 10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
