"""Offline report: raw similarity vs like_score after the 60-day serving cut.

Usage:
    python -m inference.report_like_score --new-models path/to/load.csv --latest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from data.config import DEFAULT_OUTPUT_ROOT, DEFAULT_SNAPSHOT_ROOT, load_dotenv
from inference.csv_models import read_model_codes
from inference.ranking import rank_customers_for_model
from inference.recency import (
    BUCKET_1Y_3Y,
    BUCKET_181_365,
    BUCKET_3Y_PLUS,
    BUCKET_60_180,
    days_since_last_purchase,
    recency_distribution,
    rank_customers_with_recency,
)
from inference.recommender import DEFAULT_ARTIFACT, DEFAULT_LIMIT, RecommenderService

logger = logging.getLogger("pretty-reco-ml.like-score")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report raw similarity vs like_score rankings.")
    parser.add_argument("--new-models", required=True, type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "like_score_report.md")
    return parser.parse_args(argv)


def _days_for_ids(ranked: list[dict[str, Any]], id_to_date: dict[str, str | None]) -> list[float | None]:
    return [days_since_last_purchase(id_to_date.get(str(row["customer_id"]))) for row in ranked]


def summarize(ranked: list[dict[str, Any]], id_to_date: dict[str, str | None], score_key: str) -> dict[str, Any]:
    dist = recency_distribution(_days_for_ids(ranked, id_to_date))
    values = np.array([float(row[score_key]) for row in ranked], dtype=np.float32) if ranked else np.array([], dtype=np.float32)
    return {
        "result_count": len(ranked),
        "customers_60_180d": dist[BUCKET_60_180],
        "customers_181_365d": dist[BUCKET_181_365],
        "customers_1y_3y": dist[BUCKET_1Y_3Y],
        "customers_3y_plus": dist[BUCKET_3Y_PLUS],
        "average": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "minimum": float(values.min()) if values.size else None,
        "p10": float(np.percentile(values, 10)) if values.size else None,
        "p90": float(np.percentile(values, 90)) if values.size else None,
    }


def format_summary(label: str, summary: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    return (
        f"{label}: count={summary['result_count']} "
        f"60-180d={summary['customers_60_180d']} "
        f"181-365d={summary['customers_181_365d']} "
        f"1-3y={summary['customers_1y_3y']} "
        f"3y+={summary['customers_3y_plus']} "
        f"avg={fmt(summary['average'])} median={fmt(summary['median'])} "
        f"min={fmt(summary['minimum'])} p10={fmt(summary['p10'])} p90={fmt(summary['p90'])}"
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_dotenv()
    args = parse_args(argv)
    codes = read_model_codes(args.new_models)
    service = RecommenderService.load(
        artifact_dir=args.artifact,
        snapshot_dir=args.snapshot,
        snapshot_root=args.snapshot_root,
    )
    id_to_date = {
        customer_id: date
        for customer_id, date in zip(service.customer_ids, service.last_purchase_dates, strict=True)
    }
    lines = [
        "# like_score report",
        "",
        "Raw cosine vs isotonic like_score. Recency exclusion is a serving rule only.",
        f"Artifact: `{args.artifact}`. Limit: {args.limit}.",
        "",
    ]
    for code in codes:
        try:
            scores = service.score_model(code)
        except Exception as exc:
            logger.warning("skip %s: %s", code, exc)
            lines.extend([f"## `{code}`", "", f"skipped: {exc}", ""])
            continue
        like_scores = service.like_calibrator.transform(scores)
        similarity_ranked = rank_customers_for_model(scores, service.customer_ids, top_k=args.limit)
        like_ranked, diagnostics = rank_customers_with_recency(
            scores,
            service.customer_ids,
            service.last_purchase_dates,
            like_scores=like_scores,
            top_k=args.limit,
        )
        sim_summary = summarize(similarity_ranked, id_to_date, "similarity_score")
        like_summary = summarize(like_ranked, id_to_date, "like_score")
        lines.extend(
            [
                f"## `{code}`",
                "",
                f"- scored={diagnostics['total_scored']} excluded<60d={diagnostics['excluded_lt_60d']} eligible={diagnostics['eligible']}",
                f"- {format_summary('similarity TopN (no 60d cut)', sim_summary)}",
                f"- {format_summary('like_score TopN after 60d cut', like_summary)}",
                f"- returned_distribution={diagnostics['returned_distribution']}",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
