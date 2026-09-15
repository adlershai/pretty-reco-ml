"""Benchmark original vs isolated-shoe SigLIP retrieval.

Usage:
    python -m evaluation.match_benchmark
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from data.config import LOCAL_ROOT, REPO_ROOT, load_dotenv
from embeddings.catalog_index import load_catalog_index
from embeddings.isolate import propose_crops
from embeddings.match_image import match_image
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import configure_logging, decode_image, download_image_bytes

QUERY_845 = "https://pb-il-app-images-1.s3.eu-central-1.amazonaws.com/ops/wati/6aa3e530791a84d02c58c38e/845.jpg"
CASES = [
    {"id": "845", "url": QUERY_845, "expected": "52792_006", "kind": "screenshot"},
    {"id": "self-52792_006-side", "url": "https://media.adler.co.il/app/products/52792_006_side.jpg", "expected": "52792_006", "kind": "packshot"},
    {"id": "self-40724_001-main", "url": "https://media.adler.co.il/app/products/40724_001.jpg", "expected": "40724_001", "kind": "packshot"},
]


def _metrics(rank: int | None) -> dict[str, int]:
    hits = {
        "recall@1": 1 if rank == 1 else 0,
        "recall@3": 1 if rank is not None and rank <= 3 else 0,
        "recall@5": 1 if rank is not None and rank <= 5 else 0,
        "recall@10": 1 if rank is not None and rank <= 10 else 0,
    }
    return hits


def _model_rank(models, expected: str) -> int | None:
    for hit in models:
        if hit.model == expected:
            return hit.rank
    return None


def main() -> int:
    configure_logging()
    load_dotenv(REPO_ROOT / ".env")
    catalog = load_catalog_index()
    encoder = VisionEncoder()
    session = requests.Session()
    rows = []
    for case in CASES:
        image = decode_image(download_image_bytes(case["url"], session))
        original_models = catalog.search_models(encoder.encode(image), top=10)
        isolated = match_image(image, encoder, catalog, top=10)
        original_rank = _model_rank(original_models, case["expected"])
        isolated_rank = _model_rank(isolated.candidates, case["expected"])
        original_top1 = original_models[0].model_score if original_models else None
        original_top2 = original_models[1].model_score if len(original_models) > 1 else None
        isolated_top1 = isolated.candidates[0].model_score if isolated.candidates else None
        isolated_top2 = isolated.candidates[1].model_score if len(isolated.candidates) > 1 else None
        row = {
            "id": case["id"],
            "expected": case["expected"],
            "kind": case["kind"],
            "original_top1": original_models[0].model if original_models else None,
            "original_rank": original_rank,
            "original_margin": None
            if original_top1 is None or original_top2 is None
            else original_top1 - original_top2,
            "isolated_top1": isolated.candidates[0].model if isolated.candidates else None,
            "isolated_rank": isolated_rank,
            "isolated_margin": None
            if isolated_top1 is None or isolated_top2 is None
            else isolated_top1 - isolated_top2,
            "shoe_isolated": isolated.shoe_isolated,
            "crop": isolated.crop.as_list(),
            "original_metrics": _metrics(original_rank),
            "isolated_metrics": _metrics(isolated_rank),
            "crop_count": len(propose_crops(image)),
        }
        rows.append(row)
        print(
            f"{case['id']}: original rank={original_rank} top1={row['original_top1']} "
            f"isolated rank={isolated_rank} top1={row['isolated_top1']} "
            f"isolated={isolated.shoe_isolated}"
        )

    def mean_metric(key: str, split: str) -> float:
        values = [row[split][key] for row in rows]
        return sum(values) / len(values) if values else 0.0

    original_recall1 = mean_metric("recall@1", "original_metrics")
    isolated_recall1 = mean_metric("recall@1", "isolated_metrics")
    screenshot_ok = all(row["kind"] != "screenshot" or row["isolated_rank"] == 1 for row in rows)
    packshot_ok = all(row["kind"] != "packshot" or row["isolated_rank"] == 1 for row in rows)
    if screenshot_ok and packshot_ok and isolated_recall1 >= original_recall1:
        stage3 = (
            "not required: isolation recovers known screenshot failures "
            "and packshot self-matches stay rank 1"
        )
    else:
        stage3 = (
            "warranted: isolation did not reach Recall@1 on the evaluation set; "
            "do not add a reranker until this set is expanded and re-measured"
        )
    summary = {
        "cases": rows,
        "original": {name: mean_metric(name, "original_metrics") for name in ("recall@1", "recall@3", "recall@5", "recall@10")},
        "isolated": {name: mean_metric(name, "isolated_metrics") for name in ("recall@1", "recall@3", "recall@5", "recall@10")},
        "stage3": stage3,
    }
    out_dir = LOCAL_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "match_benchmark.json"
    md_path = out_dir / "match_benchmark.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Visual match benchmark (Stage 1 isolation)",
        "",
        f"Original Recall@1={summary['original']['recall@1']:.2f}  Isolated Recall@1={summary['isolated']['recall@1']:.2f}",
        f"Original Recall@3={summary['original']['recall@3']:.2f}  Isolated Recall@3={summary['isolated']['recall@3']:.2f}",
        f"Original Recall@5={summary['original']['recall@5']:.2f}  Isolated Recall@5={summary['isolated']['recall@5']:.2f}",
        f"Original Recall@10={summary['original']['recall@10']:.2f} Isolated Recall@10={summary['isolated']['recall@10']:.2f}",
        "",
        summary["stage3"],
        "",
        "| id | expected | original top1 | original rank | isolated top1 | isolated rank | margin isolated |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        margin = "" if row["isolated_margin"] is None else f"{row['isolated_margin']:.4f}"
        lines.append(
            f"| {row['id']} | {row['expected']} | {row['original_top1']} | {row['original_rank']} | "
            f"{row['isolated_top1']} | {row['isolated_rank']} | {margin} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("Stage 3 reranker:", summary["stage3"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
