"""Pretty memory retrieval evaluation.

Separate from unit tests. Reports hit@1, hit@3, recall@5, and queries per mode.

Recognition queries: early customer wording → expected case keys.
Case/handling queries: understanding of the situation → expected case keys.

Usage (from pretty-reco-ml repo root, venv active):

    python -m evaluation.pretty_memory_eval
    python -m evaluation.pretty_memory_eval --fixture evaluation/pretty_memory_fixture.json
    python -m evaluation.pretty_memory_eval --snapshot /home/ubuntu/pretty-memory/current.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from embeddings.pretty_memory import PrettyMemoryIndex, build_index

DEFAULT_FIXTURE = Path(__file__).resolve().parent / "pretty_memory_fixture.json"
EVAL_K = 5


class KeywordEncoder:
    """Deterministic stand-in so the labeled fixture can score without GPU weights."""

    model_name = "pretty-memory-eval-keywords"

    def encode(self, texts: list[str], *, prefix: str = "passage") -> Any:
        rows = []
        for text in texts:
            lowered = text.lower()
            if "return" in lowered or "החזר" in lowered:
                rows.append([1.0, 0.0, 0.0])
            elif "hours" in lowered or "שעות" in lowered:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("INVALID_MEMORY_EVAL_FIXTURE")
    return payload


def index_from_cases(cases: list[dict[str, Any]], encoder: Any) -> PrettyMemoryIndex:
    return build_index({"published_at": "eval", "cases": cases}, encoder)


def _hit_at(retrieved: list[str], expected: list[str], k: int) -> bool:
    if not expected:
        return True
    top = retrieved[: max(1, k)]
    return any(key in top for key in expected)


def _recall_at(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    found = {key for key in retrieved[: max(1, k)] if key in expected}
    return len(found) / len(expected)


def evaluate_index(
    index: PrettyMemoryIndex,
    encoder: Any,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for query in queries:
        mode = str(query.get("mode") or "case")
        expected = [str(key) for key in (query.get("expected_case_keys") or []) if str(key).strip()]
        vector = encoder.encode([str(query.get("query_text") or "")], prefix="query")[0]
        results = index.search(mode, vector, top_k=EVAL_K)  # type: ignore[arg-type]
        retrieved = [str(item.get("case_key") or "") for item in results]
        row = {
            "id": str(query.get("id") or ""),
            "mode": mode,
            "expected": expected,
            "retrieved": retrieved,
            "hit@1": _hit_at(retrieved, expected, 1),
            "hit@3": _hit_at(retrieved, expected, 3),
            "recall@5": round(_recall_at(retrieved, expected, 5), 4),
        }
        rows.append(row)
        by_mode[mode].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"queries": 0, "hit@1": 0.0, "hit@3": 0.0, "recall@5": 0.0}
        count = len(items)
        return {
            "queries": count,
            "hit@1": round(sum(1 for item in items if item["hit@1"]) / count, 4),
            "hit@3": round(sum(1 for item in items if item["hit@3"]) / count, 4),
            "recall@5": round(sum(float(item["recall@5"]) for item in items) / count, 4),
        }

    return {
        "ok": True,
        "overall": summarize(rows),
        "by_mode": {mode: summarize(items) for mode, items in by_mode.items()},
        "queries": rows,
    }


def evaluate_fixture(path: Path = DEFAULT_FIXTURE, encoder: Any | None = None) -> dict[str, Any]:
    fixture = load_fixture(path)
    encoder = encoder or KeywordEncoder()
    index = index_from_cases(list(fixture.get("cases") or []), encoder)
    return evaluate_index(index, encoder, list(fixture.get("queries") or []))


def _fixture_passed(report: dict[str, Any]) -> bool:
    overall = report["overall"]
    return (
        overall["queries"] > 0
        and overall["hit@1"] >= 1.0
        and overall["hit@3"] >= 1.0
        and overall["recall@5"] >= 1.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretty memory retrieval evaluation")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="optional live current.json instead of fixture cases",
    )
    args = parser.parse_args()
    fixture = load_fixture(args.fixture)
    encoder = KeywordEncoder()
    if args.snapshot:
        document = json.loads(args.snapshot.read_text(encoding="utf-8"))
        index = build_index(document, encoder)
        report = evaluate_index(index, encoder, list(fixture.get("queries") or []))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = evaluate_index(index_from_cases(list(fixture.get("cases") or []), encoder), encoder, list(fixture.get("queries") or []))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if _fixture_passed(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
