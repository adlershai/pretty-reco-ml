"""Write ranking CSVs and the human-readable Top-10 report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from inference.ranking import HISTORICAL_EVAL_LIMITATION

RANKING_COLUMNS = (
    "model",
    "customer_rank",
    "customer_id",
    "customer_name",
    "similarity_score",
    "history_length",
    "last_purchase_date",
    "preferred_size",
    "last_purchased_size",
)

MATCH_COLUMNS = ("customer_id", "model", "rank_for_customer", "similarity_score")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_top_customers_markdown(
    path: Path,
    *,
    ranked: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    overlap_top10: dict[str, Any],
    overlap_top25: dict[str, Any],
    model_status: Sequence[dict[str, Any]],
    encoded_customers: int,
    encoded_models: int,
    new_model_count: int,
) -> None:
    by_id = {str(record["customer_id"]): record for record in records}
    lines = [
        "# New-model customer ranking",
        "",
        "Current customer vectors scored against cold-start new shoes (frozen 64D two-tower).",
        f"New models in this run: **{new_model_count}** (CSV is the source of truth; the phase brief said 25).",
        f"Customers encoded: **{encoded_customers}**. New models encoded: **{encoded_models}**.",
        "",
        "## Model encode status",
        "",
        "| model | vector | visual | missing attributes |",
        "| --- | --- | --- | --- |",
    ]
    for row in model_status:
        missing = ", ".join(row.get("missing_attributes") or []) or "—"
        lines.append(
            f"| `{row['model']}` | "
            f"{'yes' if row.get('vector_generated') else 'no'} | "
            f"{'yes' if row.get('visual_available') else 'no'} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Per-model score summary",
            "",
            "| model | score_max | median_top_50 | min_top_50 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            f"| `{row['model']}` | {_fmt_score(row.get('score_max'))} | "
            f"{_fmt_score(row.get('score_median_top_50'))} | {_fmt_score(row.get('score_min_top_50'))} |"
        )
    lines.extend(
        [
            "",
            "## Customer overlap",
            "",
            f"- Unique customers in all Top 10 lists: **{overlap_top10.get('unique_customers', 0)}**",
            f"- Unique customers in all Top 25 lists: **{overlap_top25.get('unique_customers', 0)}**",
            f"- Average models per selected Top-25 customer: **{overlap_top25.get('average_models_per_selected_customer', 0):.2f}**",
            f"- Max models assigned to one customer (Top 25): **{overlap_top25.get('max_models_assigned_to_one_customer', 0)}**",
            "",
        ]
    )
    if overlap_top25.get("highest_repeated_customer_id"):
        lines.append(
            f"Customer `{overlap_top25['highest_repeated_customer_id']}` appears in Top 25 for "
            f"{overlap_top25['highest_repeated_customer_count']} of the new models."
        )
        lines.append("")
    lines.extend(["## Top 10 customers per model", ""])
    current_model = None
    for row in ranked:
        if int(row["customer_rank"]) > 10:
            continue
        if row["model"] != current_model:
            current_model = row["model"]
            lines.extend([f"### `{current_model}`", ""])
        record = by_id.get(str(row["customer_id"])) or {}
        name = row.get("customer_name") or record.get("customer_name") or "—"
        recent = ", ".join(str(code) for code in (record.get("recent_models") or [])[-5:]) or "—"
        lines.append(
            f"{int(row['customer_rank'])}. `{row['customer_id']}` / {name} / {_fmt_score(row['similarity_score'])}"
        )
        lines.append(
            f"   - history length: {record.get('history_length', row.get('history_length'))}; "
            f"last purchase: {record.get('last_purchase_date', row.get('last_purchase_date')) or '—'}"
        )
        lines.append(f"   - recent models: {recent}")
        lines.append("")
    lines.extend(["## Historical evaluation", "", HISTORICAL_EVAL_LIMITATION, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
