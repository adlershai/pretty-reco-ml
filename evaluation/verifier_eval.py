"""Retrieve Top-10, send verifier jobs through CRM OpenAI, decide MATCH/UNCERTAIN, write HTML."""

from __future__ import annotations

import json
import subprocess
import sys
from html import escape
from pathlib import Path

from data.config import LOCAL_ROOT, load_dotenv
from embeddings.catalog_index import load_catalog_index
from embeddings.match_image import decide_match, match_image, match_result_to_dict
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import configure_logging, decode_image
from evaluation.image_match import catalog_image_url
from evaluation.wati_labels import CONFIRMED, NOT_SHOES, UNCERTAIN_LABELS

CACHE = Path(r"C:\Users\LENOVO\node-projects\pretty-ballerinas\pretty-crm-api\data\wati\image-eval\cache")
CRM_ROOT = Path(r"C:\Users\LENOVO\node-projects\pretty-ballerinas\pretty-crm-api")
OUT_DIR = LOCAL_ROOT / "outputs"
JOBS = OUT_DIR / "verifier_jobs.json"
OPENAI_OUT = OUT_DIR / "verifier_openai.json"
METRICS = OUT_DIR / "verifier_eval.json"
HTML = OUT_DIR / "verifier_eval.html"
K_VALUES = (1, 3, 5, 10, 20, 50)


def _label_state(event_id: int) -> str:
    if event_id in CONFIRMED:
        return "confirmed"
    if event_id in NOT_SHOES:
        return "not_shoes"
    if event_id in UNCERTAIN_LABELS:
        return "uncertain"
    return "unlabeled"


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / float(len(ranks))


def _img(src: str, alt: str) -> str:
    return (
        f'<img src="{escape(src, quote=True)}" alt="{escape(alt)}" '
        'width="180" height="180" style="width:180px;height:180px;object-fit:cover;background:#eee"/>'
    )


def write_html(rows: list[dict], metrics: dict) -> None:
    cards: list[str] = []
    for row in rows:
        expected = row.get("expected") or ""
        returned = row.get("returnedModel") or ""
        status = str(row.get("status") or "")
        wati = f"data:image/jpeg;base64,{row['thumb']}" if row.get("thumb") else ""
        returned_url = row.get("returnedPackshot") or ""
        expected_url = row.get("expectedPackshot") or ""
        if status == "uncertain":
            title = "UNCERTAIN"
            third = ""
            if row.get("leadingPackshots"):
                third = "".join(
                    _img(url, model) + f"<div>{escape(model)}</div>"
                    for model, url in row["leadingPackshots"][:3]
                )
            pair = f"{_img(wati, 'WATI')}{_img(returned_url, returned) if returned_url else ''}<div>{third}</div>"
        elif expected and returned and expected != returned:
            title = f"MATCH {returned} (expected {expected})"
            pair = (
                f"{_img(wati, 'WATI')}{_img(returned_url, returned)}"
                f"{_img(expected_url, expected)}"
            )
        else:
            title = f"{status.upper()} {returned or '—'}".strip()
            pair = f"{_img(wati, 'WATI')}{_img(returned_url, returned) if returned_url else ''}"
        cards.append(
            f"<article style='border:1px solid #ddd;padding:12px;margin:0 0 16px 0'>"
            f"<h3>Event {row['eventId']} · {escape(title)}</h3>"
            f"<div style='display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start'>{pair}</div>"
            f"<p>label={escape(row.get('labelState') or '')} "
            f"siglip_rank={row.get('expectedRank')} "
            f"siglip_top={escape(str(row.get('siglipTop') or ''))} "
            f"view={escape(str(row.get('returnedView') or ''))} "
            f"reason={escape(str(row.get('reason') or ''))}</p>"
            f"<p>{escape(str(row.get('verificationSummary') or ''))}</p>"
            f"</article>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>WATI verifier eval</title></head>
<body style="font-family:sans-serif;max-width:1100px;margin:24px">
<h1>WATI identity verifier</h1>
<p>OpenAI via pretty-crm-api · verify_top=10 · MATCH precision {metrics.get('precision')}
· coverage {metrics.get('coverage')} · n_confirmed {metrics.get('confirmed')}</p>
<p>Recall@1 {metrics.get('recallAt1')} · Recall@10 {metrics.get('recallAt10')}
· Recall@20 {metrics.get('recallAt20')}</p>
{''.join(cards)}
</body></html>
"""
    HTML.write_text(html, encoding="utf-8")


def main() -> int:
    configure_logging()
    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog_index()
    encoder = VisionEncoder()
    event_ids = sorted(
        {int(path.stem) for path in CACHE.glob("*.bin")},
        reverse=True,
    )
    retrieved: list[dict] = []
    jobs: list[dict] = []
    ranks: list[int | None] = []
    for event_id in event_ids:
        image = decode_image((CACHE / f"{event_id}.bin").read_bytes())
        result = match_image(image, encoder, catalog, top=50)
        payload = match_result_to_dict(
            result,
            image_size=image.size,
            image=image,
            catalog=catalog,
            verify_top=10,
        )
        expected = CONFIRMED.get(event_id)
        expected_rank = None
        if expected:
            for hit in result.candidates:
                if hit.model == expected:
                    expected_rank = hit.rank
                    break
            ranks.append(expected_rank)
        siglip_top = result.candidates[0].model if result.candidates else None
        retrieved.append(
            {
                "eventId": event_id,
                "labelState": _label_state(event_id),
                "expected": expected,
                "expectedRank": expected_rank,
                "siglipTop": siglip_top,
                "relevance": result.relevance,
                "payload": payload,
                "thumb": payload["preprocessing"].get("crop_jpeg_base64"),
            }
        )
        if payload["status"] == "needs_verification" and event_id not in {880, 753, 611, 589, 568, 559}:
            jobs.append(
                {
                    "eventId": event_id,
                    "verifier": payload["verifier"],
                    "crop_jpeg_base64": payload["preprocessing"].get("crop_jpeg_base64"),
                    "candidates": payload["candidates"],
                }
            )
        print(
            f"retrieve {event_id} rel={result.relevance} top={siglip_top} "
            f"expected={expected or '-'} rank={expected_rank}",
            flush=True,
        )

    JOBS.write_text(json.dumps(jobs), encoding="utf-8")
    cmd = [
        "node",
        str(CRM_ROOT / "ops" / "run-openai-verifier.js"),
        str(JOBS),
        str(OPENAI_OUT),
    ]
    print("openai", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(CRM_ROOT), check=False)
    if completed.returncode != 0:
        print("openai verifier failed", completed.returncode, file=sys.stderr)
        return completed.returncode
    openai_rows = {
        int(item["eventId"]): item
        for item in json.loads(OPENAI_OUT.read_text(encoding="utf-8"))
    }

    accepted = 0
    accepted_correct = 0
    html_rows: list[dict] = []
    detail: list[dict] = []
    for row in retrieved:
        event_id = int(row["eventId"])
        payload = row["payload"]
        openai_item = openai_rows.get(event_id)
        decided = None
        if openai_item is not None:
            decided = decide_match(openai_item.get("openai_output"), payload["candidates"])
        status = payload["status"]
        returned = None
        returned_view = None
        reason = ""
        verification = []
        if decided:
            status = decided["status"]
            reason = decided["reason"]
            verification = decided["verification"]
            if decided["match"]:
                returned = decided["match"]["model"]
                returned_view = decided["match"].get("best_image_type")
        elif status == "irrelevant":
            reason = "relevance_gate"
        expected = row["expected"]
        if row["labelState"] == "confirmed":
            if status == "match":
                accepted += 1
                if returned == expected:
                    accepted_correct += 1
        leading = []
        for hit in payload["candidates"][:3]:
            model = hit["model"]
            view = hit.get("best_image_type") or "main"
            leading.append((model, catalog_image_url(model, view)))
        html_rows.append(
            {
                "eventId": event_id,
                "labelState": row["labelState"],
                "status": status,
                "expected": expected,
                "expectedRank": row["expectedRank"],
                "siglipTop": row["siglipTop"],
                "returnedModel": returned,
                "returnedView": returned_view,
                "returnedPackshot": catalog_image_url(returned, returned_view or "main") if returned else "",
                "expectedPackshot": catalog_image_url(expected, "main") if expected else "",
                "leadingPackshots": leading,
                "reason": reason,
                "thumb": row["thumb"],
                "verificationSummary": "; ".join(
                    f"{item.get('model')}={item.get('verdict')}" for item in verification[:8]
                ),
            }
        )
        detail.append(
            {
                "eventId": event_id,
                "labelState": row["labelState"],
                "status": status,
                "expected": expected,
                "returned": returned,
                "expectedRank": row["expectedRank"],
                "reason": reason,
                "elapsedMs": None if openai_item is None else openai_item.get("elapsedMs"),
            }
        )

    confirmed_n = len(CONFIRMED)
    metrics = {
        "confirmed": confirmed_n,
        "accepted": accepted,
        "acceptedCorrect": accepted_correct,
        "precision": None if accepted == 0 else accepted_correct / float(accepted),
        "coverage": accepted / float(confirmed_n) if confirmed_n else 0.0,
        "recallAt1": recall_at_k(ranks, 1),
        "recallAt3": recall_at_k(ranks, 3),
        "recallAt5": recall_at_k(ranks, 5),
        "recallAt10": recall_at_k(ranks, 10),
        "recallAt20": recall_at_k(ranks, 20),
        "recallAt50": recall_at_k(ranks, 50),
        "verifyTop": 10,
        "model": "gpt-4.1 via pretty-crm-api watiImageIdentity",
        "rows": detail,
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_html(html_rows, metrics)
    print("precision", metrics["precision"], "coverage", metrics["coverage"], flush=True)
    print("wrote", HTML, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
