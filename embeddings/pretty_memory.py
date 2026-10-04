"""Pretty case-memory snapshot index.

CRM publishes structured cases to a shared JSON file. Reco owns embeddings,
indexing, and retrieval. The live index is swapped only after a full rebuild.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger("pretty-reco-ml.pretty_memory")

SNAPSHOT_FILENAME = "current.json"
DEFAULT_SNAPSHOT_DIR = "/home/ubuntu/pretty-memory"
REVIEW_WEIGHT = {
    "approved": 1.10,
    "draft": 1.02,
    "unreviewed": 1.0,
}


def snapshot_path(directory: str | None = None) -> Path:
    root = Path(directory or os.environ.get("PRETTY_MEMORY_SNAPSHOT_DIR") or DEFAULT_SNAPSHOT_DIR)
    return root / SNAPSHOT_FILENAME


def review_weight(status: str) -> float:
    return REVIEW_WEIGHT.get(str(status or "unreviewed").strip().lower(), 1.0)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


class PrettyMemoryIndex:
    def __init__(
        self,
        records: list[dict[str, Any]],
        recognition_vectors: np.ndarray,
        case_vectors: np.ndarray,
        snapshot_version: str,
        embedding_model: str,
    ) -> None:
        self.records = records
        self.recognition_vectors = recognition_vectors
        self.case_vectors = case_vectors
        self.snapshot_version = snapshot_version
        self.embedding_model = embedding_model

    def search(
        self,
        mode: Literal["recognition", "case"],
        query_vector: np.ndarray,
        *,
        role_id: str = "customer_success",
        top_k: int = 5,
        exclude_wa_ids: list[str] | None = None,
        exclude_conversation_ids: list[str] | None = None,
        exclude_case_keys: list[str] | None = None,
        exclude_ticket_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        matrix = self.recognition_vectors if mode == "recognition" else self.case_vectors
        if matrix.size == 0 or not self.records:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(query))
        if q_norm == 0.0:
            return []
        query = query / q_norm

        norms = np.linalg.norm(matrix, axis=1)
        safe = np.where(norms == 0.0, 1.0, norms)
        cosine = matrix.dot(query) / safe
        cosine = np.where(norms == 0.0, -1.0, cosine)

        skip_wa = {str(item) for item in (exclude_wa_ids or []) if str(item).strip()}
        skip_conv = {str(item) for item in (exclude_conversation_ids or []) if str(item).strip()}
        skip_keys = {str(item) for item in (exclude_case_keys or []) if str(item).strip()}
        skip_tickets = {str(item) for item in (exclude_ticket_ids or []) if str(item).strip()}
        wanted_role = str(role_id or "customer_success")

        scored: list[tuple[float, int]] = []
        for index, record in enumerate(self.records):
            if str(record.get("role_id") or "customer_success") != wanted_role:
                continue
            if skip_wa and str(record.get("wa_id") or "") in skip_wa:
                continue
            if skip_conv and str(record.get("conversation_id") or "") in skip_conv:
                continue
            if skip_keys and str(record.get("case_key") or "") in skip_keys:
                continue
            if skip_tickets and str(record.get("ticket_id") or "") in skip_tickets:
                continue
            field = "recognition_text" if mode == "recognition" else "case_text"
            text = _as_text(record.get(field)) or _as_text(record.get("case_summary"))
            if not text:
                continue
            weight = review_weight(str(record.get("review_status") or ""))
            scored.append((float(cosine[index]) * weight, index))

        scored.sort(key=lambda item: (-item[0], str(self.records[item[1]].get("case_key") or "")))
        top = scored[: max(1, int(top_k))]
        results: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(top, start=1):
            record = self.records[index]
            results.append(
                {
                    "case_key": str(record.get("case_key") or ""),
                    "ticket_id": str(record.get("ticket_id") or ""),
                    "rank": rank,
                    "score": score,
                    "review_status": str(record.get("review_status") or "unreviewed"),
                    "case_type": str(record.get("case_type") or ""),
                    "case_summary": str(record.get("case_summary") or ""),
                    "handling_summary": str(record.get("handling_summary") or ""),
                    "outcome_summary": str(record.get("outcome_summary") or ""),
                    "learning_summary": str(record.get("learning_summary") or ""),
                    "manager_feedback": str(
                        record.get("manager_correction")
                        or record.get("manager_feedback")
                        or ""
                    ),
                    "recognition_text": str(record.get("recognition_text") or ""),
                    "case_text": str(record.get("case_text") or ""),
                    "representation": mode,
                }
            )
        return results


def load_snapshot_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("INVALID_MEMORY_SNAPSHOT")
    return payload


def build_index(document: dict[str, Any], encoder: Any) -> PrettyMemoryIndex:
    cases = [row for row in document.get("cases") or [] if isinstance(row, dict) and row.get("case_key")]
    recognition_texts = [_as_text(row.get("recognition_text")) or "." for row in cases]
    case_texts = [_as_text(row.get("case_text")) or _as_text(row.get("case_summary")) or "." for row in cases]
    if cases:
        recognition_vectors = encoder.encode(recognition_texts, prefix="passage")
        case_vectors = encoder.encode(case_texts, prefix="passage")
    else:
        recognition_vectors = np.zeros((0, 1), dtype=np.float32)
        case_vectors = np.zeros((0, 1), dtype=np.float32)
    version = str(
        document.get("published_at")
        or document.get("snapshot_version")
        or datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )
    model_name = str(getattr(encoder, "model_name", "") or "")
    return PrettyMemoryIndex(cases, recognition_vectors, case_vectors, version, model_name)


class PrettyMemoryStore:
    def __init__(self, directory: str | None, encoder: Any) -> None:
        self.path = snapshot_path(directory)
        self.encoder = encoder
        self._lock = threading.Lock()
        self._index: PrettyMemoryIndex | None = None
        self._mtime_ns: int | None = None

    def loaded(self) -> bool:
        return self._index is not None

    def reload_if_changed(self) -> PrettyMemoryIndex | None:
        if not self.path.is_file():
            with self._lock:
                self._index = None
                self._mtime_ns = None
            return None
        mtime_ns = self.path.stat().st_mtime_ns
        with self._lock:
            if self._index is not None and self._mtime_ns == mtime_ns:
                return self._index
        document = load_snapshot_document(self.path)
        built = build_index(document, self.encoder)
        with self._lock:
            self._index = built
            self._mtime_ns = mtime_ns
        logger.info("pretty memory snapshot loaded path=%s cases=%s", self.path, len(built.records))
        return built

    def index(self) -> PrettyMemoryIndex | None:
        return self.reload_if_changed()
