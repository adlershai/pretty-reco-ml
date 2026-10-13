"""Authenticated catalog rebuild and atomic hot-swap for visual search."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from embeddings.catalog_index import CatalogIndex
from embeddings.vision_encoder import EMBEDDING_DIMENSION

logger = logging.getLogger("embeddings.catalog_refresh")

Loader = Callable[[], CatalogIndex]
Activate = Callable[[CatalogIndex], None]
GetCurrent = Callable[[], CatalogIndex | None]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def catalog_counts(catalog: CatalogIndex | None) -> tuple[int, int]:
    if catalog is None:
        return 0, 0
    return int(catalog.image_count), int(catalog.model_count)


def validate_catalog(catalog: CatalogIndex | None) -> CatalogIndex:
    if catalog is None:
        raise RuntimeError("catalog is empty")
    images, models = catalog_counts(catalog)
    if images <= 0 or models <= 0:
        raise RuntimeError("catalog is empty")
    matrix = catalog.matrix
    if matrix is None or getattr(matrix, "ndim", 0) != 2:
        raise RuntimeError("catalog matrix is invalid")
    if int(matrix.shape[0]) != images:
        raise RuntimeError("catalog matrix mismatch")
    if int(matrix.shape[1]) != int(EMBEDDING_DIMENSION):
        raise RuntimeError("catalog dimension mismatch")
    return catalog


@dataclass
class CatalogRefreshJob:
    job_id: str
    status: str = "running"
    kept_previous: bool = False
    built_at: str | None = None
    image_count: int = 0
    model_count: int = 0
    error: str | None = None

    def as_dict(self, *, coalesced: bool = False, ok: bool | None = None) -> dict[str, Any]:
        ready = self.status == "ready"
        return {
            "ok": ready if ok is None else bool(ok),
            "job_id": self.job_id,
            "status": self.status,
            "coalesced": coalesced,
            "kept_previous": self.kept_previous,
            "built_at": self.built_at,
            "image_count": self.image_count,
            "model_count": self.model_count,
            "error": self.error,
        }


class CatalogRefreshService:
    def __init__(
        self,
        *,
        loader: Loader,
        activate: Activate,
        get_current: GetCurrent,
    ) -> None:
        self._loader = loader
        self._activate = activate
        self._get_current = get_current
        self._lock = threading.Lock()
        self._job: CatalogRefreshJob | None = None
        self._event: threading.Event | None = None
        self._running = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            job = self._job
            running = self._running
        if job is None:
            images, models = catalog_counts(self._get_current())
            return {
                "ok": images > 0,
                "job_id": "",
                "status": "idle",
                "coalesced": False,
                "kept_previous": False,
                "built_at": None,
                "image_count": images,
                "model_count": models,
                "error": None,
            }
        payload = job.as_dict(coalesced=False, ok=True if running else None)
        if running:
            payload["status"] = "running"
        return payload

    def refresh(self) -> dict[str, Any]:
        with self._lock:
            if self._running and self._job is not None and self._event is not None:
                job_id = self._job.job_id
                event = self._event
                coalesced = True
            else:
                job = CatalogRefreshJob(job_id=str(uuid.uuid4()), status="running")
                event = threading.Event()
                self._job = job
                self._event = event
                self._running = True
                coalesced = False
                job_id = job.job_id
                threading.Thread(
                    target=self._run,
                    name=f"catalog-refresh-{job_id[:8]}",
                    daemon=True,
                ).start()
        event.wait()
        with self._lock:
            finished = self._job
        if finished is None:
            finished = CatalogRefreshJob(job_id=job_id, status="failed", error="missing_job")
        return finished.as_dict(coalesced=coalesced)

    def _run(self) -> None:
        previous = self._get_current()
        prev_images, prev_models = catalog_counts(previous)
        with self._lock:
            job = self._job
            event = self._event
        if job is None:
            if event is not None:
                event.set()
            return
        try:
            built = validate_catalog(self._loader())
            self._activate(built)
            images, models = catalog_counts(built)
            with self._lock:
                job.status = "ready"
                job.kept_previous = False
                job.built_at = utc_now()
                job.image_count = images
                job.model_count = models
                job.error = None
                self._running = False
            logger.info(
                "catalog refresh ready images=%s models=%s job=%s",
                images,
                models,
                job.job_id,
            )
        except Exception as exc:
            logger.exception("catalog refresh failed job=%s", job.job_id)
            with self._lock:
                job.status = "failed"
                job.kept_previous = previous is not None
                job.built_at = None
                job.image_count = prev_images
                job.model_count = prev_models
                job.error = str(exc) or exc.__class__.__name__
                self._running = False
        finally:
            if event is not None:
                event.set()
