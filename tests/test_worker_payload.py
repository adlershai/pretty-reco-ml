"""Payload validation and partial-failure behaviour (no vision model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embeddings.worker import PayloadError, collect_jobs, load_payload, parse_args


def test_malformed_json_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PayloadError, match="malformed"):
        load_payload(path)


def test_missing_models_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(PayloadError, match="unsupported payload"):
        load_payload(path)


def test_partial_views_are_jobs() -> None:
    payload = {
        "models": [
            {
                "model_id": 1,
                "model": "40724_001",
                "images": {
                    "main": "https://example.com/main.jpg",
                    "side": "https://example.com/side.jpg",
                },
            }
        ]
    }
    jobs, errors = collect_jobs(payload)
    assert errors == []
    assert [job.image_type for job in jobs] == ["main", "side"]


def test_invalid_url_is_per_image_error() -> None:
    payload = {
        "models": [
            {
                "model_id": 9,
                "model": "x",
                "images": {"main": "", "pers": "https://example.com/p.jpg"},
            }
        ]
    }
    jobs, errors = collect_jobs(payload)
    assert len(jobs) == 1
    assert jobs[0].image_type == "pers"
    assert errors == [
        {
            "model_id": 9,
            "model": "x",
            "image_type": "main",
            "error": "INVALID_URL",
        }
    ]


def test_batch_size_is_configurable() -> None:
    args = parse_args(["input.json", "--batch-size", "16"])
    assert args.batch_size == 16
    assert args.input_json == "input.json"
