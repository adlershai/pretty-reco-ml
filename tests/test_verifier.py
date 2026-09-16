"""Identity verifier policy. No OpenAI calls."""

from __future__ import annotations

from embeddings.verifier import (
    DEFAULT_VERIFY_TOP,
    build_verifier_request,
    decide_identity,
    parse_openai_output,
    verify_top_value,
)


def _same(model: str, **overrides: object) -> dict:
    row = {
        "model": model,
        "verdict": "same",
        "shape": 0.96,
        "pattern": 0.95,
        "material": 0.9,
        "details": 0.94,
        "color_placement": 0.8,
        "contradictions": [],
        "reason": "same last and pattern",
    }
    row.update(overrides)
    return row


def test_verify_top_defaults_to_10(monkeypatch) -> None:
    monkeypatch.delenv("MATCH_VERIFY_TOP", raising=False)
    assert verify_top_value() == DEFAULT_VERIFY_TOP
    monkeypatch.setenv("MATCH_VERIFY_TOP", "7")
    assert verify_top_value() == 7
    assert verify_top_value(3) == 3


def test_unique_same_is_match() -> None:
    decision = decide_identity(
        {
            "candidates": [
                _same("51604_A"),
                _same("50583_C", verdict="different", shape=0.4, pattern=0.2, reason="round vs pointed"),
            ]
        },
        [{"model": "51604_A", "score": 0.78}, {"model": "50583_C", "score": 0.79}],
    )
    assert decision.status == "match"
    assert decision.model == "51604_A"
    assert decision.confidence is not None and decision.confidence >= 0.9


def test_two_sames_are_uncertain() -> None:
    decision = decide_identity(
        {"candidates": [_same("51604_A"), _same("50583_C")]},
        [{"model": "51604_A"}, {"model": "50583_C"}],
    )
    assert decision.status == "uncertain"
    assert decision.model is None
    assert "ambiguous" in decision.reason


def test_exclusionary_contradiction_is_not_a_match() -> None:
    decision = decide_identity(
        {
            "candidates": [
                _same(
                    "50583_C",
                    shape=0.3,
                    pattern=0.2,
                    contradictions=["pointed toe vs round toe"],
                )
            ]
        },
        [{"model": "50583_C"}],
    )
    assert decision.status == "uncertain"
    assert decision.model is None


def test_failed_openai_wrapper_is_uncertain() -> None:
    decision = decide_identity({"result": "failed", "message": "no key"}, [{"model": "A"}])
    assert decision.status == "uncertain"
    assert decision.reason == "verifier_failed"


def test_parse_responses_output_array() -> None:
    parsed = parse_openai_output(
        [{"content": [{"text": '{"candidates":[{"model":"X","verdict":"different","shape":0.1,"pattern":0.1,"material":0.1,"details":0.1,"color_placement":0.1,"contradictions":[],"reason":"no"}]}'}]}]
    )
    assert parsed is not None
    assert parsed["candidates"][0]["model"] == "X"


def test_build_verifier_request_uses_verify_top() -> None:
    rows = [
        type("Row", (), {"model": "51604_A", "image_type": "main"})(),
        type("Row", (), {"model": "51604_A", "image_type": "side"})(),
        type("Row", (), {"model": "50583_C", "image_type": "pers"})(),
    ]
    payload = build_verifier_request(
        [{"model": "51604_A"}, {"model": "50583_C"}, {"model": "OTHER"}],
        catalog_rows=rows,
        verify_top=2,
    )
    assert payload["verify_top"] == 2
    assert payload["requestType"] == "watiImageIdentity"
    slots = [item["slot"] for item in payload["images"]]
    assert slots[0] == "customer"
    assert "51604_A_main" in slots
    assert "51604_A_side" in slots
    assert "50583_C_pers" in slots
    assert "OTHER_main" not in slots
