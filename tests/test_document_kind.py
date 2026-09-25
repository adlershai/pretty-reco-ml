"""Document wording can choose order vs delivery without SigLIP."""

from __future__ import annotations

from embeddings.document_kind import classify_document_kind
from embeddings.relevance import IMAGE_KIND_DELIVERY, IMAGE_KIND_ORDER


def test_on_the_way_is_delivery_even_with_digits() -> None:
    text = "On The Way!\nמספר משלוח 102312132\nמעקב משלוח"
    assert classify_document_kind(text) == IMAGE_KIND_DELIVERY


def test_thank_you_for_your_order_is_order() -> None:
    text = "Thank You For Your Order\nYour Order Number #87610"
    assert classify_document_kind(text) == IMAGE_KIND_ORDER


def test_hebrew_order_confirmation_is_order() -> None:
    text = "אישור הזמנה CS2247177794\nהזמנתך התקבלה בהצלחה\nמספר הזמנה CS2247177794"
    assert classify_document_kind(text) == IMAGE_KIND_ORDER


def test_strong_order_needs_number_and_wording() -> None:
    from embeddings.document_kind import has_strong_order_evidence

    text = "אישור הזמנה CS2247177794"
    assert has_strong_order_evidence(text, {"order_number": "CS2247177794"}) is True
    assert has_strong_order_evidence("", {"order_number": "CS2247177794"}) is True
    assert has_strong_order_evidence(text, {"order_number": None}) is False
    assert has_strong_order_evidence("", {"order_number": "87610"}) is False


def test_empty_text_does_not_decide() -> None:
    assert classify_document_kind("") is None
    assert classify_document_kind("   ") is None
