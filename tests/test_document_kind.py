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


def test_empty_text_does_not_decide() -> None:
    assert classify_document_kind("") is None
    assert classify_document_kind("   ") is None
