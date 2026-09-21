"""Parse delivery-notice tracking text (issue #4 WATI 1057)."""

from __future__ import annotations

from embeddings.delivery_extract import parse_delivery_fields


EVENT_1057_TEXT = """
On The Way!
הפריטי שלך כבר ארוזות והן בדרך אליך
מספר משלוח
102312132
מעקב משלוח
"""


def test_parse_1057_on_the_way_tracking() -> None:
    fields = parse_delivery_fields(EVENT_1057_TEXT)
    assert fields["tracking_number"] == "102312132"
    assert fields["status"] == "on_the_way"


def test_parse_delivery_empty() -> None:
    fields = parse_delivery_fields("Thank You For Your Order #87610")
    assert fields["tracking_number"] is None
    assert fields["status"] is None
