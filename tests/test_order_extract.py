"""Parse order screenshot text (issue #4 fixture #87610 / Kristen 38.5)."""

from __future__ import annotations

from embeddings.order_extract import parse_order_fields


ORDER_753_TEXT = """
Thank You For Your Order
הזמנתך #87610 התקבלה בהצלחה
Your Order Number #87610 (30/08/2026)
Kristen - 38.5
(01 53698_003385)
Subtotal: 1,190.00
"""


def test_parse_order_87610_kristen() -> None:
    fields = parse_order_fields(ORDER_753_TEXT)
    assert fields["order_number"] == "87610"
    assert fields["model"] == "53698_003"
    assert fields["size"] == "38.5"


def test_parse_order_empty() -> None:
    fields = parse_order_fields("a clothing advertisement")
    assert fields["order_number"] is None
    assert fields["model"] is None
    assert fields["size"] is None


def test_parse_new_order_87919() -> None:
    fields = parse_order_fields("Pretty Ballerinas\nNew Order: #87919")
    assert fields["order_number"] == "87919"


def test_parse_hebrew_order_bs580427134() -> None:
    text = """
Pretty Ballerinas
מספר הזמנה BS580427134
Kristen - 40.0
52797_004
"""
    fields = parse_order_fields(text)
    assert fields["order_number"] == "BS580427134"
    assert fields["model"] == "52797_004"
    assert fields["size"] == "40.0"


def test_parse_hebrew_confirmation_cs2247177794() -> None:
    text = """
אישור הזמנה CS2247177794
הזמנתך התקבלה בהצלחה
מספר הזמנה CS2247177794
Judy - 40.0
50724_001
Nicole - 40.5
50321_019
"""
    fields = parse_order_fields(text)
    assert fields["order_number"] == "CS2247177794"
    assert fields["model"] == "50724_001"
    assert fields["size"] == "40.0"


def test_parse_ocr_keeps_longest_cs_token() -> None:
    text = "5224717794\nCS224717794\nCS2247177794"
    fields = parse_order_fields(text)
    assert fields["order_number"] == "CS2247177794"


def test_parse_ocr_c5_prefix_is_cs() -> None:
    fields = parse_order_fields("מספר הזמנה C52247177794")
    assert fields["order_number"] == "CS2247177794"


def test_parse_numeric_web_order_is_not_prefixed_cs() -> None:
    fields = parse_order_fields("Thank You For Your Order\nYour Order Number #87610")
    assert fields["order_number"] == "87610"


def test_parse_size_without_hyphen_and_comma_decimal() -> None:
    assert parse_order_fields("Your Order Number #87610\nKristen 38.5")["size"] == "38.5"
    assert parse_order_fields("הזמנה #87610\nמידה 38,5")["size"] == "38.5"
