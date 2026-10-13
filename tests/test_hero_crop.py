"""Deterministic screenshot hero crop. No SigLIP."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from embeddings.isolate import clear_hero_crop, propose_crops

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PDP_FIXTURES = ("wati_1068.jpg", "wati_1070.jpg", "wati_1071.jpg", "wati_1072.jpg")


def _pdp_screenshot() -> Image.Image:
    image = Image.new("RGB", (200, 500), (30, 30, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, 64), fill=(18, 18, 20))
    draw.rectangle((8, 80, 191, 310), fill=(236, 228, 230))
    draw.polygon([(50, 230), (150, 230), (140, 280), (60, 280)], fill=(40, 30, 28))
    draw.rectangle((8, 330, 94, 410), fill=(236, 228, 230))
    draw.rectangle((106, 330, 191, 410), fill=(236, 228, 230))
    return image


def _ambiguous_two_cards() -> Image.Image:
    image = Image.new("RGB", (200, 500), (30, 30, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 40, 189, 230), fill=(236, 228, 230))
    draw.rectangle((10, 260, 189, 450), fill=(236, 228, 230))
    return image


def _story_photo_gap() -> Image.Image:
    image = Image.new("RGB", (200, 500), (40, 30, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, 90), fill=(236, 228, 230))
    draw.rectangle((0, 340, 199, 499), fill=(236, 228, 230))
    return image


def test_pdp_hero_replaces_gallery_panels() -> None:
    image = _pdp_screenshot()
    hero = clear_hero_crop(image)
    assert hero is not None
    assert hero.reason == "hero_panel"
    assert hero.top < 120
    assert hero.bottom > 250
    assert hero.bottom < 360
    production = propose_crops(image, include_hero=False)
    gated = propose_crops(image, include_hero=True)
    assert "studio_panel" in {box.reason for box in production}
    reasons = {box.reason for box in gated}
    assert reasons == {"full", "hero_panel"}
    assert len(gated) == 2


def test_ambiguous_layout_keeps_production_panels() -> None:
    image = _ambiguous_two_cards()
    assert clear_hero_crop(image) is None
    production = propose_crops(image, include_hero=False)
    gated = propose_crops(image, include_hero=True)
    assert [box.as_list() + [box.reason] for box in production] == [
        box.as_list() + [box.reason] for box in gated
    ]
    assert "studio_panel" in {box.reason for box in gated}


def test_photo_gap_between_chrome_is_hero() -> None:
    image = _story_photo_gap()
    hero = clear_hero_crop(image)
    assert hero is not None
    assert hero.reason == "hero_gap"
    assert hero.top >= 70
    assert hero.bottom <= 350
    gated = propose_crops(image, include_hero=True)
    assert {box.reason for box in gated} == {"full", "hero_gap"}


def test_merged_pale_page_uses_upper_square() -> None:
    image = Image.new("RGB", (200, 500), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, 70), fill=(18, 18, 20))
    hero = clear_hero_crop(image)
    assert hero is not None
    assert hero.reason == "hero_upper"
    assert hero.top >= 60
    assert hero.bottom < 400


def test_env_enables_hero_without_kwarg(monkeypatch) -> None:
    monkeypatch.setenv("MATCH_HERO_CROP", "1")
    reasons = {box.reason for box in propose_crops(_pdp_screenshot())}
    assert reasons == {"full", "hero_panel"}


def test_ordinary_packshot_has_no_hero() -> None:
    image = Image.new("RGB", (1000, 1024), (236, 228, 230))
    draw = ImageDraw.Draw(image)
    draw.polygon([(200, 700), (800, 700), (750, 900), (250, 900)], fill=(30, 20, 20))
    assert clear_hero_crop(image) is None
    assert propose_crops(image, include_hero=True) == propose_crops(image, include_hero=False)


def test_order_fixture_is_not_forced_into_hero() -> None:
    path = FIXTURES / "order_87610.jpg"
    if not path.is_file():
        return
    image = Image.open(path).convert("RGB")
    production = propose_crops(image, include_hero=False)
    gated = propose_crops(image, include_hero=True)
    if clear_hero_crop(image) is None:
        assert [box.as_list() for box in production] == [box.as_list() for box in gated]


def test_pdp_fixtures_select_upper_hero_not_full_frame() -> None:
    missing = [name for name in PDP_FIXTURES if not (FIXTURES / name).is_file()]
    if missing:
        return
    for name in PDP_FIXTURES:
        image = Image.open(FIXTURES / name).convert("RGB")
        width, height = image.size
        production = propose_crops(image, include_hero=False)
        hero = clear_hero_crop(image)
        assert hero is not None, name
        assert hero.reason in {"hero_panel", "hero_upper", "hero_gap"}, (name, hero)
        assert hero.top < height * 0.45, (name, hero)
        assert hero.bottom < height * 0.75, (name, hero)
        assert hero.area < int(width * height * 0.92), name
        gated = propose_crops(image, include_hero=True)
        assert gated[0].reason == "full"
        assert any(box.reason.startswith("hero_") for box in gated), (name, gated)
        gallery = [
            box
            for box in production
            if box.reason == "studio_panel" and box.top > height * 0.55
        ]
        if gallery:
            assert all(box.top < height * 0.55 for box in gated if box.reason != "full"), name
