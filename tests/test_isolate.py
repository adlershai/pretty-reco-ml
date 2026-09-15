"""Shoe isolation crops. No SigLIP."""

from __future__ import annotations

from PIL import Image, ImageDraw

from embeddings.isolate import isolation_applied, pale_panels, propose_crops


def _tall_screenshot() -> Image.Image:
    image = Image.new("RGB", (200, 500), (30, 30, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, 79), fill=(20, 20, 22))
    draw.rectangle((0, 90, 199, 279), fill=(236, 228, 230))
    draw.polygon([(40, 210), (160, 210), (150, 250), (50, 250)], fill=(40, 30, 28))
    draw.rectangle((0, 300, 199, 499), fill=(236, 228, 230))
    return image


def test_tall_screenshot_proposes_studio_panels() -> None:
    image = _tall_screenshot()
    boxes = propose_crops(image)
    reasons = {box.reason for box in boxes}
    assert "full" in reasons
    assert "studio_panel" in reasons
    hero = [box for box in boxes if box.reason == "studio_panel" and box.top < 120]
    assert hero, boxes
    chosen = hero[0]
    assert chosen.top < 100
    assert chosen.bottom > 250
    assert isolation_applied(chosen, image)


def test_packshot_keeps_full_frame() -> None:
    image = Image.new("RGB", (1000, 1024), (236, 228, 230))
    draw = ImageDraw.Draw(image)
    draw.polygon([(200, 700), (800, 700), (750, 900), (250, 900)], fill=(30, 20, 20))
    boxes = propose_crops(image)
    assert len(boxes) == 1
    assert boxes[0].reason == "full"
    assert boxes[0].as_list() == [0, 0, 1000, 1024]
    assert pale_panels(image)
