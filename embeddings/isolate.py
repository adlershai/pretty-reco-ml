"""Propose product-region crops before visual matching.

Tall screenshots (browser chrome, page UI) are split into pale studio-background
panels. Near-square packshots keep the original frame so toe/heel are not cut.

MATCH_HERO_CROP (off by default): when a website/story hero region is unambiguous,
propose that box instead of every studio panel so gallery thumbnails cannot win
catalog-score selection. Ambiguous layouts keep the existing panel set.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from PIL import Image

TALL_ASPECT = 1.35
MIN_PANEL_AREA = 0.12
STUDIO_LUMA = 195.0
STUDIO_SAT = 45.0
ROW_THRESH = 0.35
COL_THRESH = 0.25
MIN_RUN_FRAC = 0.05
EDGE_INSET = 8
HERO_CROP_ENV = "MATCH_HERO_CROP"
HERO_DOMINANCE = 1.8
HERO_MAX_TOP_FRAC = 0.55
HERO_GAP_FRAC = 0.18
HERO_ASPECT = 1.15
PANEL_MAX_FRAC = 0.92
GAP_PALE_MAX = 0.55


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int
    reason: str

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]

    def pil_box(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


def _clamp_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> CropBox | None:
    left = max(0, min(int(left), width - 1))
    top = max(0, min(int(top), height - 1))
    right = max(left + 1, min(int(right), width))
    bottom = max(top + 1, min(int(bottom), height))
    if right - left < 8 or bottom - top < 8:
        return None
    return CropBox(left=left, top=top, right=right, bottom=bottom, reason="panel")


def _runs(values: np.ndarray, thresh: float, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if value >= thresh:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_len:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(values) - start >= min_len:
        runs.append((start, len(values) - 1))
    return runs


def studio_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    sat = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    return (luma > STUDIO_LUMA) & (sat < STUDIO_SAT)


def pale_panels(image: Image.Image) -> list[CropBox]:
    width, height = image.size
    mask = studio_mask(image)
    min_row = max(24, int(height * MIN_RUN_FRAC))
    min_col = max(24, int(width * MIN_RUN_FRAC))
    boxes: list[CropBox] = []
    for top, bottom in _runs(mask.mean(axis=1), ROW_THRESH, min_row):
        band = mask[top : bottom + 1]
        for left, right in _runs(band.mean(axis=0), COL_THRESH, min_col):
            inset = EDGE_INSET if min(right - left, bottom - top) > EDGE_INSET * 4 else 0
            box = _clamp_box(
                left + inset,
                top + inset,
                right + 1 - inset,
                bottom + 1 - inset,
                width,
                height,
            )
            if box is None:
                continue
            if box.area / float(width * height) < MIN_PANEL_AREA:
                continue
            if box.left == 0 and box.top == 0 and box.right == width and box.bottom == height:
                continue
            boxes.append(CropBox(box.left, box.top, box.right, box.bottom, "studio_panel"))
    return boxes


def hero_crop_enabled() -> bool:
    return os.environ.get(HERO_CROP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _photo_gap_hero(panels: list[CropBox], image: Image.Image) -> CropBox | None:
    """Gap between pale chrome bands, only if that gap is a photo not empty page."""
    ordered = sorted(panels, key=lambda box: (box.top, -box.area))
    width, height = image.size
    mask = studio_mask(image)
    min_gap = int(height * HERO_GAP_FRAC)
    for index in range(len(ordered) - 1):
        gap_top = ordered[index].bottom
        gap_bottom = ordered[index + 1].top
        if gap_bottom - gap_top < min_gap:
            continue
        band = mask[gap_top:gap_bottom]
        if band.size == 0 or float(band.mean()) > GAP_PALE_MAX:
            continue
        left = min(ordered[index].left, ordered[index + 1].left)
        right = max(ordered[index].right, ordered[index + 1].right)
        box = _clamp_box(left, gap_top, right, gap_bottom, width, height)
        if box is None:
            continue
        return CropBox(box.left, box.top, box.right, box.bottom, "hero_gap")
    return None


def _upper_hero(panel: CropBox, height: int) -> CropBox:
    """Website PDP: pale hero and gallery often merge; keep the upper square."""
    hero_h = min(panel.height, int(panel.width * HERO_ASPECT), int(height * 0.62))
    hero_h = max(hero_h, 32)
    if hero_h >= int(panel.height * 0.85):
        return CropBox(panel.left, panel.top, panel.right, panel.bottom, "hero_panel")
    return CropBox(panel.left, panel.top, panel.right, panel.top + hero_h, "hero_upper")


def clear_hero_crop(image: Image.Image) -> CropBox | None:
    """Dominant PDP hero or photo-between-chrome. None when the layout is ambiguous."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if height <= width * TALL_ASPECT:
        return None
    usable = [
        box
        for box in pale_panels(rgb)
        if box.area < int(width * height * PANEL_MAX_FRAC)
    ]
    gap = _photo_gap_hero(usable, rgb)
    if gap is not None:
        return gap
    if not usable:
        return None
    ordered = sorted(usable, key=lambda box: (-box.area, box.top))
    hero = ordered[0]
    if hero.top > height * HERO_MAX_TOP_FRAC:
        return None
    if len(ordered) >= 2 and hero.area < ordered[1].area * HERO_DOMINANCE:
        return None
    return _upper_hero(hero, height)


def _dedupe_boxes(boxes: list[CropBox]) -> list[CropBox]:
    unique: list[CropBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        key = (box.left, box.top, box.right, box.bottom)
        if key in seen:
            continue
        seen.add(key)
        unique.append(box)
    return unique


def propose_crops(image: Image.Image, *, include_hero: bool | None = None) -> list[CropBox]:
    """Full frame always. Tall screenshots add studio panels, or one clear hero."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    boxes = [CropBox(0, 0, width, height, "full")]
    if height > width * TALL_ASPECT:
        use_hero = hero_crop_enabled() if include_hero is None else include_hero
        hero = clear_hero_crop(rgb) if use_hero else None
        if hero is not None:
            boxes.append(hero)
        else:
            boxes.extend(pale_panels(rgb))
    return _dedupe_boxes(boxes)


def crop_image(image: Image.Image, box: CropBox) -> Image.Image:
    return image.convert("RGB").crop(box.pil_box())


def isolation_applied(chosen: CropBox, original: Image.Image) -> bool:
    width, height = original.size
    if chosen.reason == "full":
        return False
    return chosen.area < int(width * height * 0.92)
