"""Propose product-region crops before visual matching.

Tall screenshots (browser chrome, page UI) are split into pale studio-background
panels. Near-square packshots keep the original frame so toe/heel are not cut.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def propose_crops(image: Image.Image) -> list[CropBox]:
    """Always includes the full frame; adds studio panels for tall screenshots."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    boxes = [CropBox(0, 0, width, height, "full")]
    if height > width * TALL_ASPECT:
        boxes.extend(pale_panels(rgb))
    unique: list[CropBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        key = (box.left, box.top, box.right, box.bottom)
        if key in seen:
            continue
        seen.add(key)
        unique.append(box)
    return unique


def crop_image(image: Image.Image, box: CropBox) -> Image.Image:
    return image.convert("RGB").crop(box.pil_box())


def isolation_applied(chosen: CropBox, original: Image.Image) -> bool:
    width, height = original.size
    if chosen.reason == "full":
        return False
    return chosen.area < int(width * height * 0.92)
