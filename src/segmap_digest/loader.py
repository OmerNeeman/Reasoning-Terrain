"""Load a label raster from disk, plus a colormap for eyeballing it.

Accepts GeoTIFF (via rasterio, keeps the georeference), .npy, or a single-band
PNG. Values must be class ids in 0..44.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .taxonomy import N_CLASSES, SUPERCLASS_OF


@dataclass
class LabelRaster:
    labels: np.ndarray            # (H, W) uint8, values 0..44
    gsd: float = 0.3              # metres per pixel
    transform: object | None = None   # rasterio Affine, when available
    crs: object | None = None
    dem: np.ndarray | None = None     # (H, W) float32 elevation, metres

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape

    @property
    def pixel_area_m2(self) -> float:
        return self.gsd * self.gsd


def load(path: str | Path, gsd: float | None = None) -> LabelRaster:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            labels = src.read(1)
            tr, crs = src.transform, src.crs
            inferred = abs(tr.a) if tr is not None else 0.3
        return LabelRaster(_check(labels), gsd or inferred, tr, crs)

    if suffix == ".npy":
        return LabelRaster(_check(np.load(path)), gsd or 0.3)

    if suffix in (".png", ".bmp"):
        from PIL import Image

        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            raise ValueError(
                f"{path} is a {arr.shape[2]}-band image. This tool wants a single-band "
                "label raster, not a colourised one -- recovering class ids from RGB is "
                "lossy and ambiguous."
            )
        return LabelRaster(_check(arr), gsd or 0.3)

    raise ValueError(f"unsupported label raster format: {suffix}")


def _check(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"label raster must be 2-D, got shape {labels.shape}")
    lo, hi = int(labels.min()), int(labels.max())
    if lo < 0 or hi >= N_CLASSES:
        raise ValueError(
            f"label values must be in 0..{N_CLASSES - 1}, got {lo}..{hi}"
        )
    return labels.astype(np.uint8)


# --- colormap --------------------------------------------------------------
# Hue by superclass so a glance separates rock from vegetation from built,
# lightness varying within a superclass. This is for humans; feeding a
# colourised label map to a VLM and asking it to read exact colours off a
# 45-entry legend is a task VLMs are bad at.

_SUPER_HUE = {
    "artifact": 0.83, "built": 0.00, "road": 0.08, "vehicle": 0.95,
    "water": 0.58, "soil": 0.05, "agriculture": 0.28, "vegetation": 0.35,
    "rock": 0.13,
}


def colormap() -> np.ndarray:
    """(45, 3) uint8 RGB palette."""
    counts: dict[str, int] = {}
    order: dict[int, int] = {}
    for c in range(N_CLASSES):
        sc = SUPERCLASS_OF[c]
        order[c] = counts.get(sc, 0)
        counts[sc] = order[c] + 1

    out = np.zeros((N_CLASSES, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        sc = SUPERCLASS_OF[c]
        n = counts[sc]
        frac = order[c] / max(n - 1, 1)
        hue = _SUPER_HUE[sc]
        light = 0.30 + 0.50 * frac
        sat = 0.75 if sc != "artifact" else 0.15
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        out[c] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def colourise(labels: np.ndarray) -> np.ndarray:
    """(H, W) class ids -> (H, W, 3) uint8 RGB."""
    return colormap()[labels]
