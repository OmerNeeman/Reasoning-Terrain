"""Load a label raster from disk, plus a colormap for eyeballing it.

Accepts GeoTIFF (via rasterio, keeps the georeference), .npy, a single-band
PNG, or a directory of adjacent GeoTIFFs (mosaicked by affine transform).

Two things real Smart Terrain exports do that the synthetic fixture never did:

  Wire ids are not taxonomy ids. The product emits sparse ids in 0..241 and
  ships the mapping as an `ID_TO_LABEL_MAPPING` GeoTIFF tag. We translate to
  the dense internal ids in `taxonomy` *by name*, and refuse to guess when a
  name is unknown.

  Nodata is real. Cropped tiles are mostly nodata, and the nodata value is 0 --
  the same integer as `Unclassified`. Counting those pixels as a class would
  make half of an arid AOI read as "unclassified terrain". `LabelRaster.valid`
  carries the mask; nodata is treated exactly like outside-the-tile everywhere.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .taxonomy import BY_NAME, N_CLASSES, SUPERCLASS_OF


@dataclass
class LabelRaster:
    labels: np.ndarray            # (H, W) uint8, dense taxonomy ids 0..N_CLASSES-1
    gsd: float = 0.3              # metres per pixel
    transform: object | None = None   # rasterio Affine, when available
    crs: object | None = None
    dem: np.ndarray | None = None     # (H, W) float32 elevation, metres
    valid: np.ndarray | None = None   # (H, W) bool; None = every pixel is data
    # Non-empty only when this raster is a *window* onto a larger AOI (see
    # `crop_to_max_mpx`). It says so in words, and everything that renders an
    # answer is expected to repeat it: a subset answer presented as an AOI answer
    # is a wrong answer.
    subset_note: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape

    @property
    def pixel_area_m2(self) -> float:
        return self.gsd * self.gsd

    @property
    def n_valid(self) -> int:
        """Pixels carrying an actual class. The honest denominator."""
        return int(self.valid.sum()) if self.valid is not None else self.labels.size

    @property
    def nodata_frac(self) -> float:
        return 1.0 - self.n_valid / max(self.labels.size, 1)


def load(path: str | Path, gsd: float | None = None,
         dem: str | Path | None = None,
         classes: str | Path | dict | None = None) -> LabelRaster:
    path = Path(path)
    if path.is_dir():
        from .mosaic import load_mosaic

        raster = load_mosaic(path, gsd=gsd, classes=classes)
    else:
        raster = _load_labels(path, gsd, classes)
    if dem is not None:
        raster.dem = load_dem(dem, raster.shape)
    return raster


# --- honest subsetting ------------------------------------------------------
#
# The mosaic is 1.2 Gpx and its index costs half an hour. Someone who wants to
# poke at the thing for ten minutes needs a smaller unit of work, and there are
# exactly two ways to make one:
#
#   crop   -- fewer pixels, all of them real, covering part of the AOI.
#   stride -- the whole AOI, at a coarser grid.
#
# This crops. Striding a *label* raster is nearest-neighbour downsampling: it
# deletes every feature narrower than the stride (roads, wadis, walls are 1-4 px
# here), merges regions that never touched, and changes every area and region
# count by an amount nobody can predict. A crop changes exactly one thing --
# which ground you are looking at -- and says so.

def crop_to_max_mpx(raster: LabelRaster, max_mpx: float) -> LabelRaster:
    """A centred window of at most `max_mpx` megapixels, at full resolution.

    Returns the raster unchanged when it already fits. The returned raster
    carries a `subset_note` stating the window, its share of the extent, and its
    share of the AOI's *classified* pixels -- which is the number that matters,
    and which a centre crop of a sparse mosaic can easily make small.
    """
    if max_mpx is None or max_mpx <= 0:
        raise ValueError("--max-mpx needs a positive number of megapixels")
    h, w = raster.shape
    total = h * w
    budget = int(max_mpx * 1e6)
    if total <= budget:
        return raster

    scale = (budget / total) ** 0.5
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    r0, c0 = (h - nh) // 2, (w - nw) // 2
    r1, c1 = r0 + nh, c0 + nw

    valid = raster.valid[r0:r1, c0:c1] if raster.valid is not None else None
    kept_valid = int(valid.sum()) if valid is not None else nh * nw
    aoi_valid = raster.n_valid

    transform = raster.transform
    if transform is not None:
        try:
            from rasterio.transform import Affine

            transform = transform * Affine.translation(c0, r0)
        except ImportError:  # pragma: no cover - georeference is provenance only
            transform = None

    note = (
        f"SUBSET: --max-mpx {max_mpx:g} cropped this AOI to rows {r0}:{r1}, "
        f"cols {c0}:{c1} -- {nh}x{nw} px = {nh * nw / 1e6:.1f} Mpx of "
        f"{total / 1e6:.1f} Mpx ({nh * nw / total:.1%} of the extent), holding "
        f"{kept_valid / max(aoi_valid, 1):.1%} of the AOI's classified pixels. "
        f"Full resolution, no downsampling. Every number derived from this "
        f"raster describes that window only and must not be reported as an "
        f"AOI-wide figure."
    )
    return LabelRaster(
        labels=np.ascontiguousarray(raster.labels[r0:r1, c0:c1]),
        gsd=raster.gsd,
        transform=transform,
        crs=raster.crs,
        dem=(np.ascontiguousarray(raster.dem[r0:r1, c0:c1])
             if raster.dem is not None else None),
        valid=np.ascontiguousarray(valid) if valid is not None else None,
        subset_note=note,
    )


def crop_spec(max_mpx: float | None) -> dict | None:
    """The subset as it appears in a cache key. Same crop -> same entry."""
    return None if not max_mpx else {"kind": "centre-crop", "max_mpx": float(max_mpx)}


def load_dem(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Elevation in metres, resampled to the label grid if needed.

    Nearest-neighbour resampling is fine for a POC but will quantise slope on a
    coarse DEM -- if the DEM is much coarser than the labels, say so rather than
    trusting the slope numbers.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
    else:
        arr = np.load(path).astype(np.float32)

    if arr.shape != shape:
        ry = (np.arange(shape[0]) * arr.shape[0] / shape[0]).astype(int)
        rx = (np.arange(shape[1]) * arr.shape[1] / shape[1]).astype(int)
        arr = arr[np.clip(ry, 0, arr.shape[0] - 1)][:, np.clip(rx, 0, arr.shape[1] - 1)]
    return arr


def _load_labels(path: str | Path, gsd: float | None = None,
                 classes: str | Path | dict | None = None) -> LabelRaster:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            raw = src.read(1)
            tr, crs = src.transform, src.crs
            gsd_hdr = geotiff_gsd(src)
            mapping = class_map(classes) or class_map_from_tags(src.tags())
            valid = _valid_mask(raw, src.nodata)
        labels = remap(raw, mapping) if mapping else _check(raw)
        return LabelRaster(labels, gsd or gsd_hdr, tr, crs, valid=valid)

    mapping = class_map(classes)
    if suffix == ".npy":
        raw = np.load(path)
    elif suffix in (".png", ".bmp"):
        from PIL import Image

        raw = np.array(Image.open(path))
        if raw.ndim == 3:
            raise ValueError(
                f"{path} is a {raw.shape[2]}-band image. This tool wants a single-band "
                "label raster, not a colourised one -- recovering class ids from RGB is "
                "lossy and ambiguous."
            )
    else:
        raise ValueError(f"unsupported label raster format: {suffix}")
    return LabelRaster(remap(raw, mapping) if mapping else _check(raw), gsd or 0.3)


def geotiff_gsd(src) -> float:
    """Metres per pixel from the header, converting from degrees if geographic.

    A 0.5 m/px export in EPSG:4326 has a transform of ~5e-06 -- taking `abs(a)`
    at face value silently makes every area in the report 1e10 times too small.
    """
    tr = src.transform
    if tr is None:
        return 0.3
    px_x, px_y = abs(tr.a), abs(tr.e)
    if src.crs is not None and src.crs.is_geographic:
        lat = np.radians((src.bounds.top + src.bounds.bottom) / 2.0)
        px_x *= 111_320.0 * np.cos(lat)
        px_y *= 110_540.0
    return float((px_x + px_y) / 2.0)


def _valid_mask(raw: np.ndarray, nodata) -> np.ndarray | None:
    if nodata is None:
        return None
    mask = raw != raw.dtype.type(nodata)
    return None if mask.all() else mask


# --- wire ids -> dense taxonomy ids ----------------------------------------

def class_map(classes: str | Path | dict | None) -> dict[int, str] | None:
    """Read an explicit `{wire_id: class_name}` mapping (JSON file or dict)."""
    if classes is None:
        return None
    if isinstance(classes, dict):
        raw = classes
    else:
        raw = json.loads(Path(classes).read_text())
        if isinstance(raw, dict) and "ID_TO_LABEL_MAPPING" in raw:
            raw = raw["ID_TO_LABEL_MAPPING"]
        if isinstance(raw, str):
            raw = json.loads(raw)
    return {int(k): str(v) for k, v in raw.items()}


def class_map_from_tags(tags: dict) -> dict[int, str] | None:
    """The mapping the segmenter itself wrote into the GeoTIFF, if present."""
    blob = tags.get("ID_TO_LABEL_MAPPING")
    return class_map(json.loads(blob)) if blob else None


def wire_lut(mapping: dict[int, str]) -> np.ndarray:
    """(max_wire_id+1,) uint8 LUT from wire ids to dense taxonomy ids.

    Refuses on any name the taxonomy does not define. Quietly folding an unknown
    class into Unclassified would delete real terrain from every downstream
    number, which is exactly the kind of silent loss this tool is not allowed to
    have.
    """
    unknown = sorted({n for n in mapping.values() if n not in BY_NAME})
    if unknown:
        raise ValueError(
            f"class names in the raster's id mapping are not in taxonomy.py: "
            f"{unknown}. Add them rather than dropping the pixels."
        )
    lut = np.zeros(max(mapping) + 1, dtype=np.uint8)
    for wire, name in mapping.items():
        lut[wire] = BY_NAME[name].id
    return lut


def remap(raw: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"label raster must be 2-D, got shape {raw.shape}")
    lut = wire_lut(mapping)
    seen = np.unique(raw)
    stray = seen[(seen < 0) | (seen >= len(lut))]
    if stray.size:
        raise ValueError(
            f"raster contains values with no entry in its id mapping: "
            f"{stray.tolist()[:10]}"
        )
    return lut[raw]


def _check(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"label raster must be 2-D, got shape {labels.shape}")
    lo, hi = int(labels.min()), int(labels.max())
    if lo < 0 or hi >= N_CLASSES:
        raise ValueError(
            f"label values must be in 0..{N_CLASSES - 1}, got {lo}..{hi}. "
            "If this is a real Smart Terrain export its ids are sparse wire ids -- "
            "supply the id->name mapping (--classes) instead of reinterpreting them."
        )
    return labels.astype(np.uint8)


# --- colormap --------------------------------------------------------------
# Hue by superclass so a glance separates rock from vegetation from built,
# lightness varying within a superclass. This is for humans; feeding a
# colourised label map to a VLM and asking it to read exact colours off a
# 47-entry legend is a task VLMs are bad at.

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


def colormap_hex() -> list[str]:
    return ["#%02x%02x%02x" % tuple(int(v) for v in row) for row in colormap()]


def colourise(labels: np.ndarray) -> np.ndarray:
    """(H, W) class ids -> (H, W, 3) uint8 RGB."""
    return colormap()[labels]
