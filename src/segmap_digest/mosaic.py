"""Stitch a directory of adjacent GeoTIFF tiles into one LabelRaster.

Regions are connected components, so analysing tiles one at a time cuts every
region at the tile seam: a single wadi crossing four tiles becomes four regions
with four wrong areas, four wrong perimeters and four wrong neighbour lists.
The mosaic is the honest unit of analysis.

Placement comes from the affine transforms, not from the filenames -- the
filename indices are a convenience, the georeference is the truth. Tiles must
share a CRS and a pixel size; anything else raises rather than being resampled,
because silently resampling a *label* raster invents classes at every boundary.

The mosaic is cropped to the bounding box of actual data. That drops no labelled
pixel -- only all-nodata margin, which for these crops is most of the extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys

from .loader import (LabelRaster, _check, class_map, class_map_from_tags,
                     geotiff_gsd, remap)


def tile_paths(directory: str | Path) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in (".tif", ".tiff"))


def _check_tile(raw: np.ndarray, path: Path) -> np.ndarray:
    """`loader._check`, with the offending tile named.

    The single-tile path already refuses a raster whose values are outside
    0..N_CLASSES-1 and tells the operator to supply `--classes`. The mosaic path
    used to skip that check entirely, which is how a sparse-wire-id tile with no
    mapping tag got as far as `digests.py`.
    """
    try:
        return _check(raw)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None


def load_mosaic(directory: str | Path, gsd: float | None = None,
                classes: str | Path | dict | None = None,
                crop_to_data: bool = True) -> LabelRaster:
    import rasterio

    paths = tile_paths(directory)
    if not paths:
        raise ValueError(
            f"{directory} contains no GeoTIFFs -- nothing to mosaic. "
            "(An AOI whose data has not landed yet is expected to be skipped, "
            "not to produce an empty report.)"
        )

    # Where each tile's id mapping came from. A mapping GIVEN on the command line
    # is one borrowed from somewhere else -- the aza drop is read with the sinai
    # drop's mapping, and until now nothing anywhere said so. Getting that wrong
    # renames every class on the map, so the borrowing is announced rather than
    # inferred from the absence of a complaint.
    given = class_map(classes)
    heads, tagged, borrowed, untranslated = [], [], [], []
    for p in paths:
        with rasterio.open(p) as src:
            own = class_map_from_tags(src.tags())
            heads.append({
                "path": p, "w": src.width, "h": src.height,
                "transform": src.transform, "crs": src.crs,
                "res": (src.transform.a, src.transform.e),
                "nodata": src.nodata, "gsd": geotiff_gsd(src),
                "mapping": given or own,
            })
        (tagged if own else borrowed if given else untranslated).append(p.name)
    if borrowed:
        src_name = classes if isinstance(classes, (str, Path)) else "the given dict"
        print(f"# id mapping: {len(borrowed)} of {len(paths)} tiles carry no "
              f"ID_TO_LABEL_MAPPING tag and are read with the mapping from "
              f"{src_name}. That mapping is from another export -- if it is the "
              f"wrong one, every class name on this AOI is wrong and nothing "
              f"downstream can tell. ({', '.join(borrowed[:3])}"
              f"{', ...' if len(borrowed) > 3 else ''})", file=sys.stderr)

    crss = {str(h["crs"]) for h in heads}
    if len(crss) > 1:
        raise ValueError(f"tiles do not share a CRS: {sorted(crss)}")
    res = {(round(h["res"][0], 12), round(h["res"][1], 12)) for h in heads}
    if len(res) > 1:
        raise ValueError(
            f"tiles do not share a pixel size: {sorted(res)}. Resampling a label "
            "raster to a common grid invents classes at every boundary; regrid "
            "upstream instead."
        )

    a, e = heads[0]["res"]
    origin_x = min(h["transform"].c for h in heads)
    origin_y = max(h["transform"].f for h in heads)

    placed = []
    for h in heads:
        col = int(round((h["transform"].c - origin_x) / a))
        row = int(round((h["transform"].f - origin_y) / e))
        placed.append((row, col, h))
    height = max(r + h["h"] for r, _, h in placed)
    width = max(c + h["w"] for _, c, h in placed)

    labels = np.zeros((height, width), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    for row, col, h in placed:
        with rasterio.open(h["path"]) as src:
            raw = src.read(1)
        # `_check` on the untranslated path, exactly as `loader._load_labels`
        # does for a single tile. Without it a tile whose ids are sparse wire ids
        # was passed through as if they were dense taxonomy ids and surfaced much
        # later as a bare `KeyError: np.int64(94)` out of `digests.py` -- three
        # modules from the file that caused it.
        block = (remap(raw, h["mapping"]) if h["mapping"]
                 else _check_tile(raw, h["path"]))
        labels[row:row + h["h"], col:col + h["w"]] = block
        if h["nodata"] is not None:
            block_valid = raw != raw.dtype.type(h["nodata"])
        else:
            block_valid = np.ones(raw.shape, dtype=bool)
        valid[row:row + h["h"], col:col + h["w"]] = block_valid
        del raw, block, block_valid

    from rasterio.transform import Affine

    transform = Affine(a, 0.0, origin_x, 0.0, e, origin_y)
    if crop_to_data and valid.any():
        rows = np.flatnonzero(valid.any(axis=1))
        cols = np.flatnonzero(valid.any(axis=0))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        labels = np.ascontiguousarray(labels[r0:r1, c0:c1])
        valid = np.ascontiguousarray(valid[r0:r1, c0:c1])
        transform = transform * Affine.translation(c0, r0)

    return LabelRaster(
        labels=labels,
        gsd=gsd or float(np.mean([h["gsd"] for h in heads])),
        transform=transform,
        crs=heads[0]["crs"],
        valid=valid if not valid.all() else None,
    )

