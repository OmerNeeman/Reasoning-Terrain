"""A PNG of the partition, because a partition is a thing you check by looking.

Every number the OSM layer produces rests on one assumption -- that the vectors
land on the same ground the labels describe -- and `align_report` can only tell
you about a *uniform* shift. A rotation, a wrong tile in the mosaic, or a
half-tile offset in one corner all pass the shift test and are obvious in one
glance at the image.

Downsampled by striding, on purpose. This is the one place in RT where striding
a label raster is the right thing to do, because the output is for a human eye
rather than for an area computation; `loader.crop_to_max_mpx` exists precisely
because striding is wrong everywhere else.
"""

from __future__ import annotations

import numpy as np

from ..loader import colourise

# Longest side of the written PNG. 2048 keeps a 1 Gpx mosaic legible on screen
# without writing a 60 MB image.
MAX_SIDE = 2048

# Corridor and boundary colours, chosen to be visible over the class colourmap
# in both directions -- the corridor is drawn as a dark stroke and the block
# outlines as a bright one.
ROAD_RGB = (20, 20, 24)
BLOCK_EDGE_RGB = (255, 235, 120)
BUILDING_RGB = (200, 60, 200)


def write(raster, layer, path: str, max_side: int = MAX_SIDE,
          show_buildings: bool = True) -> str:
    from PIL import Image

    h, w = raster.shape
    step = max(int(np.ceil(max(h, w) / max_side)), 1)
    labels = raster.labels[::step, ::step]
    rgb = colourise(labels)

    if raster.valid is not None:
        rgb[~raster.valid[::step, ::step]] = (30, 30, 30)

    if show_buildings and layer.burned.has("building"):
        b = layer.burned.mask("building")[::step, ::step]
        rgb[b] = np.array(BUILDING_RGB, dtype=rgb.dtype)

    # The corridor is blended, not painted over: the whole point of looking at
    # this image is to see whether ST's own road pixels lie UNDER the OSM
    # corridor, and a solid stroke hides exactly the evidence being checked.
    road = layer.burned.mask("road")[::step, ::step]
    rgb[road] = (0.45 * rgb[road].astype(np.float32)
                 + 0.55 * np.array(ROAD_RGB, dtype=np.float32)).astype(rgb.dtype)

    lab = np.asarray(layer.blocks.label_array[::step, ::step])
    edge = np.zeros(lab.shape, dtype=bool)
    edge[:-1, :] |= (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] > 0)
    edge[:, :-1] |= (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] > 0)
    rgb[edge] = np.array(BLOCK_EDGE_RGB, dtype=rgb.dtype)

    Image.fromarray(rgb).save(path)
    return path
