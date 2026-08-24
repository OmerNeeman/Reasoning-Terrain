"""segmap_digest -- turn a 47-class Smart Terrain segmentation raster into
representations an LLM can reason over.

Quick start:

    from segmap_digest import synth, digests
    from segmap_digest.index import build_regions

    tile = synth.generate(size=1024, seed=7)     # or loader.load("tile.tif")
    print(digests.l0_histogram(tile))
    print(digests.l2_regions(build_regions(tile), limit=50))
"""

from . import audit, digests, loader, synth, taxonomy, tools  # noqa: F401
from .index import build_chips, build_regions  # noqa: F401
from .loader import LabelRaster, load  # noqa: F401

__version__ = "0.1.0"
