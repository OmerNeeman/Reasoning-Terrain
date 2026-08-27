"""The OSM layer: a second, independent map of the same ground.

RT's other reasoning is self-referential -- it checks the labels against their
own geometry, slope and neighbours. This package adds an outside opinion, for the
three things OSM maps well and a segmenter guesses at: where the roads are, where
the buildings are, and where the water is.

Two products come out of it, and they are used for different things:

  `partition` -- the ground split into BLOCKS bounded by the road network, plus
    the road graph whose intersections bound them. This is the reasoning unit
    the analyst already thinks in, and there are about a hundred of them per
    square kilometre instead of a hundred thousand connected components.
  `evidence` -- per-way and per-region agreement between the two maps, which is
    what S1 audits and what S2 adjudicates with.

The layer NEVER modifies the label raster. See docs/OSM.md, "Non-goals".
"""

# NB: the `burn` *module* is deliberately not shadowed by the `burn` *function*
# here -- `from segmap_digest.osm import burn` must give you the module, so
# callers import the function explicitly (`from .osm.burn import burn`).
from .burn import BurnedOsm, align_report  # noqa: F401
from .fetch import bbox_of_raster, for_raster, load_file  # noqa: F401
from .partition import Block, BlockIndex, RoadGraph, build_blocks, build_road_graph  # noqa: F401
from .vectors import OsmVectors, OsmWay  # noqa: F401

__all__ = [
    "OsmVectors", "OsmWay", "for_raster", "load_file", "bbox_of_raster",
    "BurnedOsm", "align_report",
    "build_blocks", "build_road_graph", "Block", "BlockIndex", "RoadGraph",
]
