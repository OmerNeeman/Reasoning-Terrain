"""One object holding everything the OSM layer produces, and the cache in front of it.

Callers -- the CLI, the digests, S1 through S4 -- all want the same four things
(vectors, burned masks, road graph, block partition) built from the same
snapshot, and none of them should be deciding whether to hit the network.

What is cached and what is not, and why:

  the **block partition** is cached. `ndi.label` over the 1194 Mpx sinai mosaic
    plus the per-block zonal pass is the expensive step, it is entirely
    query-independent, and the block-id raster is exactly the kind of big array
    the region cache already memory-maps.
  the **burn** is not. It costs work proportional to the corridor area rather
    than to the raster -- 7.4 s for 255 roads and 4,361 building polygons on aza
    [measured] -- and caching it would mean writing a 1.2 GB mask per layer per
    AOI to save single-digit seconds.

The cache key carries the OSM snapshot's fingerprint, so a refetch that changes
nothing is a hit and one retagged street is a miss.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .. import cache as cache_mod
from .burn import AlignReport, BurnedOsm, align_report, burn
from .fetch import for_raster
from .partition import BlockIndex, RoadGraph, build_blocks, build_road_graph
from .vectors import OsmVectors

# Bump when a table in `tags.py` changes what a burn or a partition produces --
# a different half width, a different idea of what counts as a road. It is part
# of the block cache key, so old partitions miss rather than being reused under
# new rules.
TAGS_VERSION = 1

DEFAULT_LAYERS = ("road",)


@dataclass
class OsmLayer:
    vectors: OsmVectors
    burned: BurnedOsm
    graph: RoadGraph
    blocks: BlockIndex
    align: AlignReport | None = None

    @property
    def registered(self) -> bool:
        """Whether the two maps are aligned well enough to compare feature by
        feature. `None` (not checked) counts as yes; MISREGISTERED does not."""
        return self.align is None or self.align.verdict != "MISREGISTERED"

    def summary(self) -> str:
        parts = [self.vectors.summary(), self.burned.summary()]
        if self.align is not None:
            parts.append(self.align.render())
        parts.append(self.blocks.summary())
        if self.graph is not None:
            parts.append(f"# road graph: {len(self.graph.segments)} segments, "
                         f"{self.graph.n_junctions} junctions on this raster")
        return "\n".join(parts)


def build(raster, source=None, cache_dir=cache_mod.DEFAULT_CACHE_DIR,
          layers: tuple[str, ...] = DEFAULT_LAYERS,
          cut_layers: tuple[str, ...] = ("road",),
          include_foot: bool = False,
          refresh: bool = False, refresh_osm: bool = False,
          allow_network: bool = True, bbox=None, do_align: bool = True,
          quiet: bool = False, cache_source: dict | None = None) -> OsmLayer:
    """The whole layer for `raster`. The only OSM entry point other code uses."""
    want = tuple(dict.fromkeys(layers + tuple(cut_layers)))
    vectors = for_raster(raster, source=source, cache_dir=cache_dir,
                         refresh=refresh_osm, allow_network=allow_network,
                         bbox=bbox, quiet=quiet)
    burned = burn(raster, vectors, layers=want, include_foot_in_road=include_foot)
    graph = build_road_graph(raster, vectors, include_foot=include_foot)
    al = align_report(raster, burned) if (do_align and burned.has("road")) else None

    params = cache_mod.block_params(cut_layers, burned.vectors_fingerprint,
                                    include_foot=include_foot,
                                    tags_version=TAGS_VERSION)
    blocks = cache_mod.get(
        "blocks", raster, source=cache_source, params=params,
        cache_dir=cache_dir, refresh=refresh, quiet=quiet,
        build=lambda: build_blocks(raster, burned, vectors,
                                   cut_layers=tuple(cut_layers), graph=graph),
    )
    # A partition read back from cache has no graph attached (it is stored
    # alongside, and re-derived here anyway, which is cheap and keeps the two
    # in step).
    if blocks.graph is None:
        blocks.graph = graph

    if al is not None and al.verdict == "MISREGISTERED" and not quiet:
        print("# OSM: WARNING -- " + al.note, file=sys.stderr)
    return OsmLayer(vectors, burned, graph, blocks, al)
