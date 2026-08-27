"""The reasoning unit: ground bounded by streets, and the graph that bounds it.

RT's only spatial unit until now was the connected component -- 95,170 of them
on the aza AOI and 133,976 on sinai [measured]. Nothing about "region 41,882" is
addressable by a human, and a per-region worklist over a city is not a worklist,
it is a spreadsheet nobody opens.

A block is the other kind of unit: the ground enclosed by the road network. On
aza that is **99 blocks, median 3,896 m2** [measured] -- three orders of
magnitude fewer objects, each of which a person can point at and, where OSM has
street names, actually name.

The partition is deliberately dumb. It is the complement of the road corridor,
labelled by connected components, and that is all. It is *not* an attempt to
recover cadastral parcels, and it makes no claim that a block is homogeneous --
what is inside a block is exactly the question the ST labels answer.

Two honesty requirements, both learned from the audit's false-positive history:

  Blocks tile the ground exactly. Every valid pixel is in one block or in the
    corridor. Nothing is dropped for being small, because a "partition" with
    holes in it silently changes every area share computed from it.
  The degenerate case is reported, not hidden. Point this at open desert where
    OSM has three tracks and the answer is one block covering the AOI. That is
    the correct answer and the output says so, rather than presenting one
    enormous block as a spatial analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..taxonomy import BY_ID, N_CLASSES
from .burn import BurnedOsm
from .vectors import OsmVectors

# Coordinate rounding used to detect shared vertices when the input carries no
# node ids (a GeoJSON export). 1e-6 degrees is about 0.11 m -- tight enough that
# two different streets never collide, loose enough to survive a float round
# trip through a GIS.
COORD_QUANT = 1_000_000

# A block below this is not a place, it is a slot between two ways whose buffers
# nearly met. Kept in the partition (the tiling invariant), but excluded from
# rankings and from the digest's default view.
MIN_INTERESTING_AREA_M2 = 200.0

# Road corridor share below which the network cannot have partitioned anything.
# Reported as degenerate rather than presented as a partition.
DEGENERATE_ROAD_SHARE = 0.005
# ...or when the largest block is essentially the whole AOI.
DEGENERATE_DOMINANCE = 0.90

# Dilation, in pixels, used to decide which ways bound a block. One pixel is
# enough when the corridor is the block's own boundary; two absorbs the
# rasterisation jitter where three ways meet.
BOUNDARY_TOUCH_PX = 2

# Street names listed in a block's label. More than three is not a name.
NAME_LIMIT = 3


# --- the road graph -------------------------------------------------------

@dataclass
class Intersection:
    node_id: int
    row: float
    col: float
    way_ids: tuple[int, ...]
    names: tuple[str, ...]
    # The query bbox is padded, so some junctions sit off the raster. They are
    # kept, because they are what makes a segment's topology correct, and
    # excluded from anything user-facing: "junction at row -903" is not a place
    # in this AOI.
    inside: bool = True

    @property
    def degree(self) -> int:
        return len(self.way_ids)

    @property
    def unique_names(self) -> tuple[str, ...]:
        seen, out = set(), []
        for n in self.names:
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return tuple(out)

    @property
    def label(self) -> str:
        named = list(self.unique_names)
        if len(named) >= 2:
            return " x ".join(sorted(set(named))[:2])
        if named:
            return f"{named[0]} junction"
        return f"junction of {self.degree} ways"


@dataclass
class RoadSegment:
    id: int
    way_id: int
    name: str
    grade: str
    surface: str
    node_a: int                 # index into RoadGraph.intersections, -1 = dangling
    node_b: int
    length_m: float


@dataclass
class RoadGraph:
    intersections: list[Intersection]
    segments: list[RoadSegment]
    node_source: str = "osm-node-ids"      # or "coordinate-matching"

    @property
    def junctions(self) -> list[Intersection]:
        """Junctions on this raster. Degree >= 2 and inside the grid."""
        return [i for i in self.intersections if i.degree >= 2 and i.inside]

    @property
    def n_junctions(self) -> int:
        return len(self.junctions)

    def render(self, limit: int | None = 25) -> str:
        js = sorted(self.junctions, key=lambda i: -i.degree)
        total_m = sum(s.length_m for s in self.segments)
        off = sum(1 for i in self.intersections if i.degree >= 2 and not i.inside)
        lines = [
            f"# OSM road graph: {len(self.segments)} segments, "
            f"{total_m / 1000:.1f} km, {self.n_junctions} junctions on this "
            f"raster (node identity from {self.node_source})",
            "node\tdeg\trow\tcol\tstreets",
        ]
        shown = js if limit is None else js[:limit]
        for i in shown:
            lines.append(f"{i.node_id}\t{i.degree}\t{i.row:.0f}\t{i.col:.0f}\t"
                         f"{', '.join(i.unique_names) or '-'}")
        if limit is not None and len(js) > limit:
            lines.append(f"# NOTE: {len(js) - limit} junctions omitted")
        if off:
            lines.append(f"# NOTE: {off} further junctions lie outside the raster "
                         f"(the OSM bbox is padded so edge blocks close)")
        return "\n".join(lines)


def build_road_graph(raster, vectors: OsmVectors,
                     include_foot: bool = False) -> RoadGraph:
    """Intersections are nodes shared by two or more road ways.

    Shared *node ids* are the real signal and Overpass `out geom` carries them.
    Where they are absent -- a GeoJSON export drops them -- fall back to matching
    quantised coordinates, and say which was used, because the fallback misses a
    junction whose ways were digitised a decimetre apart.
    """
    from .burn import to_pixels

    roads = vectors.roads(include_foot=include_foot)
    have_nodes = sum(1 for w in roads if w.nodes) > len(roads) // 2
    source = "osm-node-ids" if have_nodes else "coordinate-matching"

    # key -> (lon, lat, [way ids], [names])
    seen: dict[object, list] = {}
    per_way_keys: list[tuple] = []
    for w in roads:
        if have_nodes and len(w.nodes) == len(w.lon):
            keys = list(w.nodes)
        else:
            keys = [(int(round(x * COORD_QUANT)), int(round(y * COORD_QUANT)))
                    for x, y in zip(w.lon, w.lat)]
        per_way_keys.append(tuple(keys))
        for k, x, y in zip(keys, w.lon, w.lat):
            ent = seen.setdefault(k, [float(x), float(y), [], []])
            if w.id not in ent[2]:
                ent[2].append(w.id)
                ent[3].append(w.name)

    junction_keys = {k for k, v in seen.items() if len(v[2]) >= 2}
    if not junction_keys:
        return RoadGraph([], [], source)

    ordered = sorted(junction_keys, key=lambda k: (seen[k][1], seen[k][0]))
    lons = np.array([seen[k][0] for k in ordered])
    lats = np.array([seen[k][1] for k in ordered])
    rows, cols = to_pixels(raster, lons, lats)
    index_of = {k: i for i, k in enumerate(ordered)}
    h, w = raster.shape
    intersections = [
        Intersection(node_id=(k if isinstance(k, int) else -(i + 1)),
                     row=float(rows[i]), col=float(cols[i]),
                     way_ids=tuple(seen[k][2]),
                     names=tuple(seen[k][3]),
                     inside=bool(0 <= rows[i] < h and 0 <= cols[i] < w))
        for i, k in enumerate(ordered)
    ]

    # Segments: cut each way at every junction it passes through.
    segments: list[RoadSegment] = []
    sid = 0
    for w, keys in zip(roads, per_way_keys):
        rr, cc = to_pixels(raster, w.lon, w.lat)
        cut = [0] + [i for i, k in enumerate(keys)
                     if k in index_of and 0 < i < len(keys) - 1] + [len(keys) - 1]
        for a, b in zip(cut, cut[1:]):
            if b <= a:
                continue
            seg_len = float(np.hypot(np.diff(rr[a:b + 1]), np.diff(cc[a:b + 1])).sum())
            segments.append(RoadSegment(
                id=sid, way_id=w.id, name=w.name, grade=w.grade or "",
                surface=w.surface,
                node_a=index_of.get(keys[a], -1), node_b=index_of.get(keys[b], -1),
                length_m=seg_len * raster.gsd,
            ))
            sid += 1
    return RoadGraph(intersections, segments, source)


# --- blocks ---------------------------------------------------------------

@dataclass
class Block:
    id: int
    area_px: int
    area_m2: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    hist: np.ndarray                       # (N_CLASSES,) pixel counts
    n_valid_px: int
    boundary_way_ids: tuple[int, ...] = ()
    street_names: tuple[str, ...] = ()
    grades: tuple[str, ...] = ()
    junction_node_ids: tuple[int, ...] = ()

    @property
    def frac(self) -> np.ndarray:
        return self.hist / max(self.n_valid_px, 1)

    @property
    def entropy(self) -> float:
        f = self.frac
        nz = f[f > 0]
        return float(-(nz * np.log2(nz)).sum())

    @property
    def n_classes(self) -> int:
        return int((self.hist > 0).sum())

    def dominant(self, k: int = 3) -> list[tuple[str, float]]:
        if self.n_valid_px == 0:
            return []
        order = np.argsort(self.hist)[::-1][:k]
        return [(BY_ID[int(c)].name, float(self.hist[c] / self.n_valid_px))
                for c in order if self.hist[c] > 0]

    @property
    def label(self) -> str:
        """What a person would call this block. Names first, then grades."""
        named = [n for n in self.street_names if n][:NAME_LIMIT]
        if named:
            return "bounded by " + ", ".join(named)
        if self.grades:
            counts: dict[str, int] = {}
            for g in self.grades:
                counts[g] = counts.get(g, 0) + 1
            parts = [f"{v} {k}" for k, v in
                     sorted(counts.items(), key=lambda kv: -kv[1])[:2]]
            return "bounded by " + " + ".join(parts) + " ways"
        return "not bounded by any mapped way"


@dataclass
class BlockIndex:
    blocks: list[Block]
    label_array: np.ndarray                # (H, W) int32, 0 = corridor or nodata
    gsd: float
    road_px: int
    valid_px: int
    degenerate: bool = False
    note: str = ""
    cut_layers: tuple[str, ...] = ("road",)
    graph: RoadGraph | None = None
    vectors_fingerprint: str = ""

    def get(self, bid: int) -> Block:
        return self.blocks[bid - 1]

    @property
    def block_px(self) -> int:
        return sum(b.area_px for b in self.blocks)

    def interesting(self, min_area_m2: float = MIN_INTERESTING_AREA_M2) -> list[Block]:
        return [b for b in self.blocks if b.area_m2 >= min_area_m2]

    def summary(self) -> str:
        areas = np.array([b.area_m2 for b in self.blocks]) if self.blocks else np.zeros(1)
        keep = self.interesting()
        share = sum(b.area_m2 for b in keep) / max(areas.sum(), 1e-9)
        named = sum(1 for b in self.blocks if any(b.street_names))
        lines = [
            f"# {len(self.blocks)} blocks from the OSM road network "
            f"(cut on: {', '.join(self.cut_layers)})",
            f"# corridor {self.road_px:,} px "
            f"({100 * self.road_px / max(self.valid_px, 1):.1f}% of classified "
            f"ground); blocks hold the rest",
            f"# area m2: max {areas.max():,.0f}  median {np.median(areas):,.0f}  "
            f"min {areas.min():,.1f};  {len(keep)} blocks >= "
            f"{MIN_INTERESTING_AREA_M2:g} m2 hold {share:.1%} of block area",
            f"# {named} blocks are bounded by at least one named street",
        ]
        if self.degenerate:
            lines.append(f"# DEGENERATE: {self.note}")
        return "\n".join(lines)

    def render(self, limit: int | None = 25, min_area_m2: float | None = None) -> str:
        rows = self.interesting(min_area_m2 if min_area_m2 is not None
                                else MIN_INTERESTING_AREA_M2)
        rows.sort(key=lambda b: -b.area_m2)
        shown = rows if limit is None else rows[:limit]
        out = [self.summary(),
               "block\tarea_m2\tclasses\tentropy\ttop3\tbounded_by"]
        for b in shown:
            top = " ".join(f"{n}:{f:.2f}" for n, f in b.dominant(3))
            out.append(f"{b.id}\t{b.area_m2:.0f}\t{b.n_classes}\t{b.entropy:.2f}\t"
                       f"{top}\t{b.label}")
        if limit is not None and len(rows) > limit:
            out.append(f"# NOTE: {len(rows) - limit} blocks omitted (of "
                       f"{len(self.blocks)} total, "
                       f"{len(self.blocks) - len(rows)} below "
                       f"{MIN_INTERESTING_AREA_M2:g} m2)")
        return "\n".join(out)


def build_blocks(raster, burned: BurnedOsm, vectors: OsmVectors | None = None,
                 cut_layers: tuple[str, ...] = ("road",),
                 graph: RoadGraph | None = None) -> BlockIndex:
    """Complement of the corridor, labelled. `cut_layers` are the barriers that
    separate blocks -- roads always, and optionally railways, walls or water.
    """
    h, w = raster.shape
    cut = np.zeros((h, w), dtype=bool)
    used = []
    for name in cut_layers:
        if burned.has(name):
            cut |= burned.mask(name)
            used.append(name)
    if not used:
        raise ValueError(f"none of the cut layers {cut_layers} were burned; "
                         f"burned layers are {sorted(burned.masks)}")

    free = ~cut
    if raster.valid is not None:
        free &= raster.valid
    lab, n = ndi.label(free, structure=np.ones((3, 3), dtype=bool))
    del free

    valid_px = raster.n_valid
    road_px = int(cut.sum() if raster.valid is None
                  else (cut & raster.valid).sum())
    if n == 0:
        return BlockIndex([], lab.astype(np.int32), raster.gsd, road_px, valid_px,
                          degenerate=True,
                          note="the OSM corridor covers every classified pixel; "
                               "there is no ground left to partition",
                          cut_layers=tuple(used), graph=graph,
                          vectors_fingerprint=burned.vectors_fingerprint)

    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    slices = ndi.find_objects(lab)
    centres = ndi.center_of_mass(np.ones(lab.shape, dtype=np.uint8), lab,
                                 range(1, n + 1))
    px_area = raster.gsd * raster.gsd
    blocks: list[Block] = []
    for i in range(n):
        sl = slices[i]
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        sub = lab[r0:r1, c0:c1] == (i + 1)
        sub_labels = raster.labels[r0:r1, c0:c1]
        if raster.valid is not None:
            sub = sub & raster.valid[r0:r1, c0:c1]
        npx = int(sub.sum())
        hist = (np.bincount(sub_labels[sub].ravel(), minlength=N_CLASSES)
                if npx else np.zeros(N_CLASSES, dtype=np.int64))
        blocks.append(Block(
            id=i + 1, area_px=int(areas[i]), area_m2=float(areas[i]) * px_area,
            bbox=(r0, c0, r1, c1),
            centroid=(float(centres[i][0]), float(centres[i][1])),
            hist=hist.astype(np.int64), n_valid_px=npx,
        ))

    idx = BlockIndex(blocks, lab.astype(np.int32), raster.gsd, road_px, valid_px,
                     cut_layers=tuple(used), graph=graph,
                     vectors_fingerprint=burned.vectors_fingerprint)

    biggest = max((b.area_px for b in blocks), default=0)
    total_free = sum(b.area_px for b in blocks) or 1
    if road_px / max(valid_px, 1) < DEGENERATE_ROAD_SHARE:
        idx.degenerate = True
        idx.note = (f"the road corridor is only "
                    f"{100 * road_px / max(valid_px, 1):.2f}% of classified ground "
                    f"-- OSM has almost no road network here, so this is not a "
                    f"partition of the AOI, it is the AOI with a few tracks "
                    f"scratched out of it")
    elif biggest / total_free > DEGENERATE_DOMINANCE:
        idx.degenerate = True
        idx.note = (f"the largest block holds {100 * biggest / total_free:.0f}% of "
                    f"the unpaved ground: the mapped roads do not close any "
                    f"circuits here, so block ids carry no spatial meaning "
                    f"beyond 'inside the AOI'")

    if burned.by_kind("road"):
        attach_boundaries(idx, burned, graph)
    return idx


def attach_boundaries(idx: BlockIndex, burned: BurnedOsm,
                      graph: RoadGraph | None = None) -> None:
    """Which ways bound each block, by re-realising each way's footprint.

    The alternative is an int32 way-id raster, which is 4.8 GB on the sinai
    mosaic to answer a question about a few thousand ways. Re-burning is the same
    arithmetic a second time over a few percent of the grid.
    """
    lab = idx.label_array
    h, w = lab.shape
    per_block: dict[int, dict[int, int]] = {}
    names: dict[int, dict[str, int]] = {}
    grades: dict[int, list[str]] = {}
    struct = ndi.generate_binary_structure(2, 2)

    for fp in burned.by_kind("road"):
        local = fp.local_mask((h, w))
        if local is None:
            continue
        r0, c0, r1, c1 = fp.bbox
        # Grow into the block on either side of the corridor.
        grown = ndi.binary_dilation(local, struct, iterations=BOUNDARY_TOUCH_PX)
        sub = lab[r0:r1, c0:c1]
        touched = np.unique(sub[grown & (sub > 0)])
        for bid in touched.tolist():
            n_px = int((sub == bid).sum())
            per_block.setdefault(bid, {})[fp.way_id] = n_px
            if fp.label and not fp.label.startswith("unnamed"):
                d = names.setdefault(bid, {})
                d[fp.label] = d.get(fp.label, 0) + n_px
            if fp.grade:
                grades.setdefault(bid, []).append(fp.grade)

    junctions: dict[int, set[int]] = {}
    if graph is not None:
        for i in graph.intersections:
            r, c = int(round(i.row)), int(round(i.col))
            if i.inside and 0 <= r < h and 0 <= c < w:
                r0, r1 = max(r - 4, 0), min(r + 5, h)
                c0, c1 = max(c - 4, 0), min(c + 5, w)
                for bid in np.unique(lab[r0:r1, c0:c1]).tolist():
                    if bid > 0:
                        junctions.setdefault(bid, set()).add(i.node_id)

    for b in idx.blocks:
        ways = per_block.get(b.id, {})
        b.boundary_way_ids = tuple(sorted(ways, key=lambda k: -ways[k]))
        nm = names.get(b.id, {})
        b.street_names = tuple(sorted(nm, key=lambda k: -nm[k]))
        b.grades = tuple(grades.get(b.id, ()))
        b.junction_node_ids = tuple(sorted(junctions.get(b.id, ())))


def block_of(idx: BlockIndex, row: int, col: int) -> Block | None:
    """Which block contains a pixel. 0 means the corridor itself."""
    bid = int(idx.label_array[row, col])
    return idx.get(bid) if bid > 0 else None


def blocks_of_region(idx: BlockIndex, ridx, region_id: int,
                     max_blocks: int = 3) -> list[tuple[int, float]]:
    """Which blocks an ST region falls in, by area share. A region straddling
    a street belongs to both, and that is itself informative."""
    r = ridx.get(region_id)
    r0, c0, r1, c1 = r.bbox
    sub_r = ridx.label_array[r0:r1, c0:c1] == region_id
    sub_b = idx.label_array[r0:r1, c0:c1][sub_r]
    if sub_b.size == 0:
        return []
    ids, counts = np.unique(sub_b, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = counts.sum()
    return [(int(ids[i]), float(counts[i] / total)) for i in order[:max_blocks]]


# --- addressing a window into the partition --------------------------------
#
# D1 (owner, 2026-08-26): blocks are built on the AOI and *addressed* per tile.
# At aza's 0.122 m/px a 1024 px tile is 125 m of ground -- a third of a city
# block -- so partitioning per tile would invent boundaries at the seam that
# exist nowhere on the ground. This is the other half of that decision: given a
# window, which blocks does it touch, and how much of each.
#
# Two shares, always both, because they answer different questions and reporting
# one as the other is how a subset answer gets presented as an AOI answer:
#
#   share_of_window -- how much of THIS TILE is block 84. Use it to describe the
#                      tile: "60% of this window is block 84".
#   share_of_block  -- how much of BLOCK 84 is in this tile. Use it to decide
#                      whether any statistic of block 84 may be quoted from this
#                      window at all: at 4%, none of them may.

# A block occupying less of the window than this is listed but marked marginal;
# below it, a "this tile is in block N" statement is noise.
WINDOW_MIN_SHARE = 0.01

# Below this share of the block, nothing about the block as a whole may be
# computed from the window -- its area, its composition and its bounding streets
# all describe ground the window cannot see.
BLOCK_QUOTABLE_SHARE = 0.90


@dataclass
class WindowBlock:
    block: Block
    px_in_window: int
    share_of_window: float
    share_of_block: float

    @property
    def truncated(self) -> bool:
        """The window holds only part of this block."""
        return self.share_of_block < 0.999

    @property
    def quotable(self) -> bool:
        """Whether block-level figures may be quoted from this window alone."""
        return self.share_of_block >= BLOCK_QUOTABLE_SHARE


def window_of_tile(mosaic, tile) -> tuple[int, int, int, int]:
    """Where a single tile sits inside the mosaic it belongs to, in mosaic pixels.

    `tile` is a path or a LabelRaster. Both rasters must be in the same CRS at
    the same pixel size; anything else raises, for the same reason `mosaic.py`
    raises rather than resampling -- a resampled label raster invents classes at
    every boundary, and a mis-placed window silently answers about the wrong
    ground.
    """
    if hasattr(tile, "transform"):
        tt, t_crs, t_shape = tile.transform, tile.crs, tile.shape
    else:
        # Header only. Reading the tile's PIXELS to find out where it is would
        # need its class mapping (`--classes`), which has nothing to do with
        # geometry -- and fails loudly on a real export whose wire ids are
        # sparse. The affine and the CRS are in the header.
        try:
            import rasterio
        except ImportError as exc:      # pragma: no cover
            raise ValueError("locating a tile inside its mosaic needs rasterio "
                             "(pip install -e '.[geo]')") from exc
        with rasterio.open(tile) as src:
            tt, t_crs, t_shape = src.transform, src.crs, (src.height, src.width)

    class _T:                            # just enough to read like a raster
        transform, crs, shape = tt, t_crs, t_shape

    t_ras = _T()
    mt = mosaic.transform
    if mt is None or tt is None:
        raise ValueError("both the mosaic and the tile need an affine transform "
                         "to be aligned; .npy and PNG inputs carry none")
    if str(mosaic.crs) != str(t_ras.crs):
        raise ValueError(f"tile CRS {t_ras.crs} does not match the mosaic's "
                         f"{mosaic.crs}")
    if abs(mt.a - tt.a) > abs(mt.a) * 1e-6 or abs(mt.e - tt.e) > abs(mt.e) * 1e-6:
        raise ValueError(f"tile pixel size ({tt.a}, {tt.e}) does not match the "
                         f"mosaic's ({mt.a}, {mt.e})")
    c0 = int(round((tt.c - mt.c) / mt.a))
    r0 = int(round((tt.f - mt.f) / mt.e))
    h, w = t_shape
    return (r0, c0, r0 + h, c0 + w)


def blocks_in_window(idx: BlockIndex, window: tuple[int, int, int, int],
                     raster=None,
                     min_share: float = WINDOW_MIN_SHARE) -> list[WindowBlock]:
    """Which blocks a window touches, with both shares, largest first."""
    r0, c0, r1, c1 = window
    h, w = idx.label_array.shape
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, h), min(c1, w)
    if r1 <= r0 or c1 <= c0:
        return []
    sub = np.asarray(idx.label_array[r0:r1, c0:c1])
    if raster is not None and raster.valid is not None and raster.shape == (h, w):
        sub = np.where(raster.valid[r0:r1, c0:c1], sub, 0)
    ids, counts = np.unique(sub, return_counts=True)
    total = int(counts.sum())
    out: list[WindowBlock] = []
    for bid, n in zip(ids.tolist(), counts.tolist()):
        if bid <= 0:
            continue
        b = idx.get(bid)
        share_w = n / max(total, 1)
        if share_w < min_share:
            continue
        out.append(WindowBlock(b, int(n), share_w, n / max(b.area_px, 1)))
    out.sort(key=lambda wb: -wb.share_of_window)
    return out


def render_window(idx: BlockIndex, window: tuple[int, int, int, int],
                  raster=None, limit: int | None = 12, label: str = "") -> str:
    """The block address of one tile. Says what may and may not be quoted."""
    rows = blocks_in_window(idx, window, raster=raster)
    h, w = idx.label_array.shape
    ar0, ac0, ar1, ac1 = window
    r0, c0 = max(ar0, 0), max(ac0, 0)
    r1, c1 = min(ar1, h), min(ac1, w)
    px = max(r1 - r0, 0) * max(c1 - c0, 0)
    gsd = idx.gsd
    clipped = (r0, c0, r1, c1) != (ar0, ac0, ar1, ac1)
    head = [
        f"# window {label or ''}rows {r0}:{r1}, cols {c0}:{c1} -- "
        f"{r1 - r0}x{c1 - c0} px = {px * gsd * gsd / 1e4:.2f} ha of the AOI",
    ]
    if clipped:
        head.append(
            f"# NOTE: the requested window was rows {ar0}:{ar1}, cols "
            f"{ac0}:{ac1} and hangs over the edge of the AOI; everything below "
            f"describes the {px * gsd * gsd / 1e4:.2f} ha that overlap it")
    head += [
        f"# touches {len(rows)} of the AOI's {len(idx.blocks)} blocks. Blocks are "
        f"built on the whole AOI and addressed here (docs/OSM-INGESTION.md, D1), "
        f"so a block is one object with one area however many tiles see it.",
    ]
    if not rows:
        head.append("# no block covers this window: it is entirely road corridor, "
                    "nodata, or outside the AOI")
        return "\n".join(head)
    corridor = 1.0 - sum(wb.share_of_window for wb in rows)
    head.append(f"# {corridor:.1%} of the window is road corridor or nodata")
    head.append("block\tof_window\tof_block\tquotable\tarea_m2\ttop3\tbounded_by")
    shown = rows if limit is None else rows[:limit]
    for wb in shown:
        b = wb.block
        top = " ".join(f"{n}:{f:.2f}" for n, f in b.dominant(3))
        head.append(
            f"{b.id}\t{wb.share_of_window:.1%}\t{wb.share_of_block:.1%}\t"
            f"{'yes' if wb.quotable else 'NO -- partial'}\t{b.area_m2:.0f}\t"
            f"{top}\t{b.label}")
    if limit is not None and len(rows) > limit:
        head.append(f"# NOTE: {len(rows) - limit} smaller blocks omitted")
    partial = [wb for wb in shown if not wb.quotable]
    if partial:
        head.append(
            f"# NOTE: {len(partial)} of the blocks listed "
            f"{'extends' if len(partial) == 1 else 'extend'} beyond this window. "
            f"Their `area_m2`, `top3` and `bounded_by` describe the WHOLE block, "
            f"including ground this window cannot see; do not report them as "
            f"measurements of this tile.")
    return "\n".join(head)
