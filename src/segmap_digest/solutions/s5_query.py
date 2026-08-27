"""S5 -- natural-language querying over the map.

Split by construction:

  * Anything QUANTITATIVE (areas, counts, distances, adjacency, connectivity) is
    answered by code here. In-context arithmetic over a big table is where LLMs
    reliably go wrong, and it is exactly the part that is easy to compute.
  * Anything INTERPRETIVE is handed to the LLM, together with the *narrowest*
    digest that can support the answer.

The naive query language below is deliberately tiny. It is not meant to be the
user interface -- it is the tool surface an LLM calls (see `tools.py`, which
compiles a tool call into one of these strings so that the arithmetic lives in
exactly one place).

    find <class> [minarea <m2>] [slope < N | slope > N] [near <class> <m>]
    count <class>
    area <class>
    distance <class> <class>
    describe
    corridor <from-class> [vehicle wheeled|tracked|foot]

Every result carries the region ids it was computed from. A number with no
provenance cannot be checked on the map, and an answer a user cannot check is
not much better than one the model made up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..index import Region, RegionIndex
from ..loader import LabelRaster
from ..taxonomy import BY_ID, BY_NAME, N_CLASSES, SUPERCLASS_OF, cid

# --- tunable heuristics (all guesses -- see docs/solutions/S5.md) ----------

# Trafficability above which a pixel counts as part of a movement corridor.
CORRIDOR_MIN_TRAFFIC = 0.45

# Corridors smaller than this are noise, not routes.
CORRIDOR_MIN_AREA_M2 = 500.0

# --- what a label raster cannot answer, whatever the question -------------
#
# The map is class membership per pixel and nothing else. These are the
# attributes people actually ask about that are simply not in it. Returning an
# empty table for them reads as "none found" -- a wrong answer, not a missing
# one -- so the query layer refuses in words instead. Keyed by the word that
# turns up in the request; the values are the sentence handed back.
UNANSWERABLE_CONCEPTS: dict[str, str] = {
    "fence": "there is no fence class; a fence is also sub-metre wide and would "
             "not survive segmentation at this GSD",
    "truck": "the taxonomy has exactly one vehicle class, Car. Vehicle *type* "
             "was never predicted",
    "tank": "the taxonomy has exactly one vehicle class, Car. Vehicle *type* "
            "was never predicted",
    "bus": "the taxonomy has exactly one vehicle class, Car. Vehicle *type* "
           "was never predicted",
    "vehicle": "the taxonomy has exactly one vehicle class, Car. Vehicle *type*, "
               "size and load are not in it",
    "school": "the taxonomy has one building class, House. Building *function* "
              "is not predicted and is not visible from nadir anyway",
    "hospital": "the taxonomy has one building class, House. Building *function* "
                "is not predicted",
    "factory": "the taxonomy has one building class, House. Building *function* "
               "is not predicted",
    "building": "the building class is called House; it carries no function, "
                "height, or occupancy",
    "colour": "no RGB imagery is involved anywhere in this pipeline -- the input "
              "is a label raster",
    "color": "no RGB imagery is involved anywhere in this pipeline -- the input "
             "is a label raster",
    "roof": "roof material and condition are not classes; only House is",
    "window": "sub-object detail is far below the class vocabulary",
    "people": "there is no person class",
    "person": "there is no person class",
    "height": "no DSM is attached; the map carries class membership, not elevation "
              "above ground",
}


def unanswerable_reason(token: str) -> str | None:
    """Why the map cannot answer for this word, or None if it is just a typo."""
    t = token.lower()
    for key, why in UNANSWERABLE_CONCEPTS.items():
        if key in t or t in key:
            return why
    return None


@dataclass
class QueryResult:
    query: str
    kind: str
    rows: list[tuple]
    header: tuple[str, ...]
    scalar: float | None = None
    note: str = ""
    # The evidence. `ids` is complete, not a sample -- whoever renders it caps it
    # and says how many were dropped.
    ids: list[int] = field(default_factory=list)
    id_kind: str = "region"      # region | component | class | none

    def render(self, limit: int = 25) -> str:
        lines = [f"# {self.kind}: {self.query}"]
        if self.note:
            lines.append(f"# {self.note}")
        if self.scalar is not None:
            lines.append(f"= {self.scalar:,.2f}")
        if self.rows:
            lines.append("\t".join(self.header))
            for r in self.rows[:limit]:
                lines.append("\t".join(str(x) for x in r))
            if len(self.rows) > limit:
                lines.append(f"# NOTE: {len(self.rows) - limit} rows omitted by limit={limit}")
        elif self.scalar is None:
            lines.append("# no matches")
        if self.ids:
            shown = self.ids[:limit]
            tail = (f" (+{len(self.ids) - limit} more)" if len(self.ids) > limit else "")
            lines.append(f"# evidence: {self.id_kind} ids "
                         f"{', '.join(str(i) for i in shown)}{tail}")
        return "\n".join(lines)


class QueryError(ValueError):
    pass


class Unanswerable(QueryError):
    """The query is well formed but the label raster cannot answer it at all.

    Distinct from a query that legitimately matched nothing: "no Houses here" is
    a finding, "the map does not know what a fence is" is not.
    """


def _need_class(tok: str) -> int:
    if tok not in BY_NAME:
        why = unanswerable_reason(tok)
        raise Unanswerable(
            f"{tok!r} is outside the 47-class vocabulary: "
            + (why or "no such class in the taxonomy; try `segmap legend`")
        )
    return cid(tok)


def _class_mask(raster: LabelRaster, class_id: int) -> np.ndarray:
    mask = raster.labels == class_id
    if raster.valid is not None:
        mask &= raster.valid
    return mask


def query(q: str, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    toks = q.split()
    if not toks:
        raise QueryError("empty query")
    verb, rest = toks[0].lower(), toks[1:]

    if verb in ("find", "count", "area"):
        return _find(q, verb, rest, raster, ridx)
    if verb == "corridor":
        return _corridor(q, rest, raster, ridx)
    if verb == "distance":
        return _distance(q, rest, raster, ridx)
    if verb == "describe":
        return _describe(q, rest, raster, ridx)
    raise QueryError(f"unknown verb {verb!r}; supported: find, count, area, "
                     f"distance, describe, corridor")


def _find(q, verb, rest, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    if not rest:
        raise QueryError(f"{verb} needs a class name")
    target = _need_class(rest[0])
    min_area = 0.0
    slope_op: tuple[str, float] | None = None
    near: tuple[int, float] | None = None

    i = 1
    while i < len(rest):
        t = rest[i].lower()
        if t == "minarea":
            min_area = float(rest[i + 1]); i += 2
        elif t == "slope":
            slope_op = (rest[i + 1], float(rest[i + 2])); i += 3
        elif t == "near":
            near = (_need_class(rest[i + 1]), float(rest[i + 2])); i += 3
        else:
            raise QueryError(f"unexpected token {rest[i]!r}")

    regions = [r for r in ridx.regions if r.class_id == target and r.area_m2 >= min_area]

    notes: list[str] = []
    if slope_op:
        # Without a DEM every mean_slope is 0.0 -- a valid-looking number that
        # means "unmeasured". Filtering on it would return a confident,
        # meaningless answer, so abstain instead.
        if not ridx.has_terrain:
            raise Unanswerable(
                "no DEM is attached to this tile, so every region's slope is "
                "unmeasured (stored as 0.0). A slope filter cannot be applied; "
                "attach a DEM with --dem and ask again"
            )
        op, v = slope_op
        regions = [r for r in regions
                   if (r.mean_slope < v if op == "<" else r.mean_slope > v)]

    if near:
        other, dist_m = near
        other_name = rest[rest.index("near") + 1]
        mask = _class_mask(raster, other)
        if not mask.any():
            notes.append(f"no {other_name} in this tile -- 'near' matched nothing")
            regions = []
        else:
            # Centroid distance, which is wrong for a long sinuous region: its
            # centroid can be far from every part of it. See open question 6.
            dt = ndi.distance_transform_edt(~mask) * raster.gsd
            regions = [r for r in regions
                       if dt[int(r.centroid[0]), int(r.centroid[1])] <= dist_m]
            notes.append(f"'near' measures centroid-to-nearest-{other_name} distance, "
                         f"not the region's closest approach")

    note = "; ".join(notes)
    ids = [r.id for r in sorted(regions, key=lambda r: -r.area_m2)]

    if verb == "count":
        return QueryResult(q, "count", [], (), float(len(regions)), note, ids)
    if verb == "area":
        return QueryResult(q, "area_m2", [], (), sum(r.area_m2 for r in regions),
                           note, ids)

    rows = [(r.id, r.class_name, f"{r.area_m2:.0f}", f"{r.mean_slope:.1f}",
             f"{r.centroid[0]:.0f}", f"{r.centroid[1]:.0f}")
            for r in sorted(regions, key=lambda r: -r.area_m2)]
    return QueryResult(q, f"find ({len(rows)} regions)", rows,
                       ("rid", "class", "area_m2", "slope", "cy", "cx"), None,
                       note, ids)


def _corridor(q, rest, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    """Connected trafficable ground reachable from a seed class -- the
    'where can a vehicle actually get to' question."""
    from .s4_products import VEHICLES, compute

    if not rest:
        raise QueryError("corridor needs a seed class, e.g. `corridor PavedRoad`")
    seed = _need_class(rest[0])
    vehicle = "wheeled"
    i = 1
    while i < len(rest):
        if rest[i].lower() == "vehicle":
            vehicle = rest[i + 1]; i += 2
        else:
            raise QueryError(f"unexpected token {rest[i]!r}")
    if vehicle not in VEHICLES:
        raise QueryError(f"unknown vehicle {vehicle!r}; try {list(VEHICLES)}")
    v = VEHICLES[vehicle]

    # via compute(), not trafficability() directly, so nodata scores 0 and the
    # corridor cannot bridge two real areas through unclassified ground.
    traf = compute(raster, "trafficability", vehicle=vehicle)
    passable = traf >= CORRIDOR_MIN_TRAFFIC

    # 8-connected labeling is single-pixel percolation: two areas joined by one
    # diagonal pixel (~half a metre at this GSD) would count as mutually
    # reachable, and no vehicle passes a half-metre gap. Open the mask (erode
    # then dilate, square element sized to the vehicle's width) so a corridor
    # must be at least vehicle-wide end to end. Foot (width 0) keeps pure
    # percolation, and the note always states which width was applied.
    structure_px = max(1, round(v.width_m / raster.gsd))
    if structure_px > 1:
        st = np.ones((structure_px, structure_px), dtype=bool)
        passable = ndi.binary_dilation(ndi.binary_erosion(passable, structure=st),
                                       structure=st)
        width_note = (f", passages narrower than the vehicle width {v.width_m} m "
                      f"({structure_px} px opening) removed")
    else:
        width_note = f", vehicle width {v.width_m} m -- no minimum passage width applied"

    comp, n = ndi.label(passable, structure=np.ones((3, 3)))
    if n == 0:
        return QueryResult(q, "corridor", [], (), 0.0,
                           f"no ground passable to {vehicle} at threshold "
                           f"{CORRIDOR_MIN_TRAFFIC}{width_note}")

    seed_mask = _class_mask(raster, seed)
    seed_ids = sorted(set(np.unique(comp[seed_mask]).tolist()) - {0})
    px_m2 = raster.pixel_area_m2
    sizes = np.bincount(comp.ravel())
    # Classified ground, not the whole extent: on a cropped tile those differ by
    # a third, and "96% of the tile is reachable" would be counting the margin.
    total = raster.n_valid * px_m2
    # A component id means nothing on a map by itself, so every row carries a
    # pixel a reviewer can go and look at.
    centres = ndi.center_of_mass(np.ones(comp.shape, dtype=np.uint8), comp,
                                 index=seed_ids) if seed_ids else []
    rows, kept, dropped_m2 = [], [], 0.0
    for c, (cy, cx) in sorted(zip(seed_ids, centres), key=lambda t: -sizes[t[0]]):
        a = sizes[c] * px_m2
        if a < CORRIDOR_MIN_AREA_M2:
            dropped_m2 += a
            continue
        rows.append((int(c), f"{a:.0f}", f"{a / total:.1%}", f"{cy:.0f}", f"{cx:.0f}"))
        kept.append(int(c))

    dropped = len(seed_ids) - len(kept)

    # A noise floor that removes EVERYTHING is not filtering noise, it is
    # deleting the answer. When no component clears `CORRIDOR_MIN_AREA_M2` the
    # floor is not applied at all: reporting "0.0% reachable" for ground that is
    # wholly reachable through pockets a little too small for the threshold is a
    # wrong answer wearing a precise number, and it is worse than the noise the
    # threshold exists to suppress.
    #
    # The threshold is absolute (500 m2) while the AOIs it runs over span four
    # orders of magnitude -- a 200 m2 test fixture and a 175 km2 mosaic -- so
    # this case is not exotic. Making the floor relative to the AOI would be the
    # deeper fix; it would also change every corridor answer on real data, so it
    # is left as an open question in the docs rather than smuggled in here.
    floor_applied = bool(kept) or not seed_ids
    if not floor_applied:
        kept = [int(c) for c in seed_ids]
        rows = [(int(c), f"{sizes[c] * px_m2:.0f}",
                 f"{sizes[c] * px_m2 / total:.1%}", f"{cy:.0f}", f"{cx:.0f}")
                for c, (cy, cx) in sorted(zip(seed_ids, centres),
                                          key=lambda t: -sizes[t[0]])]
        dropped, dropped_m2 = 0, 0.0

    # The headline scalar sums the KEPT components only: pockets dropped from
    # the table as noise must not sit inside the number the table is supposed
    # to substantiate. The note states what was excluded and how much.
    reachable = sum(sizes[c] for c in kept) * px_m2
    note = (f"{reachable / total:.1%} of the classified area is reachable from {rest[0]} "
            f"without leaving ground passable to a {vehicle} vehicle "
            f"(threshold {CORRIDOR_MIN_TRAFFIC}, slope limit "
            f"{v.max_slope_deg} deg{width_note})")
    if dropped:
        note += (f"; {dropped} reachable pocket(s) totalling {dropped_m2:.0f} m2 "
                 f"below {CORRIDOR_MIN_AREA_M2:.0f} m2 excluded as noise from "
                 f"the table and the headline figure")
    elif not floor_applied:
        note += (f"; every reachable component is below the "
                 f"{CORRIDOR_MIN_AREA_M2:.0f} m2 noise floor, so the floor was "
                 f"NOT applied -- it would have reported 0.0% for ground that is "
                 f"reachable. Read these areas as small rather than as noise")
    return QueryResult(
        q, f"corridor ({vehicle}, seeded from {rest[0]})", rows,
        ("component", "area_m2", "tile_frac", "cy", "cx"), reachable, note,
        kept, "component",
    )


def _distance(q, rest, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    """Minimum distance between any pixel of A and any pixel of B.

    Pixel-to-pixel, not centroid-to-centroid: this is the number someone means
    when they ask how close the houses get to the wadi. One full distance
    transform per call -- this is the expensive verb.
    """
    if len(rest) < 2:
        raise QueryError("distance needs two class names, e.g. `distance House Water`")
    if len(rest) > 2:
        raise QueryError(f"unexpected token {rest[2]!r}")
    a, b = _need_class(rest[0]), _need_class(rest[1])
    mask_a, mask_b = _class_mask(raster, a), _class_mask(raster, b)
    for mask, name in ((mask_a, rest[0]), (mask_b, rest[1])):
        if not mask.any():
            return QueryResult(q, "distance_m", [], (), None,
                               f"no {name} in this tile, so the pair does not exist "
                               f"here and there is no distance to report")
    if a == b:
        return QueryResult(q, "distance_m", [], (), 0.0,
                           f"both arguments are {rest[0]}; distance to itself is 0")

    dt, ind = ndi.distance_transform_edt(~mask_a, return_indices=True)
    masked = np.where(mask_b, dt, np.inf)
    ry, rx = np.unravel_index(int(np.argmin(masked)), masked.shape)
    ay, ax = int(ind[0, ry, rx]), int(ind[1, ry, rx])
    metres = float(dt[ry, rx]) * raster.gsd

    lab = ridx.label_array
    rid_a, rid_b = int(lab[ay, ax]), int(lab[ry, rx])
    rows = [(rest[0], rid_a or "-", ay, ax, rest[1], rid_b or "-", ry, rx,
             f"{metres:.2f}")]
    ids = [r for r in (rid_a, rid_b) if r]
    note = "closest approach, pixel to pixel"
    if len(ids) < 2:
        note += ("; one endpoint falls in a region below the index's min-area cut, "
                 "so it has no region id")
    return QueryResult(q, "distance_m", rows,
                       ("class_a", "rid_a", "ay", "ax", "class_b", "rid_b", "by", "bx",
                        "metres"), metres, note, ids)


def _describe(q, rest, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    """Composition of the tile: every class actually present, with its share."""
    if rest:
        raise QueryError(f"describe takes no arguments; got {rest[0]!r}")
    labels = raster.labels
    counted = labels if raster.valid is None else labels[raster.valid]
    hist = np.bincount(counted.ravel(), minlength=N_CLASSES)
    total = max(int(counted.size), 1)
    n_regions: dict[int, int] = {}
    for r in ridx.regions:
        n_regions[r.class_id] = n_regions.get(r.class_id, 0) + 1

    rows, ids = [], []
    for c in np.argsort(-hist):
        if hist[c] == 0:
            continue
        rows.append((int(c), BY_ID[c].name, SUPERCLASS_OF[c],
                     f"{hist[c] / total:.4f}",
                     f"{hist[c] * raster.pixel_area_m2:.0f}",
                     n_regions.get(int(c), 0)))
        ids.append(int(c))
    note = (f"fractions are over {total} classified pixels "
            f"({raster.nodata_frac:.1%} of the extent is nodata and excluded); "
            f"region counts are after the index's min-area cut, so small "
            f"specks are missing from them")
    return QueryResult(q, f"describe ({len(rows)} classes present)", rows,
                       ("class_id", "name", "superclass", "frac", "area_m2",
                        "n_regions"), None, note, ids, "class")


def llm_handoff(q: str, raster: LabelRaster, ridx: RegionIndex) -> tuple[str, str]:
    """For interpretive questions: pick the narrowest digest that can support an
    answer, and return (digest, question). Deliberately crude routing -- the
    point is that routing exists, not that these keywords are right."""
    from .. import digests

    ql = q.lower()
    if any(w in ql for w in ("composition", "how much", "overall", "summarise", "summarize")):
        return digests.l0_histogram(raster), q
    if any(w in ql for w in ("where", "layout", "north", "south", "east", "west")):
        return digests.l0_histogram(raster) + "\n\n" + digests.l1_grid(raster), q
    if any(w in ql for w in ("misclassif", "wrong", "suspect", "audit", "error")):
        from ..audit import audit, to_tsv
        return to_tsv(audit(ridx), limit=60), q
    return digests.l2_regions(ridx, limit=300), q
