"""S5 -- natural-language querying over the map.

Split by construction:

  * Anything QUANTITATIVE (areas, counts, distances, adjacency, connectivity) is
    answered by code here. In-context arithmetic over a big table is where LLMs
    reliably go wrong, and it is exactly the part that is easy to compute.
  * Anything INTERPRETIVE is handed to the LLM, together with the *narrowest*
    digest that can support the answer.

The naive query language below is deliberately tiny. It is not meant to be the
user interface -- it is the tool surface an LLM would call, and having it exist
first is what keeps the model out of the arithmetic.

    find <class> [minarea <m2>] [slope < N | slope > N] [near <class> <m>]
    count <class>
    area <class>
    corridor <from-class> [maxslope N] [vehicle wheeled|tracked|foot]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..index import Region, RegionIndex
from ..loader import LabelRaster
from ..taxonomy import BY_NAME, cid

# --- tunable heuristics (all guesses -- see docs/solutions/S5.md) ----------

# Trafficability above which a pixel counts as part of a movement corridor.
CORRIDOR_MIN_TRAFFIC = 0.45

# Corridors smaller than this are noise, not routes.
CORRIDOR_MIN_AREA_M2 = 500.0


@dataclass
class QueryResult:
    query: str
    kind: str
    rows: list[tuple]
    header: tuple[str, ...]
    scalar: float | None = None
    note: str = ""

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
        return "\n".join(lines)


class QueryError(ValueError):
    pass


def _need_class(tok: str) -> int:
    if tok not in BY_NAME:
        raise QueryError(f"unknown class {tok!r}; try `segmap legend`")
    return cid(tok)


def query(q: str, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    toks = q.split()
    if not toks:
        raise QueryError("empty query")
    verb, rest = toks[0].lower(), toks[1:]

    if verb in ("find", "count", "area"):
        return _find(q, verb, rest, raster, ridx)
    if verb == "corridor":
        return _corridor(q, rest, raster, ridx)
    raise QueryError(f"unknown verb {verb!r}; supported: find, count, area, corridor")


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

    if slope_op:
        op, v = slope_op
        regions = [r for r in regions
                   if (r.mean_slope < v if op == "<" else r.mean_slope > v)]

    note = ""
    if near:
        other, dist_m = near
        mask = raster.labels == other
        if not mask.any():
            note = f"no {rest[rest.index('near') + 1]} in this tile -- 'near' matched nothing"
            regions = []
        else:
            dt = ndi.distance_transform_edt(~mask) * raster.gsd
            regions = [r for r in regions
                       if dt[int(r.centroid[0]), int(r.centroid[1])] <= dist_m]

    if verb == "count":
        return QueryResult(q, "count", [], (), float(len(regions)), note)
    if verb == "area":
        return QueryResult(q, "area_m2", [], (), sum(r.area_m2 for r in regions), note)

    rows = [(r.id, r.class_name, f"{r.area_m2:.0f}", f"{r.mean_slope:.1f}",
             f"{r.centroid[0]:.0f}", f"{r.centroid[1]:.0f}")
            for r in sorted(regions, key=lambda r: -r.area_m2)]
    return QueryResult(q, f"find ({len(rows)} regions)", rows,
                       ("rid", "class", "area_m2", "slope", "cy", "cx"), None, note)


def _corridor(q, rest, raster: LabelRaster, ridx: RegionIndex) -> QueryResult:
    """Connected trafficable ground reachable from a seed class -- the
    'where can a vehicle actually get to' question."""
    from .s4_products import VEHICLES, trafficability

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

    traf = trafficability(raster, vehicle=vehicle)
    passable = traf >= CORRIDOR_MIN_TRAFFIC
    comp, n = ndi.label(passable, structure=np.ones((3, 3)))
    if n == 0:
        return QueryResult(q, "corridor", [], (), 0.0,
                           f"no ground passable to {vehicle} at threshold "
                           f"{CORRIDOR_MIN_TRAFFIC}")

    seed_ids = set(np.unique(comp[raster.labels == seed])) - {0}
    px_m2 = raster.pixel_area_m2
    sizes = np.bincount(comp.ravel())
    rows = []
    for c in sorted(seed_ids, key=lambda c: -sizes[c]):
        a = sizes[c] * px_m2
        if a < CORRIDOR_MIN_AREA_M2:
            continue
        rows.append((int(c), f"{a:.0f}", f"{a / (raster.labels.size * px_m2):.1%}"))

    reachable = sum(sizes[c] for c in seed_ids) * px_m2
    total = raster.labels.size * px_m2
    return QueryResult(
        q, f"corridor ({vehicle}, seeded from {rest[0]})", rows,
        ("component", "area_m2", "tile_frac"), reachable,
        f"{reachable / total:.1%} of the tile is reachable from {rest[0]} "
        f"without leaving ground passable to a {vehicle} vehicle "
        f"(threshold {CORRIDOR_MIN_TRAFFIC}, slope limit "
        f"{VEHICLES[vehicle].max_slope_deg} deg)",
    )


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
