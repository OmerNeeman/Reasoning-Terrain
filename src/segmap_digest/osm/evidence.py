"""What OSM says about one ST region, and how much that is worth per candidate.

S2 adjudicates a suspect label by scoring a shortlist on evidence the segmenter
never used: its neighbours, its slope, its shape. All three come out of the same
raster that produced the suspect label in the first place. This module adds the
one kind of evidence that does not: an independent map.

Three rules it follows, and the third is the one that matters:

  Silence is not a no. A region with no OSM feature anywhere near it gets
    `None`, not a low score, and S2's arithmetic then falls back exactly to what
    it does without this module. OSM's coverage is volunteered and uneven; "no
    building is mapped here" is not a statement that no building is here.
  The axis decides the strength, not the layer. `trust.py` holds the owner's
    policy: OSM is the reference for what is *there* and what it *is*; ST is the
    latest word on what the ground *looks like now*. So a mapped corridor under
    a polygon labelled `Rendzina` is strong evidence (existence -- OSM's call),
    and a mapped building over a polygon labelled `MaralBadlands` is weak
    evidence (state -- ST saw it more recently, and a demolished building is
    exactly what that looks like). The second case is the one that inverted when
    the policy was set; before it, this module would have relabelled rubble as
    House.
  Reliability is per layer and it is a guess. `tags.RELIABILITY` scales every
    score here on top of the axis, so believing OSM more or less is one table
    edit, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..taxonomy import BY_ID, cid
from . import tags as T
from . import trust
from .layer import OsmLayer

# --- thresholds (guesses) -------------------------------------------------

# Ring around a region's bounding box, in metres, searched for OSM features.
# Wider than a road half-width so that a region *beside* a street still sees it.
NEIGHBOURHOOD_M = 25.0

# Overlap share above which a layer is taken to be describing this region.
STRONG_OVERLAP = 0.50
WEAK_OVERLAP = 0.15

# The reference term's contribution to S2's total, applied only when this module
# returns a score at all. The other three terms are renormalised to 1 - this, so
# a silent OSM layer leaves S2's arithmetic bit-identical to what it was before
# there was an OSM layer at all.
W_REFERENCE = 0.30

# Support and contradiction now come from `trust.SUPPORT` / `trust.CONTRADICT`,
# keyed on which axis the evidence is about. Kept as names here only so the
# call sites read as sentences rather than as dictionary lookups.
def _support(axis: str) -> float:
    return trust.support(axis)


def _contra(axis: str) -> float:
    return trust.contradict(axis)

# Classes that being inside a built-up landuse polygon supports, mildly.
BUILT_CLASSES = ("House", "Pavement", "BrickWall", "Car", "PavedRoad", "Clutter")
# Classes that a mapped building footprint argues against: undisturbed ground
# cannot be under a roof.
NATURAL_GROUPS = ("soil", "rock", "vegetation", "agriculture", "water")


@dataclass
class RegionOsmContext:
    """Everything the OSM layer has to say about one ST region."""
    region_id: int
    area_px: int
    road_overlap: float = 0.0
    building_overlap: float = 0.0
    water_overlap: float = 0.0
    landuse_overlap: float = 0.0
    dist_to_road_m: float = float("inf")
    block_id: int = 0
    block_label: str = ""
    ways: list[tuple] = field(default_factory=list)   # (label, grade, surface, expected)

    @property
    def silent(self) -> bool:
        """No OSM feature near this region at all -- so OSM has said nothing.

        Note the asymmetry with `dist_to_road_m`: a region 200 m from the nearest
        street in a well-mapped city is a real statement (this is not a street);
        a region in an AOI where OSM has mapped nothing is not. The caller
        decides which it has, from `BlockIndex.degenerate` and the way counts.
        """
        return (self.road_overlap == 0.0 and self.building_overlap == 0.0
                and self.water_overlap == 0.0 and self.landuse_overlap == 0.0
                and not np.isfinite(self.dist_to_road_m))

    def describe(self) -> str:
        bits = []
        if self.road_overlap:
            bits.append(f"{self.road_overlap:.0%} inside an OSM road corridor")
        if self.ways:
            lab, grade, surf, exp = self.ways[0]
            bits.append(f"nearest way: {lab}"
                        + (f" [{grade}" + (f", surface={surf}" if surf else "") + "]"
                           if grade else "")
                        + (f" -> expects {exp}" if exp else ""))
        if self.building_overlap:
            bits.append(f"{self.building_overlap:.0%} inside an OSM building")
        if self.water_overlap:
            bits.append(f"{self.water_overlap:.0%} inside OSM water")
        if self.landuse_overlap:
            bits.append(f"{self.landuse_overlap:.0%} inside built-up landuse")
        if np.isfinite(self.dist_to_road_m) and not self.road_overlap:
            bits.append(f"{self.dist_to_road_m:.0f} m from the nearest mapped road")
        if self.block_label:
            bits.append(f"block {self.block_id} ({self.block_label})")
        return "; ".join(bits) or "OSM has nothing mapped in this neighbourhood"


def region_context(ridx, raster, layer: OsmLayer, region_id: int) -> RegionOsmContext:
    """Local, not global: everything is measured inside the region's own bbox
    plus a ring, so this costs the same on a 1 Gpx mosaic as on a tile."""
    r = ridx.get(region_id)
    h, w = raster.shape
    pad = max(int(round(NEIGHBOURHOOD_M / raster.gsd)), 1)
    r0, c0, r1, c1 = r.bbox
    p0, q0 = max(r0 - pad, 0), max(c0 - pad, 0)
    p1, q1 = min(r1 + pad, h), min(c1 + pad, w)

    own = np.asarray(ridx.label_array[r0:r1, c0:c1]) == region_id
    n = max(int(own.sum()), 1)
    ctx = RegionOsmContext(region_id=region_id, area_px=n)

    def overlap(name: str) -> float:
        if not layer.burned.has(name):
            return 0.0
        return float((layer.burned.mask(name)[r0:r1, c0:c1] & own).sum()) / n

    ctx.road_overlap = overlap("road")
    ctx.building_overlap = overlap("building")
    ctx.water_overlap = max(overlap("water"), overlap("flow"))
    ctx.landuse_overlap = overlap("built_landuse")

    if layer.burned.has("road"):
        ring = layer.burned.mask("road")[p0:p1, q0:q1]
        if ring.any():
            d = ndi.distance_transform_edt(~ring) * raster.gsd
            sub = d[r0 - p0:r1 - p0, c0 - q0:c1 - q0][own]
            ctx.dist_to_road_m = float(sub.min()) if sub.size else float("inf")

    # Which ways are in this neighbourhood, nearest-by-footprint-overlap first,
    # then by bbox proximity. `expected` is the whole reason to name them.
    cands = []
    for fp in layer.burned.by_kind("road"):
        f0, g0, f1, g1 = fp.bbox
        if f1 <= p0 or f0 >= p1 or g1 <= q0 or g0 >= q1:
            continue
        tags = next((wy.tags for wy in layer.vectors.ways if wy.id == fp.way_id), {})
        expected, _why = T.expected_road_class(tags)
        # Overlap of this way's footprint with the region, as a sort key.
        local = fp.local_mask((h, w))
        share = 0.0
        if local is not None:
            i0, j0 = max(f0, r0), max(g0, c0)
            i1, j1 = min(f1, r1), min(g1, c1)
            if i1 > i0 and j1 > j0:
                share = float((local[i0 - f0:i1 - f0, j0 - g0:j1 - g0]
                               & own[i0 - r0:i1 - r0, j0 - c0:j1 - c0]).sum()) / n
        cands.append((share, fp.label, fp.grade or "", fp.surface, expected or ""))
    cands.sort(key=lambda t: -t[0])
    ctx.ways = [(lab, gr, su, ex) for _s, lab, gr, su, ex in cands[:3]]

    if layer.blocks.blocks:
        rr = int(round(r.centroid[0]))
        cc = int(round(r.centroid[1]))
        if 0 <= rr < h and 0 <= cc < w:
            bid = int(layer.blocks.label_array[rr, cc])
            if bid > 0:
                ctx.block_id = bid
                ctx.block_label = layer.blocks.get(bid).label
    return ctx


def reference_score(ctx: RegionOsmContext, cand_id: int,
                    superclass_of=None) -> tuple[float | None, list[str]]:
    """(score in [0,1], notes) for one candidate class -- or (None, notes).

    None means the reference layer is not competent to speak about this
    candidate here, and the caller must fall back to its own three terms.
    """
    from ..taxonomy import SUPERCLASS_OF

    sc = (superclass_of or SUPERCLASS_OF)
    name = BY_ID[cand_id].name
    notes: list[str] = []
    if ctx.silent:
        return None, ["OSM is silent in this neighbourhood -- no reference "
                      "evidence either way"]

    rel_road = T.RELIABILITY.get("road", T.RELIABILITY_DEFAULT)
    rel_bld = T.RELIABILITY.get("building", T.RELIABILITY_DEFAULT)

    # --- roads -----------------------------------------------------------
    if name in T.ROAD_CLASSES:
        # Two different questions, two different axes, and conflating them is the
        # easy mistake here. "Is this ground a road at all" is EXISTENCE and OSM
        # is the reference. "Which grade of road is it" is STATE -- a street can
        # be resurfaced or degraded after somebody tagged it -- and there ST is
        # the later observation.
        expected = next((e for _l, _g, _s, e in ctx.ways if e), "")
        if ctx.road_overlap >= WEAK_OVERLAP:
            if expected == name:
                notes.append(f"OSM maps a way here and its surface tag expects "
                             f"{name} ({ctx.road_overlap:.0%} overlap)")
                return _blend(_support(trust.AXIS_EXISTENCE),
                              T.SURFACE_RELIABILITY), notes
            if expected:
                notes.append(f"OSM maps a way here -- so this IS road ground -- "
                             f"but its surface tag expects {expected}, not "
                             f"{name}; ST is the later observation of the surface")
                return _blend(_contra(trust.AXIS_STATE),
                              T.SURFACE_RELIABILITY), notes
            notes.append(f"OSM maps a road here ({ctx.road_overlap:.0%} overlap), "
                         f"with no surface tag to say which grade")
            return _blend(_support(trust.AXIS_EXISTENCE), rel_road * 0.9), notes
        if np.isfinite(ctx.dist_to_road_m) and ctx.dist_to_road_m > NEIGHBOURHOOD_M:
            notes.append(f"nearest mapped road is {ctx.dist_to_road_m:.0f} m away")
            return _blend(_contra(trust.AXIS_EXISTENCE), rel_road * 0.6), notes
        return None, ["OSM maps roads nearby but none over this region"]

    # --- buildings -------------------------------------------------------
    if name == "House":
        # STATE, not existence: OSM says a building was mapped here, ST says what
        # the ground looks like now. Deliberately too weak to clear
        # SWITCH_MARGIN on its own -- it puts House on the shortlist and argues
        # for it; it does not get to overturn a later observation.
        if ctx.building_overlap >= WEAK_OVERLAP:
            notes.append(f"{ctx.building_overlap:.0%} of this region is inside an "
                         f"OSM building footprint (reference map; ST is later)")
            return _blend(_support(trust.AXIS_STATE), rel_bld), notes
        if ctx.landuse_overlap >= STRONG_OVERLAP:
            notes.append("inside built-up landuse, but no mapped footprint")
            return _blend(0.58, rel_bld * 0.5), notes
        return None, ["no OSM footprint here; OSM building coverage is uneven"]

    if name == "Water":
        if ctx.water_overlap >= WEAK_OVERLAP:
            notes.append(f"{ctx.water_overlap:.0%} inside an OSM water feature")
            return _blend(_support(trust.AXIS_STATE),
                          T.RELIABILITY.get("water", 0.5)), notes
        return None, ["no OSM water feature here; seasonal water is "
                      "systematically unmapped"]

    # --- everything else -------------------------------------------------
    # A mapped building or a road corridor is a statement that the ground is
    # covered. That argues against a natural-surface label, and it is the
    # strongest thing this module says.
    if sc.get(cand_id) in NATURAL_GROUPS:
        if ctx.road_overlap >= STRONG_OVERLAP:
            # EXISTENCE: the street network is the thing OSM maps best, and a
            # polygon of soil sitting inside a mapped corridor is the strongest
            # statement this module makes.
            notes.append(f"{ctx.road_overlap:.0%} of this region is inside a "
                         f"mapped road corridor")
            return _blend(_contra(trust.AXIS_EXISTENCE),
                          T.RELIABILITY.get("road", 0.75)), notes
        if ctx.building_overlap >= STRONG_OVERLAP:
            # STATE, and weak on purpose: bare ground where OSM has a footprint
            # is what a demolished building looks like, and ST saw it later.
            # Flagged, not overruled -- S1 raises it as a change candidate.
            notes.append(f"{ctx.building_overlap:.0%} of this region is under a "
                         f"mapped building footprint -- either unlabelled built "
                         f"ground or a building that is no longer standing")
            return _blend(_contra(trust.AXIS_STATE), rel_bld), notes
        return None, []

    if name in BUILT_CLASSES:
        if ctx.landuse_overlap >= WEAK_OVERLAP or ctx.road_overlap >= WEAK_OVERLAP:
            notes.append("inside built-up or road-adjacent ground per OSM")
            return _blend(0.62, T.RELIABILITY.get("built_landuse", 0.45)), notes
        return None, []
    return None, []


def _blend(score: float, reliability: float) -> float:
    """Pull a verdict towards neutral in proportion to how much this layer is
    believed. Reliability 1.0 keeps it; 0.0 makes it say nothing."""
    return 0.5 + (score - 0.5) * max(0.0, min(1.0, reliability))


def suggested_candidates(ctx: RegionOsmContext) -> list[int]:
    """Classes the reference layer thinks are in play, whatever `class_distance`
    says. A `Rendzina` polygon sitting on a way tagged `surface=asphalt` needs
    `PavedRoad` on the shortlist, and no taxonomy distance will put it there."""
    out: list[int] = []
    if ctx.road_overlap >= WEAK_OVERLAP:
        expected = next((e for _l, _g, _s, e in ctx.ways if e), "")
        out += [cid(expected)] if expected else [cid(n) for n in T.ROAD_CLASSES]
    if ctx.building_overlap >= WEAK_OVERLAP:
        out.append(cid("House"))
    if ctx.water_overlap >= WEAK_OVERLAP:
        out.append(cid("Water"))
    return sorted(set(out))
