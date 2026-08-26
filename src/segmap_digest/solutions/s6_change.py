"""S6 -- change reasoning between two dates.

A pixel-diff of two label maps is mostly noise and phenology. The value is in
classifying each transition, because the same raw difference means completely
different things:

    GreenGrassland -> DryGrassland      season, not change
    DirtRoad       -> PavedRoad         real change: infrastructure upgrade
    Batha          -> Garigue           succession, or just a boundary wobble
    Batha          -> House             real change: construction
    Limestone*     -> Dolomite*         neither -- a lithology label flip that
                                        cannot physically happen

That last row is the one a naive change detector reports as a landslide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..loader import LabelRaster
from ..taxonomy import (
    BY_ID,
    DEGRADATION_SERIES,
    N_CLASSES,
    ROAD_SERIES,
    SUPERCLASS_OF,
    cid,
    class_distance,
)

# --- tunable heuristics (all guesses -- see docs/solutions/S6.md) ----------

# Seasonal pairs: a transition either way is phenology, not change.
SEASONAL_PAIRS = (
    ("GreenGrassland", "DryGrassland"),
    ("IrrigatedField", "DryGrassland"),
)

# Transitions covering less than this are treated as boundary noise.
MIN_EVENT_AREA_M2 = 20.0

# A change component whose shape is this compact and this rectangular reads as
# construction rather than natural change.
CONSTRUCTION_MIN_COMPACTNESS = 0.4

CATEGORIES = (
    "phenology",        # season / moisture, same underlying surface
    "succession",       # movement along an ordinal series
    "construction",     # something built
    "demolition",       # something removed
    "infrastructure",   # road grade change
    "impossible",       # lithology flip -- a label error, not a change
    "real-change",      # everything else that is semantically large
    "noise",            # small / adjacent-class wobble
)


def _seasonal() -> set[tuple[int, int]]:
    out = set()
    for a, b in SEASONAL_PAIRS:
        out.add((cid(a), cid(b)))
        out.add((cid(b), cid(a)))
    return out


_SEASONAL = _seasonal()
_BUILT = {"House", "Pavement", "BrickWall", "PavedRoad"}


def classify_transition(a: int, b: int) -> tuple[str, str]:
    """(category, why) for a single class transition."""
    if a == b:
        return "noise", "no change"
    na, nb = BY_ID[a].name, BY_ID[b].name

    if (a, b) in _SEASONAL:
        return "phenology", f"{na}->{nb} is a seasonal expression of the same surface"

    da, db = BY_ID[a], BY_ID[b]
    if da.lithology and db.lithology and da.lithology != db.lithology:
        return ("impossible",
                f"bedrock cannot change from {da.lithology} to {db.lithology}; "
                f"this is a label flip, not a change on the ground")

    for series in (DEGRADATION_SERIES, ROAD_SERIES):
        if na in series and nb in series:
            step = series.index(nb) - series.index(na)
            if series is ROAD_SERIES:
                # Road grade changes ARE instantaneous (a paving crew does it
                # in a day), so any jump size is the same event type.
                return ("infrastructure",
                        f"road grade {'upgraded' if step > 0 else 'downgraded'} "
                        f"{na}->{nb}")
            # Succession moves ONE stage per epoch at most. A multi-stage jump
            # in a single epoch is not gradual anything: down-series it is the
            # most reportable event on this landscape, up-series it is faster
            # than plants grow.
            if step <= -2:
                return ("real-change",
                        f"{-step}-stage drop along the Mediterranean series "
                        f"({na}->{nb}) in one epoch -- fire, clearance, or "
                        f"label error, not gradual succession")
            if step >= 2:
                return ("real-change",
                        f"{step}-stage regrowth ({na}->{nb}) in one epoch -- "
                        f"succession takes years per stage; planting or a "
                        f"label error, not natural regrowth")
            return ("succession",
                    f"{'regrowth' if step > 0 else 'degradation'} along the "
                    f"Mediterranean series {na}->{nb}")

    if nb in _BUILT and na not in _BUILT:
        return "construction", f"{na} replaced by built surface {nb}"
    if na in _BUILT and nb not in _BUILT:
        return "demolition", f"built surface {na} replaced by {nb}"

    if SUPERCLASS_OF[a] == "artifact" or SUPERCLASS_OF[b] == "artifact":
        return "noise", f"{na}->{nb} involves an artifact class; low confidence"

    # The gate is deliberately STRICT (< 0.4, not <=). Everything the taxonomy
    # itself calls a near-miss lands below it: series neighbours (0.25/0.33)
    # and same-lithology morphology changes (0.35). Exactly 0.4 is the
    # same-superclass floor -- e.g. GreenGrassland->Batha, herbaceous becoming
    # woody -- which is a formation change worth reporting, and gating it as
    # noise would silently delete e.g. shrub encroachment over grassland.
    d = class_distance(a, b)
    if d < 0.4:
        return "noise", f"{na}->{nb} is a near-neighbour confusion, not a change"

    sa, sb = SUPERCLASS_OF[a], SUPERCLASS_OF[b]
    if sa == sb:
        # Same superclass but semantically far apart within it (class_distance
        # == 0.4): "crosses superclasses" would be a lie here.
        return ("real-change",
                f"{na}->{nb} stays within {sa} but is semantically distant "
                f"(distance {d:.2f}) -- a formation change, not a "
                f"near-neighbour wobble")
    return "real-change", f"{na}->{nb} crosses superclasses ({sa} -> {sb})"


@dataclass
class ChangeEvent:
    category: str
    from_class: str
    to_class: str
    area_m2: float
    n_components: int
    why: str


@dataclass
class ChangeReport:
    events: list[ChangeEvent]
    changed_frac: float                  # of pixels valid in BOTH dates
    by_category: dict[str, float]        # category -> area m2
    # Share of the observed footprint (pixels valid in at least one date) that
    # was seen on exactly one date. Coverage moved, not the ground -- these
    # pixels are excluded from every class-change number above.
    coverage_changed_frac: float = 0.0


_EIGHT = np.ones((3, 3), dtype=int)


def _components_per_pair(a: np.ndarray, b: np.ndarray, changed: np.ndarray,
                         pairs: np.ndarray) -> dict[int, int]:
    """8-connected component count per transition, keyed a*N_CLASSES+b.

    Semantics are exactly those of labelling `changed & (a==ca) & (b==cb)` over
    the full raster once per pair (the reference implementation lives in
    tests/test_fixes_s6.py) -- but that costs O(n_pairs x n_pixels) with three
    full-size temporaries per pair, the complexity-class blowup this repo has
    fixed four times already (HANDOFF.md section 8.5). Instead: label `changed`
    ONCE. A per-pair mask is a subset of `changed`, so its components can never
    span two changed-components; a changed-component holding a single pair
    therefore contributes exactly one component to that pair, and only the rare
    MIXED component (different transitions touching) is relabelled -- inside
    its own bounding box, not the full raster.
    """
    out: dict[int, int] = {}
    if not pairs.size:
        return out
    comp, n_comp = ndi.label(changed, structure=_EIGHT)
    key_span = N_CLASSES * N_CLASSES
    combo = np.unique(comp[changed].astype(np.int64) * key_span + pairs)
    combo_comp = combo // key_span
    combo_key = combo % key_span

    pairs_in_comp = np.bincount(combo_comp, minlength=n_comp + 1)
    pure = pairs_in_comp[combo_comp] == 1
    for key, n in zip(*np.unique(combo_key[pure], return_counts=True)):
        out[int(key)] = int(n)

    mixed = np.flatnonzero(pairs_in_comp > 1)
    if mixed.size:
        slices = ndi.find_objects(comp)
        for ci in mixed:
            sl = slices[ci - 1]
            in_comp = comp[sl] == ci
            asl, bsl = a[sl], b[sl]
            for key in combo_key[combo_comp == ci]:
                ca, cb = divmod(int(key), N_CLASSES)
                _, nc = ndi.label(in_comp & (asl == ca) & (bsl == cb),
                                  structure=_EIGHT)
                out[int(key)] = out.get(int(key), 0) + int(nc)
    return out


def compare(t1: LabelRaster, t2: LabelRaster) -> ChangeReport:
    if t1.shape != t2.shape:
        raise ValueError(f"grids differ: {t1.shape} vs {t2.shape}; reproject first")
    a, b = t1.labels, t2.labels
    px = t1.pixel_area_m2

    # Nodata is not a class (see loader.py: on real exports nodata shares the
    # integer 0 with Unclassified, and cropped tiles are mostly nodata). A
    # pixel can only be said to have CHANGED where both dates observed it;
    # a pixel observed on exactly one date is a change in COVERAGE, reported
    # separately below -- letting it into the diff would report whatever
    # garbage sits under the nodata as construction or demolition.
    v1, v2 = t1.valid, t2.valid
    if v1 is None and v2 is None:
        changed = a != b
        n_both = a.size
        coverage_changed_frac = 0.0
    else:
        if v1 is None:
            v1 = np.ones(a.shape, dtype=bool)
        if v2 is None:
            v2 = np.ones(a.shape, dtype=bool)
        both = v1 & v2
        n_both = int(both.sum())
        n_union = int((v1 | v2).sum())
        # Denominator is the union of observed pixels, not the grid: the grid
        # can be padded arbitrarily, the observed footprint cannot.
        coverage_changed_frac = (n_union - n_both) / max(n_union, 1)
        changed = (a != b) & both

    # Key only the changed pixels. Widening the whole raster to int32 first cost
    # 12 bytes/px of peak for a result that is typically well under 1% of it.
    pairs = a[changed].astype(np.int32) * N_CLASSES + b[changed]
    uniq, counts = np.unique(pairs, return_counts=True)
    ncomp_by_pair = _components_per_pair(a, b, changed, pairs)

    events: list[ChangeEvent] = []
    by_cat: dict[str, float] = {}
    for key, n in zip(uniq, counts):
        ca, cb = divmod(int(key), N_CLASSES)
        area = float(n) * px
        if area < MIN_EVENT_AREA_M2:
            continue
        cat, why = classify_transition(ca, cb)
        events.append(ChangeEvent(cat, BY_ID[ca].name, BY_ID[cb].name,
                                  area, ncomp_by_pair[int(key)], why))
        by_cat[cat] = by_cat.get(cat, 0.0) + area

    events.sort(key=lambda e: -e.area_m2)
    # max(n_both, 1): zero overlap means no class-change statement is possible;
    # 0.0 with a 100% coverage-change line is the honest rendering of that.
    return ChangeReport(events, float(changed.sum()) / max(n_both, 1), by_cat,
                        coverage_changed_frac)


def render(rep: ChangeReport, limit: int = 20) -> str:
    total = sum(rep.by_category.values()) or 1.0
    lines = [
        f"# S6 change report -- {rep.changed_frac:.1%} of co-valid pixels differ",
    ]
    if rep.coverage_changed_frac:
        lines.append(
            f"# coverage change: {rep.coverage_changed_frac:.1%} of the observed "
            f"footprint is valid in exactly one date -- coverage moved, not the "
            f"ground; excluded from every class-change number below")
    lines += [
        "",
        "## by category (area-weighted)",
        "category\tarea_m2\tshare",
    ]
    for cat in CATEGORIES:
        if cat in rep.by_category:
            a = rep.by_category[cat]
            lines.append(f"{cat}\t{a:.0f}\t{a / total:.0%}")

    real = sum(v for k, v in rep.by_category.items()
               if k in ("construction", "demolition", "infrastructure", "real-change"))
    lines += [
        "",
        f"# {real / total:.0%} of the differing area is plausibly REAL change; "
        f"the rest is phenology, succession, noise, or label flips.",
        f"# A raw pixel diff would have reported all {rep.changed_frac:.1%} as change.",
        "",
        f"## top {limit} transitions",
        "category\tfrom\tto\tarea_m2\tcomponents\twhy",
    ]
    for e in rep.events[:limit]:
        lines.append(f"{e.category}\t{e.from_class}\t{e.to_class}\t{e.area_m2:.0f}\t"
                     f"{e.n_components}\t{e.why}")
    if len(rep.events) > limit:
        lines.append(f"# NOTE: {len(rep.events) - limit} smaller transitions omitted "
                     f"by limit={limit}")
    return "\n".join(lines)
