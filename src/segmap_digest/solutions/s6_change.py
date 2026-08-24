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
                return ("infrastructure",
                        f"road grade {'upgraded' if step > 0 else 'downgraded'} "
                        f"{na}->{nb}")
            return ("succession",
                    f"{'regrowth' if step > 0 else 'degradation'} along the "
                    f"Mediterranean series {na}->{nb}")

    if nb in _BUILT and na not in _BUILT:
        return "construction", f"{na} replaced by built surface {nb}"
    if na in _BUILT and nb not in _BUILT:
        return "demolition", f"built surface {na} replaced by {nb}"

    if SUPERCLASS_OF[a] == "artifact" or SUPERCLASS_OF[b] == "artifact":
        return "noise", f"{na}->{nb} involves an artifact class; low confidence"

    if class_distance(a, b) < 0.4:
        return "noise", f"{na}->{nb} is a near-neighbour confusion, not a change"

    return "real-change", f"{na}->{nb} crosses superclasses"


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
    changed_frac: float
    by_category: dict[str, float]        # category -> area m2


def compare(t1: LabelRaster, t2: LabelRaster) -> ChangeReport:
    if t1.shape != t2.shape:
        raise ValueError(f"grids differ: {t1.shape} vs {t2.shape}; reproject first")
    a, b = t1.labels, t2.labels
    px = t1.pixel_area_m2

    changed = a != b
    # Key only the changed pixels. Widening the whole raster to int32 first cost
    # 12 bytes/px of peak for a result that is typically well under 1% of it.
    pairs = a[changed].astype(np.int32) * N_CLASSES + b[changed]
    uniq, counts = np.unique(pairs, return_counts=True)

    events: list[ChangeEvent] = []
    by_cat: dict[str, float] = {}
    for key, n in zip(uniq, counts):
        ca, cb = divmod(int(key), N_CLASSES)
        area = float(n) * px
        if area < MIN_EVENT_AREA_M2:
            continue
        cat, why = classify_transition(ca, cb)
        mask = changed & (a == ca) & (b == cb)
        _, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
        events.append(ChangeEvent(cat, BY_ID[ca].name, BY_ID[cb].name,
                                  area, int(ncomp), why))
        by_cat[cat] = by_cat.get(cat, 0.0) + area

    events.sort(key=lambda e: -e.area_m2)
    return ChangeReport(events, float(changed.mean()), by_cat)


def render(rep: ChangeReport, limit: int = 20) -> str:
    total = sum(rep.by_category.values()) or 1.0
    lines = [
        f"# S6 change report -- {rep.changed_frac:.1%} of pixels differ",
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
