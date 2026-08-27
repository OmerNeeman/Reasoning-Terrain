"""Two maps of the same ground, compared. The reference half of the audit.

Everything else in `audit.py` is self-referential: it contradicts a label using
the label's own geometry, its slope, or its neighbours. It cannot tell you that
the thing ST called `Rendzina` is a street, because nothing in the label raster
knows where the streets are.

OSM does. So these checks are structurally different from the priors, and the
difference has to survive into the output:

  **A disagreement is not an error.** Neither map is ground truth. OSM is
    volunteered, incomplete in ways that vary by country and by tag, and stale
    wherever the ground changed recently -- which for these AOIs is exactly
    where the interesting questions are. Every message names both maps and says
    which one asserted what. `tags.RELIABILITY` scales severity accordingly, and
    it is a guess.
  **Absence of an OSM feature is not an assertion.** "ST mapped a road OSM does
    not have" is `osm-road-extra`, and the honest reading of it is *either* ST
    hallucinated a road *or* nobody has mapped that alley yet. Which one it is
    cannot be decided from the data here, and the message says so rather than
    picking.
  **Occlusion is its own finding.** On aza, 28.6% of the ground under an OSM
    road corridor is `Shadow` [measured]. A segmenter that cannot see a street
    in shadow has not made a mistake; it has been prevented from labelling. That
    is `osm-road-occluded`, a different decision for a reviewer than
    `osm-road-missing`, and rolling them together would bury the real thing.

The check families and their causes:

    osm-road-missing     OSM street, ST labels no road class and no shadow
    osm-road-occluded    OSM street, ST labels Shadow (a visibility limit)
    osm-road-grade       both agree it is a road, disagree on paved/dirt
    osm-road-extra       ST road region with no OSM way near it
    osm-building-missing OSM building footprint, ST labels no House
    osm-building-extra   ST House with no OSM footprint under it
    osm-water-extra      ST Water with no OSM water feature
"""

from __future__ import annotations

import numpy as np

from ..audit import Finding
from ..taxonomy import BY_ID, class_distance, cid
from . import tags as T
from . import trust
from .burn import BurnedOsm
from .layer import OsmLayer

# --- thresholds (guesses; see docs/OSM.md) --------------------------------

# A way whose footprint holds fewer valid pixels than this is not measurable:
# a 3 m alley at 0.5 m/px inside a mostly-nodata corner. Skipped, and counted.
MIN_WAY_PX = 60

# ST calls this share of the corridor a road class -> the two maps agree.
# Deliberately low. The corridor is a *buffer around a centreline* whose width
# came from `tags.HALF_WIDTH_M`, and on aza it is 3.4x the area of ST's own road
# pixels [measured], so demanding a majority would flag every correctly-mapped
# street in the AOI.
ROAD_AGREE_SHARE = 0.20

# Below this the disagreement is total: ST sees no road here at all.
ROAD_ABSENT_SHARE = 0.05

# Shadow share above which "ST did not label a road" is read as occlusion
# rather than as a missing label.
OCCLUSION_SHARE = 0.35

# An ST road region with less than this share of its area inside the OSM
# corridor is a road OSM does not have.
EXTRA_ROAD_OVERLAP = 0.15

# ...and it has to be big enough to be worth a row.
EXTRA_MIN_AREA_M2 = 100.0

# Building agreement, same shape. Footprints are small, so the share is over
# the polygon rather than a buffer and the bar is higher.
BUILDING_AGREE_SHARE = 0.25
BUILDING_MIN_PX = 30
EXTRA_BUILDING_MIN_AREA_M2 = 40.0

WATER_MIN_AREA_M2 = 200.0

# Severity ceiling for every reference finding. Nothing sourced from a second
# map of unknown completeness gets to outrank a physical contradiction (a
# badlands on flat ground, which `audit.py` grades up to 1.0).
MAX_SEVERITY = 0.85


def _sev(disagreement: float, reliability: float, kind: str = "") -> float:
    """Severity = disagreement x layer reliability x the axis's scale.

    Multiplicative on purpose: a total disagreement about a barrier (reliability
    0.30) must not outrank a partial one about the road network (0.75). The
    third factor is the owner's trust policy (`trust.py`): a disagreement about
    the current STATE of the ground is a change candidate rather than a label
    defect, and is ranked a little below one about what is there at all."""
    scale = trust.SEVERITY_SCALE.get(trust.resolve(kind).axis, 1.0) if kind else 1.0
    return round(min(disagreement * reliability * scale, MAX_SEVERITY), 3)


def _dominant_region(ridx, mask_local: np.ndarray, bbox) -> tuple[int, float]:
    """The ST region holding most of a footprint, and its share.

    A way-based finding still has to hang off a region id, because that is the
    handle S1's roll-up, `detail()` and every downstream tool address things by.
    """
    r0, c0, r1, c1 = bbox
    sub = np.asarray(ridx.label_array[r0:r1, c0:c1])[mask_local]
    sub = sub[sub > 0]
    if sub.size == 0:
        return 0, 0.0
    ids, counts = np.unique(sub, return_counts=True)
    i = int(np.argmax(counts))
    return int(ids[i]), float(counts[i] / sub.size)


def road_findings(ridx, raster, layer: OsmLayer,
                  min_area_m2: float = 25.0) -> list[Finding]:
    """Per-OSM-way: what does ST say is under this street?"""
    out: list[Finding] = []
    rel = T.RELIABILITY.get("road", T.RELIABILITY_DEFAULT)
    road_ids = [cid(n) for n in T.ROAD_CLASSES]
    shadow = cid("Shadow")
    unlab = cid("Unclassified")

    for fp in layer.burned.by_kind("road"):
        if fp.n_px < MIN_WAY_PX:
            continue
        st_road = float(fp.hist[road_ids].sum()) / fp.n_px
        sh = float(fp.hist[shadow] + fp.hist[unlab]) / fp.n_px
        local = fp.local_mask(raster.shape)
        if local is None:
            continue
        rid, _share = _dominant_region(ridx, local, fp.bbox)
        area = fp.n_px * raster.gsd ** 2
        dom = fp.dominant(3)
        dom_txt = ", ".join(f"{BY_ID[c].name} {f:.0%}" for c, f in dom)
        where = (f"{fp.label}"
                 + (f" (way {fp.way_id}, {fp.grade})" if fp.grade else
                    f" (way {fp.way_id})"))

        if st_road < ROAD_AGREE_SHARE:
            if sh >= OCCLUSION_SHARE:
                out.append(Finding(
                    region_id=rid, class_name=BY_ID[dom[0][0]].name if dom else "?",
                    kind="osm-road-occluded",
                    severity=_sev(min((1 - st_road) * sh, 1.0), rel,
                                  "osm-road-occluded"),
                    area_m2=area,
                    message=(f"[{trust.resolve('osm-road-occluded').reading}] "
                             f"OSM maps {where} here; ST labels {sh:.0%} of the "
                             f"corridor Shadow/Unclassified and only "
                             f"{st_road:.0%} a road class. The street is "
                             f"probably there and unseeable, not absent"),
                    cause=f"osm-road-occluded/{fp.grade or 'road'}",
                    cause_text=(f"OSM {fp.grade or 'road'}s whose corridor ST "
                                f"labels mostly Shadow or Unclassified. "
                                + trust.resolve("osm-road-occluded").text),
                ))
            else:
                total = st_road < ROAD_ABSENT_SHARE
                out.append(Finding(
                    region_id=rid, class_name=BY_ID[dom[0][0]].name if dom else "?",
                    kind="osm-road-missing",
                    severity=_sev(1.0 - st_road, rel, "osm-road-missing"),
                    area_m2=area,
                    message=(f"[{trust.resolve('osm-road-missing').reading}] "
                             f"OSM maps {where} here; ST labels "
                             f"{'none' if total else f'{st_road:.0%}'} of the "
                             f"corridor as a road. Under it: {dom_txt}"),
                    cause=f"osm-road-missing/{fp.grade or 'road'}",
                    cause_text=(f"OSM {fp.grade or 'road'}s with no ST road class "
                                f"under them. "
                                + trust.resolve("osm-road-missing").text),
                ))
            continue

        # Both maps say road. Do they agree on which kind?
        expected, why = T.expected_road_class(
            next((w.tags for w in layer.vectors.ways if w.id == fp.way_id), {}))
        if not expected:
            continue
        want = cid(expected)
        actual_id = max(road_ids, key=lambda c: fp.hist[c])
        if actual_id == want:
            continue
        actual = BY_ID[actual_id].name
        share = float(fp.hist[actual_id]) / fp.n_px
        d = class_distance(want, actual_id)
        weak = why.endswith("(weak)")
        rely = T.SURFACE_RELIABILITY if not weak else rel * 0.5
        if weak:
            continue          # a grade prior is not enough to raise a finding
        out.append(Finding(
            region_id=rid, class_name=actual, kind="osm-road-grade",
            severity=_sev(d * share, rely, "osm-road-grade"), area_m2=area,
            message=(f"[{trust.resolve('osm-road-grade').reading}] "
                     f"OSM says {where} is {expected} ({why}); ST labels "
                     f"{share:.0%} of the corridor {actual} "
                     f"(class distance {d:.2f})"),
            cause=f"osm-road-grade/{expected}-vs-{actual}",
            cause_text=(f"OSM's `surface` tag says {expected}, ST says {actual}. "
                        + trust.resolve("osm-road-grade").text),
        ))
    return out


def extra_road_findings(ridx, raster, layer: OsmLayer,
                        min_area_m2: float = EXTRA_MIN_AREA_M2) -> list[Finding]:
    """Per-ST-region: a road OSM has never heard of."""
    corridor = layer.burned.mask("road")
    rel = T.RELIABILITY.get("road", T.RELIABILITY_DEFAULT)
    out: list[Finding] = []
    for name in T.ROAD_CLASSES:
        c = cid(name)
        for r in ridx.by_class(c):
            if r.area_m2 < max(min_area_m2, EXTRA_MIN_AREA_M2):
                continue
            r0, c0, r1, c1 = r.bbox
            sub = np.asarray(ridx.label_array[r0:r1, c0:c1]) == r.id
            ov = float((corridor[r0:r1, c0:c1] & sub).sum()) / max(sub.sum(), 1)
            if ov >= EXTRA_ROAD_OVERLAP:
                continue
            out.append(Finding(
                region_id=r.id, class_name=name, kind="osm-road-extra",
                severity=_sev(1.0 - ov, rel * 0.8, "osm-road-extra"),
                area_m2=r.area_m2,
                message=(f"[{trust.resolve('osm-road-extra').reading}] "
                         f"ST maps {r.area_m2:.0f} m2 of {name} here; "
                         f"{ov:.0%} of it falls inside any OSM road corridor. "
                         f"Either an unmapped track -- OSM's coverage of minor "
                         f"tracks is the weakest thing in this comparison -- or "
                         f"an ST false positive"),
                cause=f"osm-road-extra/{name}",
                cause_text=(f"ST {name} regions with no OSM way under them. "
                            + trust.resolve("osm-road-extra").text),
            ))
    return out


def building_findings(ridx, raster, layer: OsmLayer) -> list[Finding]:
    """House vs building footprint, both directions."""
    if not layer.burned.has("building"):
        return []
    out: list[Finding] = []
    rel = T.RELIABILITY.get("building", T.RELIABILITY_DEFAULT)
    house = cid("House")
    built = [cid(n) for n in ("House", "BrickWall", "Pavement", "Clutter")]

    for fp in layer.burned.by_kind("building"):
        if fp.n_px < BUILDING_MIN_PX:
            continue
        st_house = float(fp.hist[house]) / fp.n_px
        if st_house >= BUILDING_AGREE_SHARE:
            continue
        local = fp.local_mask(raster.shape)
        if local is None:
            continue
        rid, _ = _dominant_region(ridx, local, fp.bbox)
        st_built = float(fp.hist[built].sum()) / fp.n_px
        dom = fp.dominant(2)
        out.append(Finding(
            region_id=rid, class_name=BY_ID[dom[0][0]].name if dom else "?",
            kind="osm-building-missing",
            severity=_sev(1.0 - st_house, rel, "osm-building-missing"),
            area_m2=fp.n_px * raster.gsd ** 2,
            message=(f"[{trust.resolve('osm-building-missing').reading}] "
                     f"OSM has a building footprint here (way {fp.way_id}); ST "
                     f"labels {st_house:.0%} House, {st_built:.0%} anything "
                     f"built. Under it: "
                     + ", ".join(f"{BY_ID[c].name} {f:.0%}" for c, f in dom)),
            # Keyed on what ST says instead, not just on the kind: "OSM
            # building, ST says DryGrassland" and "OSM building, ST says Shadow"
            # are two different decisions -- the first is a missed footprint or a
            # demolished one, the second is a visibility limit. One cause holding
            # both is the 25-identical-rows worklist this module exists to avoid.
            cause=f"osm-building-missing/{BY_ID[dom[0][0]].name if dom else '?'}",
            cause_text=(f"OSM building footprints where ST says "
                        f"{BY_ID[dom[0][0]].name if dom else '?'} instead of "
                        f"House. "
                        + trust.resolve("osm-building-missing").text
                        + " A third reading, on this AOI: a class-id mapping "
                          "that does not mean what it says (see docs/OSM.md)."),
        ))

    bmask = layer.burned.mask("building")
    for r in ridx.by_class(house):
        if r.area_m2 < EXTRA_BUILDING_MIN_AREA_M2:
            continue
        r0, c0, r1, c1 = r.bbox
        sub = np.asarray(ridx.label_array[r0:r1, c0:c1]) == r.id
        ov = float((bmask[r0:r1, c0:c1] & sub).sum()) / max(sub.sum(), 1)
        if ov >= BUILDING_AGREE_SHARE:
            continue
        out.append(Finding(
            region_id=r.id, class_name="House", kind="osm-building-extra",
            severity=_sev(1.0 - ov, rel * 0.7, "osm-building-extra"),
            area_m2=r.area_m2,
            message=(f"[{trust.resolve('osm-building-extra').reading}] "
                     f"ST maps a {r.area_m2:.0f} m2 House here; {ov:.0%} of it "
                     f"is inside an OSM footprint. Unmapped building, or a "
                     f"structure OSM never had"),
            cause="osm-building-extra",
            cause_text=("ST House regions with no OSM footprint. "
                        + trust.resolve("osm-building-extra").text
                        + " OSM building coverage is the least uniform layer in "
                          "this comparison."),
        ))
    return out


def water_findings(ridx, raster, layer: OsmLayer) -> list[Finding]:
    if not (layer.burned.has("water") or layer.burned.has("flow")):
        return []
    mask = None
    for name in ("water", "flow"):
        if layer.burned.has(name):
            m = layer.burned.mask(name)
            mask = m if mask is None else (mask | m)
    rel = T.RELIABILITY.get("water", T.RELIABILITY_DEFAULT)
    out: list[Finding] = []
    for r in ridx.by_class(cid("Water")):
        if r.area_m2 < WATER_MIN_AREA_M2:
            continue
        r0, c0, r1, c1 = r.bbox
        sub = np.asarray(ridx.label_array[r0:r1, c0:c1]) == r.id
        ov = float((mask[r0:r1, c0:c1] & sub).sum()) / max(sub.sum(), 1)
        if ov >= 0.2:
            continue
        out.append(Finding(
            region_id=r.id, class_name="Water", kind="osm-water-extra",
            severity=_sev(1.0 - ov, rel, "osm-water-extra"), area_m2=r.area_m2,
            message=(f"[{trust.resolve('osm-water-extra').reading}] "
                     f"ST maps {r.area_m2:.0f} m2 of Water here and OSM maps no "
                     f"water feature within it ({ov:.0%} overlap). A seasonal "
                     f"pool, a new impoundment, or a wet-looking surface"),
            cause="osm-water-extra",
            cause_text=("ST Water with no OSM water feature. "
                        + trust.resolve("osm-water-extra").text),
        ))
    return out


# Which families this module can contribute, for S1's coverage block. Every one
# of them is substantive -- they test the LABEL against an outside statement,
# which is exactly what the self-referential checks cannot do.
CHECK_FAMILIES = (
    "osm-road-missing", "osm-road-occluded", "osm-road-grade", "osm-road-extra",
    "osm-building-missing", "osm-building-extra", "osm-water-extra",
)


def classes_covered(layer: OsmLayer) -> dict[str, list[str]]:
    """class name -> the OSM checks that actually ran on it, given what was burned."""
    out: dict[str, list[str]] = {}
    road = ["osm-road-missing", "osm-road-occluded", "osm-road-grade",
            "osm-road-extra"]
    for n in T.ROAD_CLASSES:
        out[n] = list(road)
    if layer.burned.has("building"):
        out["House"] = ["osm-building-missing", "osm-building-extra"]
    if layer.burned.has("water") or layer.burned.has("flow"):
        out["Water"] = ["osm-water-extra"]
    return out


def findings(ridx, raster, layer: OsmLayer, min_area_m2: float = 25.0) -> list[Finding]:
    """Every reference check, or nothing at all if the maps are not registered.

    The refusal is the point. A 30 m offset would make every narrow feature
    disagree, and the audit would report thousands of confident findings about
    a georeference bug.
    """
    if layer.align is not None and layer.align.verdict == "MISREGISTERED":
        return []
    out = road_findings(ridx, raster, layer, min_area_m2=min_area_m2)
    out += extra_road_findings(ridx, raster, layer, min_area_m2=min_area_m2)
    out += building_findings(ridx, raster, layer)
    out += water_findings(ridx, raster, layer)
    return out
