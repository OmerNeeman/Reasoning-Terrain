"""Deterministic consistency audit over the region index.

This is the cheap half of the reasoning layer. It applies the co-occurrence and
morphology priors in `taxonomy.PRIORS` -- world knowledge the segmenter never
had, because it learned "LimestoneRockDipSlope" as an integer and has no way to
know that a dip slope is one coherent facet or it isn't.

The output is a *candidate* list. Deciding which candidates are real errors is
the LLM's job; enumerating them is not, and paying an LLM to enumerate them
would defeat the point.

Two things this module is deliberately careful about:

  Root cause, not row count. Every finding carries a `cause` key naming the
    class-level fact it is an instance of. "Rendzina rarely borders chalk here"
    is one thing to decide about, whether it fires on 3 regions or 9,833; S1
    rolls up on this key so a reviewer sees N distinct problems, not N rows.
  Severity is graded, not stamped. A per-check constant makes every finding of
    a kind indistinguishable, and then the ranking inside that kind is
    arbitrary -- which is how a single prior with a fixed 0.45 swept an entire
    review budget. Each check maps its own evidence onto a band instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import Region, RegionIndex
from .taxonomy import (
    BY_ID,
    PRIORS_BY_SUBJECT,
    SUPERCLASS_OF,
    cid,
    class_distance,
)

# --- thresholds ------------------------------------------------------------
# Every number here is a guess unless its comment says otherwise. Grouped so
# they can be swept as a set; see docs/solutions/S1-audit.md for the "where
# this should actually come from" column.

# Boundary-share at which a "contradicts" / "expects" prior fires.
# WHERE THIS SHOULD COME FROM: a sweep against reviewed regions.
CONTRADICTS_SHARE = 0.40
EXPECTS_SHARE = 0.05

# A rock unit whose boundary is this fraction other-lithology is an island.
# WHERE THIS SHOULD COME FROM: the geological map, as ground truth.
LITHOLOGY_ISOLATION_SHARE = 0.85

# Nari caps units: large AND blocky contradicts that.
# WHERE THIS SHOULD COME FROM: a geologist, or the measured size/shape
# distribution of mapped nari outcrops in this region.
NARI_BLOB_AREA_M2 = 20_000.0
NARI_BLOB_COMPACTNESS = 0.35

# --- enclosed components: two windows, because these are two questions ------
# One check used to answer both, through a single 25-400 m2 window set against
# a 0.3 m/px synthetic fixture. That window is the wrong shape for at least one
# of them, and on real data it was the wrong shape for both:
#
#   measured, aza  (0.122 m/px):  25-400 m2 = 1,667-26,675 resolution cells
#   measured, sinai (0.496 m/px): 25-400 m2 =   102- 1,625 resolution cells
#
# a 16x difference in what "small" means. That is why the aza worklist ranked
# 300 m2 BUILDINGS as isolated specks.
#
# isolated-speck asks "is this below the scale at which the segmenter can
# assert anything?" -- a question about resolution cells. It is deliberately
# NOT subject to the audit's min_area_m2 gate: it is the check about components
# beneath that gate, and forcing it above the gate is what produced the
# buildings-as-specks result.
#
#   SPECK_MIN_PX = 32, SPECK_MAX_PX = 128: the modal component size measured on
#   the sinai tile is the 32-128 cell bin (28.5% of its 14,257 components, the
#   peak of the distribution). Below 32 cells is the sub-resolution tail --
#   38.9% of sinai-tile and 70.5% of aza components, carrying 0.09% and 1.62%
#   of classified area; enumerating it is measuring salt-and-pepper, so
#   s1_audit reports it once as a fragmentation rate instead. Above 128 cells
#   the segmenter is drawing objects. In ground units the window is 7.9-31.5 m2
#   on sinai and 0.48-1.9 m2 on aza -- small in both, which is the point.
#   The aza distribution has NO mode above 1 cell: it decays monotonically from
#   the floor, which is the signature of a map with no object scale at all.
#
# oov-candidate asks "is there an unnamed OBJECT here worth a detector pass?"
# -- a question about ground size, not cells. A car is a car at any GSD, so it
# keeps a ground-area window and stays behind min_area_m2.
#
# WHERE THESE SHOULD COME FROM: SPECK_* from the segmenter's receptive field
# and its post-processing minimum mapping unit; OOV_MAX_M2 from the size of the
# objects the downstream detector is being paid to find.
SPECK_MIN_PX = 32
SPECK_MAX_PX = 128
OOV_MAX_M2 = 400.0
ENCLOSURE_SHARE = 0.90
SPECK_MIN_CLASS_DISTANCE = 0.70

# --- severity --------------------------------------------------------------
# SEVERITY IS AN ORDINAL RANKING SIGNAL WITHIN A KIND. It is not a calibrated
# probability that the label is wrong, and nothing has measured whether the
# bands are comparable ACROSS kinds -- that needs the reviewed sample S1's
# blocking open question asks for.
#
# Each check computes a violation fraction in 0..1 from its own evidence and
# maps it onto (floor, span) below. The floors are ordered by how hard the
# constraint is: aspect and slope test physics, context tests a tendency.
SEVERITY_BAND: dict[str, tuple[float, float]] = {
    "aspect-incoherent":        (0.60, 0.35),
    "slope-violation":          (0.55, 0.35),
    "contradicted-context":     (0.50, 0.40),
    "isolated-lithology":       (0.40, 0.40),
    "oov-candidate":            (0.35, 0.40),
    "nari-morphology":          (0.35, 0.35),
    "isolated-speck":           (0.25, 0.45),
    "missing-expected-context": (0.20, 0.40),
}

# A context claim made from 5 m of shared boundary is a weaker observation than
# the same claim made from 5 km of it. Below this the violation fraction is
# discounted; above it, taken at face value.
# WHERE THIS SHOULD COME FROM: the boundary length at which neighbour-class
# share stops being noisy -- measurable today from the region index.
EVIDENCE_FULL_PERIMETER_M = 200.0

# A slope outside its required band by this many standard deviations of the
# region's own slope is a full violation. Uses std_slope so a mean computed
# over wildly varying ground is not treated as a hard measurement.
# UNEXERCISED: neither sinai nor aza ships a DEM, so no slope or aspect check
# has ever fired on real data.
SLOPE_VIOLATION_SIGMA = 2.0


def _saturate(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _grade(kind: str, violation: float, confidence: float = 1.0) -> float:
    """Map an evidence violation fraction onto this kind's severity band."""
    floor, span = SEVERITY_BAND[kind]
    x = _saturate(violation) * (0.5 + 0.5 * _saturate(confidence))
    return round(floor + span * x, 3)


def _perimeter_confidence(r: Region) -> float:
    return _saturate(r.perimeter_m / EVIDENCE_FULL_PERIMETER_M)


@dataclass
class Finding:
    region_id: int
    class_name: str
    kind: str
    severity: float            # 0..1, ordinal within `kind` -- see SEVERITY_BAND
    area_m2: float
    message: str
    # The class-level fact this finding is one instance of. Findings sharing a
    # cause are ONE thing for a reviewer to decide about; S1 rolls up on it.
    cause: str = ""
    # That fact stated once, without per-region parameters.
    cause_text: str = ""

    def __post_init__(self) -> None:
        if not self.cause:
            self.cause = f"{self.kind}/{self.class_name}"
        if not self.cause_text:
            self.cause_text = self.message

    def as_row(self) -> str:
        return (f"{self.region_id}\t{self.class_name}\t{self.kind}\t"
                f"{self.severity:.2f}\t{self.area_m2:.0f}\t{self.message}")


def audit(ridx: RegionIndex, min_area_m2: float = 25.0) -> list[Finding]:
    findings: list[Finding] = []
    for r in ridx.regions:
        big_enough = r.area_m2 >= min_area_m2
        # The speck check is the one check ABOUT small components; gating it on
        # min_area_m2 inverts it. See the window comment above.
        specky = SPECK_MIN_PX <= r.area_px <= SPECK_MAX_PX
        if not (big_enough or specky):
            continue
        nh = ridx.neighbor_class_hist(r)
        if big_enough:
            findings += _check_priors(r, nh, ridx.has_terrain)
            findings += _check_lithology_context(r, nh)
            findings += _check_nari_geometry(r)
            findings += _check_oov(r, nh, min_area_m2)
        if specky:
            findings += _check_speck(r, nh)
    findings.sort(key=lambda f: (-f.severity, -f.area_m2))
    return findings


def _check_priors(r: Region, nh: dict[int, float], has_terrain: bool) -> list[Finding]:
    out: list[Finding] = []
    total = sum(nh.values()) or 1.0
    conf = _perimeter_confidence(r)

    for p in PRIORS_BY_SUBJECT.get(r.class_name, ()):
        if p.kind == "contradicts" and p.others:
            share = sum(nh.get(cid(n), 0.0) for n in p.others) / total
            if share > CONTRADICTS_SHARE:
                out.append(Finding(
                    r.id, r.class_name, "contradicted-context",
                    _grade("contradicted-context",
                           (share - CONTRADICTS_SHARE) / (1.0 - CONTRADICTS_SHARE), conf),
                    r.area_m2,
                    f"{share:.0%} of {r.perimeter_m:.0f} m boundary is with "
                    f"{{{', '.join(p.others)}}} -- {p.why}",
                    cause=f"contradicted-context/{r.class_name}/{'+'.join(p.others)}",
                    cause_text=f"{r.class_name} bordering {{{', '.join(p.others)}}} -- {p.why}",
                ))

        if p.kind == "expects" and p.others:
            share = sum(nh.get(cid(n), 0.0) for n in p.others) / total
            if share < EXPECTS_SHARE:
                out.append(Finding(
                    r.id, r.class_name, "missing-expected-context",
                    _grade("missing-expected-context",
                           1.0 - share / EXPECTS_SHARE, conf),
                    r.area_m2,
                    f"{share:.1%} of {r.perimeter_m:.0f} m boundary is with any of "
                    f"{{{', '.join(p.others)}}} -- {p.why}",
                    cause=f"missing-expected-context/{r.class_name}/{'+'.join(p.others)}",
                    cause_text=f"{r.class_name} not bordering any of "
                               f"{{{', '.join(p.others)}}} -- {p.why}",
                ))

        # No DEM means mean_slope is 0.0 and aspect_circvar is 0.0 for *every*
        # region. Running these checks anyway converts "we never measured slope"
        # into a high-severity finding against every badlands polygon on the
        # map, which then dominates the worklist. Abstaining is the honest move;
        # S1's coverage block reports how many checks abstained and why.
        if not has_terrain:
            continue

        if p.slope_deg is not None:
            lo, hi = p.slope_deg
            if not (lo <= r.mean_slope <= hi):
                excess = max(lo - r.mean_slope, r.mean_slope - hi)
                sigma = max(r.std_slope, 0.5)
                out.append(Finding(
                    r.id, r.class_name, "slope-violation",
                    _grade("slope-violation", excess / (SLOPE_VIOLATION_SIGMA * sigma)),
                    r.area_m2,
                    f"mean slope {r.mean_slope:.1f}+-{r.std_slope:.1f} deg is "
                    f"{excess:.1f} deg outside the required {lo}-{hi} deg -- {p.why}",
                    cause=f"slope-violation/{r.class_name}",
                    cause_text=f"{r.class_name} on slopes outside {lo}-{hi} deg -- {p.why}",
                ))

        if p.aspect_variance_max is not None and r.aspect_circvar > p.aspect_variance_max:
            cap = p.aspect_variance_max
            out.append(Finding(
                r.id, r.class_name, "aspect-incoherent",
                _grade("aspect-incoherent", (r.aspect_circvar - cap) / (1.0 - cap)),
                r.area_m2,
                f"aspect circular variance {r.aspect_circvar:.2f} exceeds "
                f"{cap} -- {p.why}",
                cause=f"aspect-incoherent/{r.class_name}",
                cause_text=f"{r.class_name} without a coherent aspect -- {p.why}",
            ))
    return out


def _check_lithology_context(r: Region, nh: dict[int, float]) -> list[Finding]:
    """A rock unit whose neighbours are all a *different* lithology is the
    classic RGB-blind confusion: limestone/dolomite/nari are near-identical in
    colour, and only the geological map separates them.

    This is a claim about two ROCK units being confusable with each other, not
    about what a soil formed on -- see taxonomy.RETIRED_PRIORS for that
    distinction and why it matters."""
    d = BY_ID[r.class_id]
    if not d.lithology:
        return []
    total = sum(nh.values()) or 1.0
    other = 0.0
    litho_share: dict[str, float] = {}
    for k, v in nh.items():
        kl = BY_ID[k].lithology
        if kl is None:
            continue
        litho_share[kl] = litho_share.get(kl, 0.0) + v
        if kl != d.lithology:
            other += v
    share = other / total
    if share > LITHOLOGY_ISOLATION_SHARE and litho_share:
        dom = max(litho_share, key=litho_share.get)
        if dom != d.lithology:
            dom_share = litho_share[dom] / total
            return [Finding(
                r.id, r.class_name, "isolated-lithology",
                _grade("isolated-lithology",
                       (share - LITHOLOGY_ISOLATION_SHARE)
                       / (1.0 - LITHOLOGY_ISOLATION_SHARE),
                       dom_share),
                r.area_m2,
                f"{share:.0%} of boundary is with other-lithology units, "
                f"{dom_share:.0%} of it {dom}; a {d.lithology} island inside "
                f"{dom} is usually a lithology confusion, not a real contact "
                f"(unresolvable in RGB -- needs the geological map)",
                cause=f"isolated-lithology/{r.class_name}/in-{dom}",
                cause_text=f"{d.lithology} unit {r.class_name} isolated inside {dom} -- "
                           f"the limestone/dolomite/nari confusion, needs the "
                           f"geological map",
            )]
    return []


def _check_nari_geometry(r: Region) -> list[Finding]:
    """Nari is a calcrete crust: it caps units. Expect thin high bands, not
    large low-lying blobs. Geometry, not genesis."""
    if BY_ID[r.class_id].lithology != "Nari":
        return []
    if r.area_m2 > NARI_BLOB_AREA_M2 and r.compactness > NARI_BLOB_COMPACTNESS:
        big = _saturate((r.area_m2 / NARI_BLOB_AREA_M2 - 1.0) / 4.0)
        blocky = _saturate((r.compactness - NARI_BLOB_COMPACTNESS)
                           / (1.0 - NARI_BLOB_COMPACTNESS))
        return [Finding(
            r.id, r.class_name, "nari-morphology",
            _grade("nari-morphology", float(np.sqrt(max(big, 1e-6) * max(blocky, 1e-6)))),
            r.area_m2,
            f"large ({r.area_m2:.0f} m2, {r.area_m2 / NARI_BLOB_AREA_M2:.1f}x the "
            f"blob threshold) and blocky (compactness {r.compactness:.2f}); nari "
            f"caps units and should appear as thin plateau-edge bands",
            cause=f"nari-morphology/{r.class_name}",
            cause_text=f"{r.class_name} mapped as large blocky bodies rather than "
                       f"thin capping bands",
        )]
    return []


def _host(r: Region, nh: dict[int, float]) -> tuple[int, float] | None:
    """The class this region is enclosed by, if one class dominates its whole
    boundary. None if it is not an island."""
    if not r.neighbors or not nh:
        return None
    total = sum(nh.values()) or 1.0
    host, shared = max(nh.items(), key=lambda kv: kv[1])
    share = shared / total
    return (host, share) if share >= ENCLOSURE_SHARE else None


def _check_speck(r: Region, nh: dict[int, float]) -> list[Finding]:
    """A component too small for the segmenter to have textured anything over,
    fully enclosed by a semantically distant class.

    The window is in resolution cells, not m2; see SPECK_MIN_PX above for the
    measurement that decided it. This is the noise question."""
    hs = _host(r, nh)
    if hs is None:
        return []
    host, share = hs
    if SUPERCLASS_OF[r.class_id] == "artifact":
        return []                     # handled by _check_oov, at object scale
    dist = class_distance(r.class_id, host)
    if dist < SPECK_MIN_CLASS_DISTANCE:
        return []

    enclosure = (share - ENCLOSURE_SHARE) / (1.0 - ENCLOSURE_SHARE)
    smallness = (SPECK_MAX_PX - r.area_px) / (SPECK_MAX_PX - SPECK_MIN_PX)
    strangeness = (dist - SPECK_MIN_CLASS_DISTANCE) / (1.0 - SPECK_MIN_CLASS_DISTANCE)
    host_name = BY_ID[host].name
    return [Finding(
        r.id, r.class_name, "isolated-speck",
        _grade("isolated-speck", (strangeness + enclosure + smallness) / 3.0, share),
        r.area_m2,
        f"{r.area_px} cell ({r.area_m2:.1f} m2) island {share:.0%} enclosed by "
        f"{host_name} (class distance {dist:.2f})",
        cause=f"isolated-speck/{r.class_name}/in-{host_name}",
        cause_text=f"{r.class_name} islands inside {host_name} (class distance "
                   f"{dist:.2f}) -- salt-and-pepper, or a real inclusion",
    )]


def _check_oov(r: Region, nh: dict[int, float], min_area_m2: float) -> list[Finding]:
    """An artifact-superclass object, enclosed, at OBJECT scale: the segmenter
    saying "something is here and I lack a class for it". Ground area, not
    cells -- a car is a car at any GSD."""
    if r.area_m2 > OOV_MAX_M2 or SUPERCLASS_OF[r.class_id] != "artifact":
        return []
    hs = _host(r, nh)
    if hs is None:
        return []
    host, share = hs
    enclosure = (share - ENCLOSURE_SHARE) / (1.0 - ENCLOSURE_SHARE)
    span = max(OOV_MAX_M2 - min_area_m2, 1.0)
    smallness = _saturate((OOV_MAX_M2 - r.area_m2) / span)
    host_name = BY_ID[host].name
    return [Finding(
        r.id, r.class_name, "oov-candidate",
        _grade("oov-candidate", 0.5 * enclosure + 0.5 * smallness, share),
        r.area_m2,
        f"{r.area_m2:.0f} m2 {r.class_name} object {share:.0%} enclosed by "
        f"{host_name} -- by definition outside the 47-class vocabulary; "
        f"prime target for a detector pass",
        cause=f"oov-candidate/{r.class_name}/in-{host_name}",
        cause_text=f"{r.class_name} objects enclosed by {host_name} -- outside the "
                   f"47-class vocabulary; detector targets",
    )]


def to_tsv(findings: list[Finding], limit: int | None = None) -> str:
    rows = findings[:limit] if limit else findings
    lines = [
        "# consistency audit -- CANDIDATES, not confirmed errors",
        "rid\tclass\tkind\tseverity\tarea_m2\tmessage",
    ]
    lines += [f.as_row() for f in rows]
    if limit and len(findings) > limit:
        lines.append(f"# NOTE: {len(findings) - limit} lower-severity findings omitted")
    return "\n".join(lines)


def summary(findings: list[Finding]) -> str:
    if not findings:
        return "no findings"
    kinds: dict[str, int] = {}
    for f in findings:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    n_causes = len({f.cause for f in findings})
    return (f"{len(findings)} findings from {n_causes} distinct root causes "
            f"({parts})")
