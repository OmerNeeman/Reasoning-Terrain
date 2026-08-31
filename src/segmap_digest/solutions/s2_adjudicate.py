"""S2 -- confusion adjudication.

S1 says "region 245 looks wrong". S2 answers "then what is it?" -- by scoring a
short candidate list on evidence the segmenter never used, and by refusing to
answer when the evidence cannot separate the candidates.

That refusal is the important part. Limestone / Dolomite / Nari rocky terrain
are not separable in RGB even for a human expert. A model that picks one
confidently is producing a corruption, not a correction. The verdict
UNDECIDABLE-NEEDS-<data> is a first-class output here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..index import Region, RegionIndex
from ..taxonomy import (
    BY_ID,
    PRIORS_BY_SUBJECT,
    N_CLASSES,
    cid,
    class_distance,
)

# --- tunable heuristics (all guesses -- see docs/solutions/S2.md) ----------

# Evidence term weights. Must sum to 1.0.
W_CONTEXT = 0.45      # do the neighbours fit this class's co-occurrence priors
W_MORPHOLOGY = 0.35   # does slope / aspect coherence fit
W_GEOMETRY = 0.20     # does size / shape fit

# Weight of the reference term (OSM), when there is one. The three weights above
# are renormalised to 1 - this, rather than being changed, so that the no-OSM
# path stays byte-for-byte what it was.
#
# How much of a say that buys depends on the AXIS of the evidence, not on this
# weight -- see `osm/trust.py`, which holds the owner's policy: OSM is the
# reference for what is there, ST is the latest word on what it looks like now.
# Read on aza after that policy was set:
#
#   IDENTITY/EXISTENCE, where OSM is the reference. A polygon 70% inside a
#     mapped corridor whose way is tagged `surface=unpaved`: `DirtRoad` scores
#     0.74 on the reference term and `PavedRoad` 0.42 -- and the incumbent
#     `DirtRoad` survives a challenge from `MaralSmoothRockSlopes` that beats it
#     on the label raster alone. This is the case the join exists for.
#   STATE, where ST is the latest. A polygon 78% under a mapped building
#     footprint, labelled `MaralBadlands`: `House` scores 0.57 and the incumbent
#     0.44 -- a 0.13 gap, worth 0.30 * 0.13 = 0.04 of the total, which does NOT
#     clear SWITCH_MARGIN. Correct under the policy: a mapped footprint is what
#     the ground USED to be, bare rubble is what ST saw later, and S1 raises it
#     as a change candidate instead.
#
# So the reference term can defend a label, can put a candidate on the shortlist
# that `class_distance` would never offer, and can flip one only when OSM is the
# authority on that axis. All of it moves with `tags.RELIABILITY`, which is a
# guess, and with `trust.SUPPORT`/`trust.CONTRADICT`, which is the policy.
W_REFERENCE = 0.30

# What a candidate's reference term is worth when OSM had an opinion about at
# least one OTHER candidate in the same ranking but was silent about THIS one.
# Without this, a ranking where OSM covers some candidates and not others
# compares a blended score for the covered ones against a raw `own` for the
# silent ones -- two different scales in one sort, which has measurably
# penalised candidates for having OSM support. 0.5 is this file's own existing
# convention for "no evidence either way" (see `_context_score`'s neutral
# start and `_morphology_score`'s no-constraint return), so blending with it
# changes a candidate's score by exactly zero on average and never argues for
# or against anything by itself -- it only puts every candidate through the
# same formula. See `Evidence.total`.
NEUTRAL_REFERENCE = 0.5

# Candidates are drawn from classes within this semantic distance of the
# incumbent, plus the dominant neighbouring classes.
CANDIDATE_MAX_DISTANCE = 0.5

# Switch the label only if the challenger beats the incumbent by this margin.
# Deliberately high: the cost of a corruption exceeds the value of a correction.
SWITCH_MARGIN = 0.15

# If the top candidates differ *only* by lithology, no amount of geometry will
# separate them -- emit UNDECIDABLE instead of a coin flip.
UNDECIDABLE_MARGIN = 0.08

# Ceiling on how far a SATISFIED slope band can lift the morphology score above
# the 0.5 neutral (the actual lift is this times the band's specificity, see
# _morphology_score). 0.4, not 0.5, so that W_MORPHOLOGY * bonus < SWITCH_MARGIN:
# a satisfied constraint alone -- context and geometry equal -- can never flip a
# label. A VIOLATED constraint, decaying towards 0.0, still can. The asymmetry
# is deliberate: satisfaction is weak evidence for, violation strong evidence
# against, and the cost of a corruption exceeds the value of a correction.
SATISFIED_MAX_BONUS = 0.4
assert W_MORPHOLOGY * SATISFIED_MAX_BONUS < SWITCH_MARGIN, \
    "a satisfied morphology band alone must not be able to clear the switch margin"

# Full range of the slope input, degrees. A band this wide constrains nothing.
SLOPE_RANGE_DEG = 90.0

# Expected slope band per morphology, degrees. Coarse, and the biggest single
# source of error in this module.
MORPHOLOGY_SLOPE = {
    "Terrace": (0.0, 6.0),
    "SmoothRockSlopes": (6.0, 30.0),
    "StoneyTerrain": (0.0, 20.0),
    "RockyTerrain": (10.0, 45.0),
    "BeddedRock": (8.0, 35.0),
    "RockDipSlope": (10.0, 40.0),
    "Boulder": (15.0, 60.0),
    "Badlands": (12.0, 50.0),
}

# Morphologies that require a single coherent facet.
COHERENT_FACET = {"RockDipSlope", "Terrace"}
COHERENT_FACET_MAX_CIRCVAR = 0.3

# Expected area band per class, m2. A guess, used only when an analyst has not
# written down a real one (see `_SCALE_AREA_RE` / `_parse_scale_area` below,
# which is now consulted first): only a few classes have a real constraint
# here; the rest are unconstrained.
AREA_BAND = {
    "Car": (4.0, 30.0),
    "House": (25.0, 2000.0),
    "BrickWall": (2.0, 500.0),
}

# An analyst's `scale` class note ("typical size and shape as a polygon", see
# `class_notes.py`) is prose, not a structured value -- there is no parsed
# `scale` property on `ClassNote` to consume. This picks out the one shape of
# prose this module knows how to act on: an explicit numeric range in square
# metres ("10-40 m2", "25 to 2000 m2", "4-30 m²"). Anything else -- a shape
# description with no numbers, an elongation-only note, a single figure -- is
# left alone rather than guessed at, and `_geometry_score` falls back to
# `AREA_BAND` (or unconstrained) exactly as if the note did not exist.
_SCALE_AREA_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<hi>\d+(?:\.\d+)?)\s*"
    r"(?:m2|m²|sq\s*\.?\s*m|sqm|square\s+met(?:re|er)s?)",
    re.IGNORECASE,
)

# Classes whose shape is intrinsically linear.
LINEAR = {"PavedRoad", "DirtRoad", "DirtRoadB", "BrickWall"}
LINEAR_MIN_ELONGATION = 2.0


def _parse_scale_area(text: str) -> tuple[float, float] | None:
    """Pull a `lo-hi m2` area range out of an analyst's free-text `scale`
    note, or None if the note does not contain one (see `_SCALE_AREA_RE`)."""
    m = _SCALE_AREA_RE.search(text)
    if not m:
        return None
    lo, hi = float(m.group("lo")), float(m.group("hi"))
    return (lo, hi) if lo <= hi else (hi, lo)


@dataclass
class Evidence:
    context: float
    morphology: float
    geometry: float
    notes: list[str]
    # The reference term: what an independent map (OSM) says about this
    # candidate here. `None` means OSM had nothing to say about THIS candidate
    # -- no OSM layer at all, or a layer that is silent about it specifically.
    # When no candidate anywhere in the ranking got a reference score, `total`
    # is EXACTLY the three-term score this module computed before the layer
    # existed. That identity is a test
    # (`test_osm.py::test_s2_is_bit_identical_without_osm`), not a hope. When a
    # SIBLING candidate did get a reference score, see `ranking_has_reference`.
    reference: float | None = None
    # Set by `adjudicate()`, once, after the whole ranking's candidates exist
    # and before anything sorts or reads `.total`: True if ANY candidate in
    # this same ranking got a real reference score. It exists so `total` can
    # apply one formula to every candidate in a ranking rather than switching
    # formulas per candidate -- see the comment on `total` below.
    ranking_has_reference: bool = False

    @property
    def own(self) -> float:
        """The three terms derived from the label raster itself."""
        return (W_CONTEXT * self.context
                + W_MORPHOLOGY * self.morphology
                + W_GEOMETRY * self.geometry)

    @property
    def total(self) -> float:
        if self.reference is not None:
            return (1.0 - W_REFERENCE) * self.own + W_REFERENCE * self.reference
        if self.ranking_has_reference:
            # OSM spoke about a sibling candidate in this ranking but not about
            # this one. Falling back to raw `own` here (as if OSM had said
            # nothing at all) would compare this candidate on a different
            # scale than the one OSM DID cover -- so it gets the same blended
            # formula, with a neutral stand-in for the missing reference,
            # rather than a discount for OSM's silence.
            return (1.0 - W_REFERENCE) * self.own + W_REFERENCE * NEUTRAL_REFERENCE
        return self.own


@dataclass
class Adjudication:
    region_id: int
    incumbent: str
    ranked: list[tuple[str, Evidence]]
    verdict: str          # KEEP | SWITCH:<name> | UNDECIDABLE-NEEDS-<data>
    rationale: str
    # What the reference map said, kept whole so the render can show it. None
    # when no OSM layer was joined.
    osm: object | None = None

    @property
    def top(self) -> str:
        return self.ranked[0][0]


def candidates(ridx: RegionIndex, r: Region, osm_extra=(),
               notes=None) -> list[int]:
    """Short list. Never the full 47 -- adjudication is a multiple-choice
    question, not a re-run of perception."""
    out = {r.class_id}
    # An analyst's `confused_with` beats `class_distance` every time: the
    # taxonomy metric is a guess about which classes are alike, and this is a
    # record of which ones actually get mixed up in practice. Empty until
    # somebody fills in the class notes (see `class_notes.default()`), at
    # which point the shortlist changes and this is the only place it changes.
    if notes is not None:
        note = notes.get(r.class_name)
        if note:
            for name, _why in note.confused_with:
                out.add(cid(name))
    for c in range(N_CLASSES):
        if class_distance(r.class_id, c) <= CANDIDATE_MAX_DISTANCE:
            out.add(c)
    nh = ridx.neighbor_class_hist(r)
    for c, _ in sorted(nh.items(), key=lambda kv: -kv[1])[:3]:
        out.add(c)
    # Whatever the reference map thinks is in play, regardless of taxonomy
    # distance: a polygon labelled Rendzina that sits on a way tagged
    # surface=asphalt needs PavedRoad on the shortlist, and class_distance will
    # never put it there (soil to road is the 1.0 maximum).
    out.update(osm_extra)
    return sorted(out)


def _context_score(ridx: RegionIndex, r: Region, cand: int) -> tuple[float, list[str]]:
    name = BY_ID[cand].name
    nh = ridx.neighbor_class_hist(r)
    total = sum(nh.values()) or 1.0
    score, notes = 0.5, []

    for p in PRIORS_BY_SUBJECT.get(name, ()):
        if not p.others:
            continue
        share = sum(nh.get(cid(n), 0.0) for n in p.others) / total
        if p.kind == "expects":
            score += 0.5 * share
            notes.append(f"{share:.0%} boundary with expected {{{', '.join(p.others[:3])}...}}")
        elif p.kind == "contradicts":
            score -= 0.6 * share
            if share > 0.05:
                notes.append(f"{share:.0%} boundary with contradicting context")

    # Lithology continuity: a rock unit surrounded by its own lithology fits.
    lit = BY_ID[cand].lithology
    if lit:
        same = sum(v for k, v in nh.items() if BY_ID[k].lithology == lit) / total
        score += 0.3 * same - 0.15 * (1 - same)
        notes.append(f"{same:.0%} boundary with same-lithology units")

    return max(0.0, min(1.0, score)), notes


def _morphology_score(r: Region, cand: int,
                      has_terrain: bool = True) -> tuple[float, list[str]]:
    d = BY_ID[cand]
    if not d.morphology:
        return 0.5, []
    if not has_terrain:
        # Slope and aspect are the only inputs to this term. With no DEM they
        # are 0.0 for every region, which scores Terrace perfectly and Boulder
        # at zero everywhere -- a systematic push towards flat morphologies that
        # reads exactly like evidence. Return the neutral score and say why,
        # rather than adjudicating on a constant.
        return 0.5, ["morphology unscored: no DEM, slope and aspect unmeasured"]
    notes = []
    lo, hi = MORPHOLOGY_SLOPE.get(d.morphology, (0.0, SLOPE_RANGE_DEG))
    # A satisfied band is evidence in proportion to how SPECIFIC the band is:
    # a 0-20 deg band containing a 3 deg slope has said almost nothing, a 0-6
    # deg terrace band containing it has said a lot. So the score is
    #     0.5 + SATISFIED_MAX_BONUS * specificity,
    #     specificity = 1 - band_width / 90 (the full slope range),
    # which makes an unconstrained (0-90) morphology score exactly the 0.5
    # neutral a no-morphology candidate gets -- HAVING a satisfiable constraint
    # is not evidence by itself. (The previous flat 1.0 handed every rock class
    # whose wide band happened to contain the slope a free +0.5*W_MORPHOLOGY
    # over every soil/vegetation candidate, larger than SWITCH_MARGIN.)
    specificity = 1.0 - (hi - lo) / SLOPE_RANGE_DEG
    satisfied = 0.5 + SATISFIED_MAX_BONUS * specificity
    if lo <= r.mean_slope <= hi:
        score = satisfied
        if specificity > 0.0:
            notes.append(f"slope {r.mean_slope:.1f} deg fits {lo}-{hi} for "
                         f"{d.morphology} (band specificity {specificity:.2f})")
    else:
        # Same 1/15-per-degree decay as before, but anchored at the satisfied
        # score rather than at 1.0: a slope just outside the band must never
        # outscore one inside it.
        miss = min(abs(r.mean_slope - lo), abs(r.mean_slope - hi))
        score = max(0.0, satisfied - miss / 15.0)
        notes.append(f"slope {r.mean_slope:.1f} deg outside {lo}-{hi} for {d.morphology}")

    if d.morphology in COHERENT_FACET:
        if r.aspect_circvar > COHERENT_FACET_MAX_CIRCVAR:
            score *= 0.3
            notes.append(
                f"aspect circvar {r.aspect_circvar:.2f} too incoherent for a "
                f"single {d.morphology} facet"
            )
        else:
            notes.append(f"aspect coherent ({r.aspect_circvar:.2f})")
    return score, notes


def _geometry_score(r: Region, cand: int, notes=None) -> tuple[float, list[str]]:
    """`notes`, when given, is the class-notes `NoteSet` (see `class_notes.py`
    and `candidates()` above, which threads it the same way). An analyst's
    `scale` note on this candidate's class beats the hardcoded `AREA_BAND`
    guess when it parses into a real area range; `AREA_BAND` is the fallback,
    and no constraint at all -- the pre-existing behaviour -- is what is left
    when neither has one."""
    name = BY_ID[cand].name
    score, msgs = 0.5, []
    band, band_source = None, ""
    note = notes.get(name) if notes is not None else None
    if note is not None and note.filled("scale"):
        band = _parse_scale_area(note.get("scale"))
        if band is not None:
            band_source = "analyst scale note"
    if band is None and name in AREA_BAND:
        band = AREA_BAND[name]
        band_source = "AREA_BAND guess"
    if band is not None:
        lo, hi = band
        if lo <= r.area_m2 <= hi:
            score = 1.0
        else:
            score = 0.05
            msgs.append(f"area {r.area_m2:.0f} m2 outside {lo}-{hi} for {name} "
                        f"(per {band_source})")
    if name in LINEAR:
        if r.elongation >= LINEAR_MIN_ELONGATION:
            score = max(score, 0.9)
            msgs.append(f"elongation {r.elongation:.1f} fits a linear feature")
        else:
            score = min(score, 0.2)
            msgs.append(f"elongation {r.elongation:.1f} too blobby for {name}")
    return score, msgs


def adjudicate(ridx: RegionIndex, region_id: int,
               osm=None, raster=None) -> Adjudication:
    r = ridx.get(region_id)

    ref_ctx = None
    osm_extra: tuple[int, ...] = ()
    if osm is not None:
        if raster is None:
            raise ValueError("the reference term needs the raster the OSM layer "
                             "was burned onto: adjudicate(..., osm=layer, raster=r)")
        from ..osm import evidence as osm_evidence

        ref_ctx = osm_evidence.region_context(ridx, raster, osm, region_id)
        osm_extra = tuple(osm_evidence.suggested_candidates(ref_ctx))

    from .. import class_notes as _class_notes

    notes = _class_notes.default()
    incumbent_note = notes.get(r.class_name)

    ranked: list[tuple[str, Evidence]] = []
    for cand in candidates(ridx, r, osm_extra=osm_extra, notes=notes):
        ctx, n1 = _context_score(ridx, r, cand)
        mor, n2 = _morphology_score(r, cand, ridx.has_terrain)
        geo, n3 = _geometry_score(r, cand, notes=notes)
        ref, n4 = (None, [])
        if ref_ctx is not None:
            from ..osm import evidence as osm_evidence

            ref, n4 = osm_evidence.reference_score(ref_ctx, cand)
        n5: list[str] = []
        if incumbent_note is not None:
            for name, why in incumbent_note.confused_with:
                if name == BY_ID[cand].name and why:
                    n5.append(f"analyst note on {r.class_name} vs {name}: {why}")
        ranked.append((BY_ID[cand].name,
                       Evidence(ctx, mor, geo, n1 + n2 + n3 + n4 + n5,
                                reference=ref)))
    # Every candidate in a ranking must be scored through the same formula: if
    # ANY of them got a real reference score, everyone does (a neutral one for
    # the rest) rather than mixing a blended scale for the OSM-covered
    # candidates with a raw `own` scale for the ones OSM was silent about. Must
    # run before anything reads `.total` -- nothing above this line does.
    any_ref = any(ev.reference is not None for _n, ev in ranked)
    for _n, ev in ranked:
        ev.ranking_has_reference = any_ref
    ranked.sort(key=lambda kv: -kv[1].total)

    incumbent = r.class_name
    top_name, top_ev = ranked[0]
    inc_ev = next(ev for n, ev in ranked if n == incumbent)

    # Do any of the leading candidates differ only by lithology? Then RGB and
    # geometry cannot separate them, and neither can we. Scan every pair in
    # the near-top set, not just ranks 1 and 2: an unrelated candidate wedged
    # between two lithology twins does not make them any more decidable.
    # Every member of the set is within UNDECIDABLE_MARGIN of the leader, so
    # every pair in it is within the margin of each other as well.
    near_top = [n for n, ev in ranked
                if ranked[0][1].total - ev.total < UNDECIDABLE_MARGIN]
    for i, name_a in enumerate(near_top):
        a = BY_ID[cid(name_a)]
        for name_b in near_top[i + 1:]:
            b = BY_ID[cid(name_b)]
            if a.lithology and b.lithology and a.lithology != b.lithology \
                    and a.morphology == b.morphology:
                return Adjudication(
                    region_id, incumbent, ranked, "UNDECIDABLE-NEEDS-geological-map",
                    f"{a.name} and {b.name} differ only by lithology and score within "
                    f"{UNDECIDABLE_MARGIN} of the leader; this separation is not "
                    f"present in the segmentation, the DEM, or the imagery. A "
                    f"geological map settles it.",
                    osm=ref_ctx,
                )

    if top_name != incumbent and top_ev.total - inc_ev.total >= SWITCH_MARGIN:
        driver = ""
        if top_ev.reference is not None or inc_ev.reference is not None:
            # Would the switch still happen on the label raster alone? If not,
            # the reference map is what flipped it, and a reviewer has to know
            # that -- it is the one input here that can be wrong for reasons
            # nothing in the segmentation would reveal.
            own_gap = top_ev.own - inc_ev.own
            driver = (" -- driven by the OSM reference term; on the label raster "
                      f"alone the gap is {own_gap:+.2f}, "
                      + ("still a switch" if own_gap >= SWITCH_MARGIN
                         else "not enough to switch"))
        return Adjudication(
            region_id, incumbent, ranked, f"SWITCH:{top_name}",
            f"{top_name} scores {top_ev.total:.2f} vs {inc_ev.total:.2f} for the "
            f"incumbent ({', '.join(top_ev.notes[:2]) or 'no specific cue'})"
            + driver,
            osm=ref_ctx,
        )
    return Adjudication(
        region_id, incumbent, ranked, "KEEP",
        f"no challenger beats the incumbent by the {SWITCH_MARGIN} margin "
        f"(best challenger {top_name} at {top_ev.total:.2f})"
        if top_name != incumbent else
        f"incumbent is also the best-supported candidate ({inc_ev.total:.2f})",
        osm=ref_ctx,
    )


def render(adj: Adjudication, top_k: int = 4) -> str:
    has_ref = any(ev.reference is not None for _n, ev in adj.ranked)
    lines = [
        f"# region {adj.region_id}: incumbent {adj.incumbent}",
        f"# verdict: {adj.verdict}",
        f"# {adj.rationale}",
    ]
    if adj.osm is not None:
        lines.append(f"# OSM says: {adj.osm.describe()}")
        if not has_ref:
            lines.append("# the reference term is unused here: OSM had nothing to "
                         "say about any candidate, so this is the label raster's "
                         "own verdict")
    lines.append("candidate\ttotal\tcontext\tmorph\tgeom"
                 + ("\tosm" if has_ref else "") + "\tnotes")
    for name, ev in adj.ranked[:top_k]:
        mark = " *" if name == adj.incumbent else ""
        ref = ("\t" + ("-" if ev.reference is None else f"{ev.reference:.2f}")) \
            if has_ref else ""
        lines.append(
            f"{name}{mark}\t{ev.total:.2f}\t{ev.context:.2f}\t{ev.morphology:.2f}\t"
            f"{ev.geometry:.2f}{ref}\t{'; '.join(ev.notes[:2])}"
        )
    if len(adj.ranked) > top_k:
        lines.append(f"# NOTE: {len(adj.ranked) - top_k} lower-scoring candidates omitted")
    return "\n".join(lines)


def ledger(adjudications: list[Adjudication]) -> str:
    """The metric that matters is not accuracy -- it is corrections minus
    corruptions. Without ground truth this only shows the *shape* of the
    override behaviour; wire it to labelled regions to get the real number."""
    kinds: dict[str, int] = {}
    for a in adjudications:
        k = a.verdict.split(":")[0]
        kinds[k] = kinds.get(k, 0) + 1
    total = len(adjudications) or 1
    lines = ["# override ledger (no ground truth -- shape only)", "verdict\tn\tshare"]
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"{k}\t{v}\t{v / total:.0%}")
    lines.append(
        "# net gain = corrections - corruptions. Needs labelled regions to compute; "
        "if it is negative, S2 is hurting you."
    )
    return "\n".join(lines)
