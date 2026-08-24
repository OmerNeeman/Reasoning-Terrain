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

# Candidates are drawn from classes within this semantic distance of the
# incumbent, plus the dominant neighbouring classes.
CANDIDATE_MAX_DISTANCE = 0.5

# Switch the label only if the challenger beats the incumbent by this margin.
# Deliberately high: the cost of a corruption exceeds the value of a correction.
SWITCH_MARGIN = 0.15

# If the top candidates differ *only* by lithology, no amount of geometry will
# separate them -- emit UNDECIDABLE instead of a coin flip.
UNDECIDABLE_MARGIN = 0.08

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

# Expected area band per class, m2. Only a few classes have a real constraint;
# the rest are unconstrained.
AREA_BAND = {
    "Car": (4.0, 30.0),
    "House": (25.0, 2000.0),
    "BrickWall": (2.0, 500.0),
}

# Classes whose shape is intrinsically linear.
LINEAR = {"PavedRoad", "DirtRoad", "DirtRoadB", "BrickWall"}
LINEAR_MIN_ELONGATION = 2.0


@dataclass
class Evidence:
    context: float
    morphology: float
    geometry: float
    notes: list[str]

    @property
    def total(self) -> float:
        return (W_CONTEXT * self.context
                + W_MORPHOLOGY * self.morphology
                + W_GEOMETRY * self.geometry)


@dataclass
class Adjudication:
    region_id: int
    incumbent: str
    ranked: list[tuple[str, Evidence]]
    verdict: str          # KEEP | SWITCH:<name> | UNDECIDABLE-NEEDS-<data>
    rationale: str

    @property
    def top(self) -> str:
        return self.ranked[0][0]


def candidates(ridx: RegionIndex, r: Region) -> list[int]:
    """Short list. Never the full 47 -- adjudication is a multiple-choice
    question, not a re-run of perception."""
    out = {r.class_id}
    for c in range(N_CLASSES):
        if class_distance(r.class_id, c) <= CANDIDATE_MAX_DISTANCE:
            out.add(c)
    nh = ridx.neighbor_class_hist(r)
    for c, _ in sorted(nh.items(), key=lambda kv: -kv[1])[:3]:
        out.add(c)
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
    lo, hi = MORPHOLOGY_SLOPE.get(d.morphology, (0.0, 90.0))
    if lo <= r.mean_slope <= hi:
        score = 1.0
    else:
        miss = min(abs(r.mean_slope - lo), abs(r.mean_slope - hi))
        score = max(0.0, 1.0 - miss / 15.0)
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


def _geometry_score(r: Region, cand: int) -> tuple[float, list[str]]:
    name = BY_ID[cand].name
    score, notes = 0.5, []
    if name in AREA_BAND:
        lo, hi = AREA_BAND[name]
        if lo <= r.area_m2 <= hi:
            score = 1.0
        else:
            score = 0.05
            notes.append(f"area {r.area_m2:.0f} m2 outside {lo}-{hi} for {name}")
    if name in LINEAR:
        if r.elongation >= LINEAR_MIN_ELONGATION:
            score = max(score, 0.9)
            notes.append(f"elongation {r.elongation:.1f} fits a linear feature")
        else:
            score = min(score, 0.2)
            notes.append(f"elongation {r.elongation:.1f} too blobby for {name}")
    return score, notes


def adjudicate(ridx: RegionIndex, region_id: int) -> Adjudication:
    r = ridx.get(region_id)
    ranked: list[tuple[str, Evidence]] = []
    for cand in candidates(ridx, r):
        ctx, n1 = _context_score(ridx, r, cand)
        mor, n2 = _morphology_score(r, cand, ridx.has_terrain)
        geo, n3 = _geometry_score(r, cand)
        ranked.append((BY_ID[cand].name, Evidence(ctx, mor, geo, n1 + n2 + n3)))
    ranked.sort(key=lambda kv: -kv[1].total)

    incumbent = r.class_name
    top_name, top_ev = ranked[0]
    inc_ev = next(ev for n, ev in ranked if n == incumbent)

    # Do the leading candidates differ only by lithology? Then RGB and geometry
    # cannot separate them, and neither can we.
    if len(ranked) > 1:
        a, b = BY_ID[cid(ranked[0][0])], BY_ID[cid(ranked[1][0])]
        close = abs(ranked[0][1].total - ranked[1][1].total) < UNDECIDABLE_MARGIN
        if close and a.lithology and b.lithology and a.lithology != b.lithology \
                and a.morphology == b.morphology:
            return Adjudication(
                region_id, incumbent, ranked, "UNDECIDABLE-NEEDS-geological-map",
                f"{a.name} and {b.name} differ only by lithology and score within "
                f"{UNDECIDABLE_MARGIN}; this separation is not present in the "
                f"segmentation, the DEM, or the imagery. A geological map settles it.",
            )

    if top_name != incumbent and top_ev.total - inc_ev.total >= SWITCH_MARGIN:
        return Adjudication(
            region_id, incumbent, ranked, f"SWITCH:{top_name}",
            f"{top_name} scores {top_ev.total:.2f} vs {inc_ev.total:.2f} for the "
            f"incumbent ({', '.join(top_ev.notes[:2]) or 'no specific cue'})",
        )
    return Adjudication(
        region_id, incumbent, ranked, "KEEP",
        f"no challenger beats the incumbent by the {SWITCH_MARGIN} margin "
        f"(best challenger {top_name} at {top_ev.total:.2f})"
        if top_name != incumbent else
        f"incumbent is also the best-supported candidate ({inc_ev.total:.2f})",
    )


def render(adj: Adjudication, top_k: int = 4) -> str:
    lines = [
        f"# region {adj.region_id}: incumbent {adj.incumbent}",
        f"# verdict: {adj.verdict}",
        f"# {adj.rationale}",
        "candidate\ttotal\tcontext\tmorph\tgeom\tnotes",
    ]
    for name, ev in adj.ranked[:top_k]:
        mark = " *" if name == adj.incumbent else ""
        lines.append(
            f"{name}{mark}\t{ev.total:.2f}\t{ev.context:.2f}\t{ev.morphology:.2f}\t"
            f"{ev.geometry:.2f}\t{'; '.join(ev.notes[:2])}"
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
