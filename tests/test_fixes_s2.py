"""Regression tests for three verified fixes in S2 and taxonomy.class_distance.

1. `_morphology_score` gave every candidate whose (often very wide) slope band
   happened to contain the region's slope a perfect 1.0 -- a free
   +0.5*W_MORPHOLOGY head start over every no-morphology class, larger than
   SWITCH_MARGIN, for merely HAVING a satisfiable constraint.
2. The UNDECIDABLE gate compared only ranked[0] vs ranked[1], so lithology
   twins split by an unrelated candidate in between skipped abstention.
3. `taxonomy.class_distance` normalised the ordinal series by len(series)
   instead of len(series)-1, giving the two series inconsistent, undocumented
   scales (vegetation max 0.75, roads max 0.667).
"""

import numpy as np
import pytest

from segmap_digest import synth
from segmap_digest.index import Region, RegionIndex, build_regions
from segmap_digest.solutions import s1_audit, s2_adjudicate
from segmap_digest.taxonomy import BY_ID, CLASSES, cid, class_distance


@pytest.fixture(scope="module")
def tile():
    return synth.generate(size=384, seed=5)


@pytest.fixture(scope="module")
def ridx(tile):
    return build_regions(tile)


def _region(class_name: str, mean_slope: float = 0.0,
            aspect_circvar: float = 0.0) -> Region:
    """A minimal hand-built region -- geometry values are unconstrained filler."""
    return Region(
        id=1, class_id=cid(class_name), area_px=100, area_m2=100.0,
        perimeter_m=40.0, compactness=0.8, elongation=1.2,
        bbox=(0, 0, 10, 10), centroid=(5.0, 5.0),
        mean_slope=mean_slope, aspect_circvar=aspect_circvar,
    )


def _class_with_morphology(morph: str) -> int:
    return next(c.id for c in CLASSES if c.morphology == morph)


# --- finding 1: satisfied wide slope bands are weak evidence ----------------

def test_s2_satisfied_band_advantage_is_bounded_and_specificity_scaled():
    """Structural assertion (robust to fixture drift): a SATISFIED slope band
    is evidence in proportion to the band's specificity, and even the most
    specific band, weighted by W_MORPHOLOGY, must stay below SWITCH_MARGIN --
    'my band contains your slope' is never, on its own, grounds to overwrite
    a label. A VIOLATED band (decaying towards 0.0) still can swing past the
    margin: satisfaction is weak evidence for, violation strong evidence
    against."""
    scored = []
    for morph, (lo, hi) in s2_adjudicate.MORPHOLOGY_SLOPE.items():
        cand = _class_with_morphology(morph)
        r = _region("HydromorpicSoil", mean_slope=(lo + hi) / 2.0)
        score, _ = s2_adjudicate._morphology_score(r, cand)
        # evidence, but never certainty, and never above-neutral by enough to
        # clear the switch margin unaided
        assert 0.5 < score < 1.0, f"{morph}: satisfied score {score}"
        adv = s2_adjudicate.W_MORPHOLOGY * (score - 0.5)
        assert adv < s2_adjudicate.SWITCH_MARGIN, \
            f"{morph}: satisfied-band advantage {adv:.3f} can flip a label alone"
        scored.append((hi - lo, score))
    # wider band => weaker evidence, strictly
    scored.sort()
    for (w1, s1), (w2, s2) in zip(scored, scored[1:]):
        if w1 < w2:
            assert s1 > s2, f"width {w1} scored {s1} <= width {w2} at {s2}"


def test_s2_unconstrained_morphology_scores_exactly_neutral():
    """A morphology with no entry in MORPHOLOGY_SLOPE falls back to the full
    0-90 band. Zero specificity must score exactly the 0.5 a no-morphology
    candidate gets: HAVING a constraint is not evidence."""
    cand = _class_with_morphology("HardRockLineament")   # no slope band defined
    assert BY_ID[cand].morphology not in s2_adjudicate.MORPHOLOGY_SLOPE
    score, _ = s2_adjudicate._morphology_score(_region("HydromorpicSoil", 3.0), cand)
    assert score == 0.5


def test_s2_no_morphology_candidate_stays_neutral():
    score, notes = s2_adjudicate._morphology_score(
        _region("HydromorpicSoil", 3.0), cid("TerraRosa"))
    assert score == 0.5 and notes == []


def test_s2_violation_never_outscores_satisfaction():
    """The violation decay must anchor at the satisfied score, not at 1.0 --
    a slope just outside the band must never outscore one inside it."""
    cand = _class_with_morphology("StoneyTerrain")       # band 0-20
    lo, hi = s2_adjudicate.MORPHOLOGY_SLOPE["StoneyTerrain"]
    inside, _ = s2_adjudicate._morphology_score(_region("HydromorpicSoil", hi - 0.1), cand)
    just_out, _ = s2_adjudicate._morphology_score(_region("HydromorpicSoil", hi + 0.5), cand)
    far_out, _ = s2_adjudicate._morphology_score(_region("HydromorpicSoil", hi + 30.0), cand)
    assert just_out < inside
    assert far_out < just_out
    assert far_out == 0.0


def test_s2_fixture_hydromorphic_regions_not_outranked_on_morphology_alone(ridx):
    """The observed case: S1 flags the HydromorpicSoil fringe regions (slope
    violates their 0-5 deg prior). Their slopes sit inside the wide bands of
    rock challengers (MaralSmoothRockSlopes 6-30, BasaltRockyTerrain 10-45),
    which pre-fix took morph 1.0 vs the incumbent's neutral 0.5 -- a +0.175
    head start exceeding SWITCH_MARGIN (0.15). Assert the morphology term can
    no longer clear the switch margin on its own for any challenger, and that
    none of these regions gets switched to a rock class."""
    rep = s1_audit.run(ridx)
    rids = sorted({f.region_id for f in rep.findings
                   if f.class_name == "HydromorpicSoil"})
    assert rids, "fixture is expected to flag HydromorpicSoil regions via S1"

    exercised = False
    for rid in rids:
        adj = s2_adjudicate.adjudicate(ridx, rid)
        assert not adj.verdict.startswith("SWITCH"), \
            f"region {rid} switched away from HydromorpicSoil: {adj.rationale}"
        inc_ev = next(ev for n, ev in adj.ranked if n == adj.incumbent)
        for name, ev in adj.ranked:
            if name == adj.incumbent:
                continue
            morph_adv = s2_adjudicate.W_MORPHOLOGY * (ev.morphology - inc_ev.morphology)
            assert morph_adv < s2_adjudicate.SWITCH_MARGIN, \
                (f"region {rid}: {name} gains {morph_adv:.3f} from morphology "
                 f"alone vs the no-morphology incumbent")
            if ev.morphology > 0.5:
                exercised = True    # a satisfied-band rock challenger was scored
    assert exercised, "fixture no longer exercises a satisfied-band challenger"


# --- finding 2: UNDECIDABLE gate must scan all near-top pairs ---------------

def _fixed_score_index(monkeypatch, totals: dict[str, float],
                       incumbent: str) -> RegionIndex:
    """A one-region index plus monkeypatched evidence terms that pin each
    candidate's total (the three weights sum to 1.0) -- a constructed ranking."""
    def fake(cand: int) -> float:
        return totals.get(BY_ID[cand].name, 0.10)
    monkeypatch.setattr(s2_adjudicate, "_context_score",
                        lambda ridx, r, cand: (fake(cand), []))
    monkeypatch.setattr(s2_adjudicate, "_morphology_score",
                        lambda r, cand, has_terrain=True: (fake(cand), []))
    monkeypatch.setattr(s2_adjudicate, "_geometry_score",
                        lambda r, cand, notes=None: (fake(cand), []))
    return RegionIndex(regions=[_region(incumbent)],
                       label_array=np.zeros((10, 10), dtype=np.int32),
                       has_terrain=True)


def test_s2_undecidable_gate_scans_past_an_interloper(monkeypatch):
    """Lithology twins at ranks 1 and 3, split by a same-lithology candidate
    at rank 2. The twins are 0.05 apart -- inside UNDECIDABLE_MARGIN (0.08) --
    so the verdict must be abstention, not KEEP: rank 2 being unrelated does
    not make Limestone-vs-Dolomite any more decidable."""
    ridx = _fixed_score_index(monkeypatch, {
        "LimestoneStoneyTerrain": 0.60,   # rank 1, incumbent
        "LimestoneBoulder": 0.58,         # rank 2: same lithology -- not a twin
        "DolomiteStoneyTerrain": 0.55,    # rank 3: twin of rank 1, within margin
    }, incumbent="LimestoneStoneyTerrain")
    adj = s2_adjudicate.adjudicate(ridx, 1)
    assert adj.verdict.startswith("UNDECIDABLE-NEEDS-"), adj.verdict
    names = [n for n, _ in adj.ranked[:3]]
    assert names == ["LimestoneStoneyTerrain", "LimestoneBoulder",
                     "DolomiteStoneyTerrain"], "constructed ranking drifted"


def test_s2_undecidable_gate_still_fires_on_adjacent_twins(monkeypatch):
    ridx = _fixed_score_index(monkeypatch, {
        "LimestoneStoneyTerrain": 0.60,
        "DolomiteStoneyTerrain": 0.57,
    }, incumbent="LimestoneStoneyTerrain")
    adj = s2_adjudicate.adjudicate(ridx, 1)
    assert adj.verdict.startswith("UNDECIDABLE-NEEDS-")


def test_s2_twin_far_below_leader_does_not_force_abstention(monkeypatch):
    """A lithology twin outside the margin is a settled ranking, not a tie --
    abstention there would just be a different way of refusing to answer a
    question the evidence did answer."""
    ridx = _fixed_score_index(monkeypatch, {
        "LimestoneStoneyTerrain": 0.60,
        "LimestoneBoulder": 0.58,
        "DolomiteStoneyTerrain": 0.40,    # 0.20 behind: decidable
    }, incumbent="LimestoneStoneyTerrain")
    adj = s2_adjudicate.adjudicate(ridx, 1)
    assert adj.verdict == "KEEP"


# --- finding 3: one scale for both ordinal series ----------------------------

def test_class_distance_series_share_one_documented_scale():
    """Both ordinal series use distance = 0.75 * |i-j| / (len-1): the in-series
    maximum is 0.75 for every series and stays strictly below the 1.0
    cross-superclass maximum. Pre-fix the roads series was normalised by len,
    so its maximum was 0.667 on an undocumented, series-dependent scale."""
    # in-series maxima: both exactly 0.75
    assert class_distance(cid("DryGrassland"), cid("Maquis")) == pytest.approx(0.75)
    assert class_distance(cid("DirtRoadB"), cid("PavedRoad")) == pytest.approx(0.75)
    # adjacent steps: 0.25 on the 4-stage vegetation series, 0.375 on the
    # 3-grade road series
    assert class_distance(cid("DryGrassland"), cid("Batha")) == pytest.approx(0.25)
    assert class_distance(cid("Batha"), cid("Garigue")) == pytest.approx(0.25)
    assert class_distance(cid("DirtRoadB"), cid("DirtRoad")) == pytest.approx(0.375)
    assert class_distance(cid("DirtRoad"), cid("PavedRoad")) == pytest.approx(0.375)
    # symmetric, and strictly below the cross-superclass maximum
    assert class_distance(cid("PavedRoad"), cid("DirtRoadB")) \
        == class_distance(cid("DirtRoadB"), cid("PavedRoad")) < 1.0


def test_class_distance_series_change_does_not_move_s2_shortlists():
    """CANDIDATE_MAX_DISTANCE is 0.5: adjacent road grades (0.375, was 0.333)
    stay inside it and the series extremes (0.75, was 0.667) stay outside --
    the rescale must not change which classes S2 shortlists."""
    m = s2_adjudicate.CANDIDATE_MAX_DISTANCE
    assert class_distance(cid("DirtRoadB"), cid("DirtRoad")) <= m
    assert class_distance(cid("DirtRoad"), cid("PavedRoad")) <= m
    assert class_distance(cid("DirtRoadB"), cid("PavedRoad")) > m
    assert class_distance(cid("DryGrassland"), cid("Batha")) <= m
    assert class_distance(cid("DryGrassland"), cid("Maquis")) > m
