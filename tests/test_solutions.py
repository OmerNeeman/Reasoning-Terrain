import numpy as np
import pytest

from segmap_digest import synth
from segmap_digest.index import build_chips, build_regions
from segmap_digest.solutions import (
    s1_audit,
    s2_adjudicate,
    s3_triage,
    s4_products,
    s5_query,
    s6_change,
)
from segmap_digest.taxonomy import cid


@pytest.fixture(scope="module")
def tile():
    return synth.generate(size=384, seed=5)


@pytest.fixture(scope="module")
def ridx(tile):
    return build_regions(tile)


# --- shared infrastructure -------------------------------------------------

def test_label_array_matches_region_ids(tile, ridx):
    """Regression: build_regions renumbers region ids, and the label array must
    be renumbered with it or every zonal statistic reads the wrong region."""
    present = set(np.unique(ridx.label_array)) - {0}
    assert present == {r.id for r in ridx.regions}
    from scipy import ndimage as ndi
    areas = ndi.sum_labels(np.ones_like(ridx.label_array), ridx.label_array,
                           index=[r.id for r in ridx.regions])
    for r, a in zip(ridx.regions, areas):
        assert int(a) == r.area_px


# --- S1 --------------------------------------------------------------------

def test_s1_report_and_worklist(ridx):
    rep = s1_audit.run(ridx)
    assert rep.n_regions == len(ridx.regions)
    assert sum(rep.by_kind.values()) == len(rep.findings)
    text = s1_audit.worklist(rep, budget=3)
    assert "review worklist" in text
    if len(rep.findings) > 3:
        assert "omitted by budget" in text


def test_s1_ranks_by_review_value(ridx):
    rep = s1_audit.run(ridx)
    vals = [s1_audit.review_value(f) for f in rep.findings]
    assert vals == sorted(vals, reverse=True)


# --- S2 --------------------------------------------------------------------

def test_s2_shortlist_is_short_and_contains_incumbent(ridx):
    r = max(ridx.regions, key=lambda r: r.area_m2)
    cands = s2_adjudicate.candidates(ridx, r)
    assert r.class_id in cands
    assert len(cands) < 20, "adjudication is multiple-choice, not a re-run of perception"


def test_s2_verdict_vocabulary(ridx):
    for r in sorted(ridx.regions, key=lambda r: -r.area_m2)[:15]:
        adj = s2_adjudicate.adjudicate(ridx, r.id)
        assert (adj.verdict == "KEEP"
                or adj.verdict.startswith("SWITCH:")
                or adj.verdict.startswith("UNDECIDABLE-NEEDS-"))
        assert adj.rationale


def test_s2_abstains_on_pure_lithology_confusion():
    """Limestone vs Dolomite stony terrain differ only by lithology. Whatever
    else happens, S2 must not silently switch between them."""
    from segmap_digest.taxonomy import BY_ID
    a, b = cid("LimestoneStoneyTerrain"), cid("DolomiteStoneyTerrain")
    assert BY_ID[a].morphology == BY_ID[b].morphology
    assert BY_ID[a].lithology != BY_ID[b].lithology
    # the module's own distance metric must treat this as a near-tie
    from segmap_digest.taxonomy import class_distance
    assert class_distance(a, b) <= s2_adjudicate.CANDIDATE_MAX_DISTANCE


# --- S3 --------------------------------------------------------------------

def test_s3_hard_exclude_zeroes_dominated_chips(tile):
    cidx = build_chips(tile, size=128)
    policy = s3_triage.BUILTIN["vehicles"]
    for s in s3_triage.score_chips(cidx, policy):
        if s.excluded:
            assert s.score == 0.0 and s.reason


def test_s3_selection_accounting(tile):
    cidx = build_chips(tile, size=128)
    policy, sel = s3_triage.run(cidx, "vehicles", budget_frac=0.25)
    assert len(sel.selected) + len(sel.rejected) == len(cidx.chips)
    assert 0 <= sel.cost_reduction <= 1
    assert all(c in sel.rejected for c in sel.control), \
        "the control set must be drawn from REJECTED chips or it measures nothing"


def test_s3_control_set_is_never_silently_zero(tile):
    """Without a control set there is no signal that you over-pruned."""
    cidx = build_chips(tile, size=64)
    policy, sel = s3_triage.run(cidx, "vehicles", budget_frac=0.2)
    assert sel.rejected, "fixture should reject something at 20% budget"
    assert len(sel.control) >= 1


def test_s3_policy_json_roundtrip(tmp_path):
    p = tmp_path / "policy.json"
    s3_triage.BUILTIN["vehicles"].to_json(p)
    back = s3_triage.Policy.from_json(p)
    assert back.policy_id == "find-vehicles@v0"
    assert back.hard_exclude == s3_triage.BUILTIN["vehicles"].hard_exclude
    assert back.context_boosts[0].factor == 1.6


def test_s3_render_flags_prior_mass_not_recall(tile):
    cidx = build_chips(tile, size=128)
    policy, sel = s3_triage.run(cidx)
    assert "NOT recall" in s3_triage.render(sel, policy)


# --- S4 --------------------------------------------------------------------

def test_s4_products_are_bounded(tile):
    for name in s4_products.PRODUCTS:
        arr = s4_products.compute(tile, name)
        assert arr.shape == tile.shape
        assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0


def test_s4_roads_are_trafficable(tile):
    """Known-route check: an existing sealed road must not score NO-GO."""
    traf = s4_products.trafficability(tile, "wheeled")
    road = tile.labels == cid("PavedRoad")
    if road.any():
        assert traf[road].mean() > 0.5, "paved road scored as impassable"


def test_s4_water_is_impassable(tile):
    traf = s4_products.trafficability(tile, "wheeled")
    water = tile.labels == cid("Water")
    if water.any():
        assert traf[water].max() == 0.0


def test_s4_tracked_beats_wheeled_offroad(tile):
    w = s4_products.trafficability(tile, "wheeled")
    t = s4_products.trafficability(tile, "tracked")
    assert t.mean() >= w.mean()


# --- S5 --------------------------------------------------------------------

def test_s5_count_matches_region_index(tile, ridx):
    res = s5_query.query("count Water", tile, ridx)
    assert res.scalar == len([r for r in ridx.regions if r.class_name == "Water"])


def test_s5_area_matches_pixels(tile, ridx):
    res = s5_query.query("area House", tile, ridx)
    expected = sum(r.area_m2 for r in ridx.regions if r.class_name == "House")
    assert res.scalar == pytest.approx(expected)


def test_s5_filters_narrow_results(tile, ridx):
    wide = s5_query.query("find Batha", tile, ridx)
    narrow = s5_query.query("find Batha minarea 100", tile, ridx)
    assert len(narrow.rows) <= len(wide.rows)


def test_s5_rejects_bad_input(tile, ridx):
    for bad in ("", "find NotAClass", "sing House", "find House wat 3"):
        with pytest.raises((s5_query.QueryError, IndexError)):
            s5_query.query(bad, tile, ridx)


def test_s5_corridor_reports_its_threshold(tile, ridx):
    res = s5_query.query("corridor PavedRoad vehicle wheeled", tile, ridx)
    assert "threshold" in res.note, "a corridor answer must state its main assumption"


# --- S6 --------------------------------------------------------------------

def test_s6_null_test(tile):
    """A tile compared against itself must report zero change."""
    rep = s6_change.compare(tile, tile)
    assert rep.changed_frac == 0.0
    assert rep.events == []


def test_s6_lithology_flip_is_impossible_not_change():
    cat, why = s6_change.classify_transition(
        cid("LimestoneStoneyTerrain"), cid("DolomiteStoneyTerrain"))
    assert cat == "impossible"
    assert "label flip" in why


def test_s6_seasonal_is_phenology():
    a, b = cid("GreenGrassland"), cid("DryGrassland")
    assert s6_change.classify_transition(a, b)[0] == "phenology"
    assert s6_change.classify_transition(b, a)[0] == "phenology"


def test_s6_construction_and_infrastructure():
    assert s6_change.classify_transition(cid("Batha"), cid("House"))[0] == "construction"
    assert s6_change.classify_transition(cid("House"), cid("Batha"))[0] == "demolition"
    assert s6_change.classify_transition(
        cid("DirtRoad"), cid("PavedRoad"))[0] == "infrastructure"


def test_s6_second_date_produces_mixed_categories(tile):
    t2 = synth.second_date(tile, seed=3)
    rep = s6_change.compare(tile, t2)
    assert rep.changed_frac > 0
    assert len(rep.by_category) >= 2, "fixture should exercise more than one category"
    assert "REAL change" in s6_change.render(rep)


def test_s6_rejects_mismatched_grids(tile):
    other = synth.generate(size=128, seed=1)
    with pytest.raises(ValueError, match="grids differ"):
        s6_change.compare(tile, other)


# --- report ----------------------------------------------------------------

def test_report_builds_and_flags_synthetic_dem(tmp_path, tile):
    from segmap_digest import report

    index = report.build(tile, tmp_path / "out", source="fixture", synthetic=True)
    text = index.read_text()
    assert index.exists() and (tmp_path / "out" / "img" / "labels.png").exists()
    for anchor in ("map", "s1", "s2", "s3", "s4", "s5", "s6", "cost"):
        assert f'id="{anchor}"' in text
    assert "DEM is synthetic" in text, "must not present fixture slope as real"
    assert "no detector" in text, "S3 reframe must be visible in the output"


def test_report_flags_missing_dem(tmp_path):
    from segmap_digest import report

    flat = synth.generate(size=192, seed=8, with_dem=False)
    text = report.build(flat, tmp_path / "o2", synthetic=True).read_text()
    assert "No DEM supplied" in text


def test_tsv_table_escapes_html():
    from segmap_digest.report import tsv_table

    out = tsv_table("a\tb\n<script>\t&x")
    assert "<script>" not in out and "&lt;script&gt;" in out
