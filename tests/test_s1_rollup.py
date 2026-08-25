"""Regressions for the three things S1 got wrong on the first real export:

  a prior that asserted soil genesis and flooded the output with false positives
  a worklist that showed one finding twenty-five times
  a silence that read as a clean bill of health

Plus the scale-awareness of the enclosed-component checks, which were tuned
against a synthetic 0.3 m/px fixture and broke at 0.12 and 0.5 m/px.
"""

import numpy as np
import pytest

from segmap_digest import audit as audit_mod
from segmap_digest import synth, taxonomy
from segmap_digest.index import build_regions
from segmap_digest.loader import LabelRaster
from segmap_digest.solutions import s1_audit


@pytest.fixture(scope="module")
def ridx():
    return build_regions(synth.generate(size=384, seed=5))


# --- Task 1: the soil-genesis priors are gone and must stay gone -----------

SOIL_CLASSES = ("TerraRosa", "Rendzina", "ClayeySoil", "ClayeyDeepSoil")


def test_no_soil_class_asserts_a_parent_rock_neighbour():
    """The class owner's ruling: these are SOIL TYPES, not genetic units. A
    prior requiring a soil class to border a particular lithology is a claim
    about how the soil formed, which the label never made.

    HydromorpicSoil is deliberately excluded: its `others` are Water and
    ClayeyDeepSoil, which is a topographic claim, not a parent-rock one."""
    rock_names = {c.name for c in taxonomy.CLASSES if c.lithology}
    for p in taxonomy.PRIORS:
        if p.subject in SOIL_CLASSES:
            assert not (set(p.others) & rock_names), (
                f"{p.subject} {p.kind} {p.others} is a parent-rock claim; see "
                f"taxonomy.RETIRED_PRIORS before adding it back"
            )


def test_soil_definitions_make_no_parent_rock_claim():
    """The definitions are what an LLM reasons over. Deleting the check but
    leaving the genetic story in the text keeps the bad inference alive."""
    for name in ("TerraRosa", "Rendzina"):
        text = taxonomy.BY_NAME[name].definition.lower()
        for banned in ("forms on", "formed on", "parent material",
                       "decalcification", "misclassification signal",
                       "implausible"):
            assert banned not in text, f"{name} definition still asserts genesis"
        assert "soil type" in text


def test_retired_priors_are_recorded_not_merely_deleted():
    subjects = {s for s, _, _ in taxonomy.RETIRED_PRIORS}
    assert {"TerraRosa", "Rendzina"} <= subjects
    assert all(why for _, _, why in taxonomy.RETIRED_PRIORS)


# --- Task 2a: root-cause roll-up ------------------------------------------

def test_every_finding_carries_a_root_cause(ridx):
    for f in audit_mod.audit(ridx):
        assert f.cause and f.cause_text


def test_worklist_rows_are_distinct_causes(ridx):
    """The failure being regressed: 25 rows, one finding, repeated."""
    rep = s1_audit.run(ridx)
    rows = [ln.split("\t") for ln in s1_audit.worklist(rep, budget=25).splitlines()
            if ln and not ln.startswith("#")][1:]
    keys = [r[1] for r in rows]
    assert len(keys) == len(set(keys)), "worklist repeated a root cause"
    messages = [r[-1] for r in rows]
    assert len(messages) == len(set(messages)), "worklist repeated a message"


def test_roll_up_conserves_findings(ridx):
    rep = s1_audit.run(ridx)
    assert sum(c.n_findings for c in rep.causes) == len(rep.findings)
    assert len(rep.causes) <= len(rep.findings)


def test_detail_is_addressable_from_a_cause(ridx):
    rep = s1_audit.run(ridx)
    if not rep.causes:
        pytest.skip("fixture produced no findings")
    key = rep.causes[0].key
    text = s1_audit.detail(rep, key, budget=5)
    assert key in text
    assert "no such root cause" in s1_audit.detail(rep, "nope/nope")


def test_worklist_states_what_the_budget_dropped(ridx):
    rep = s1_audit.run(ridx)
    if len(rep.causes) < 2:
        pytest.skip("fixture produced too few causes")
    text = s1_audit.worklist(rep, budget=1)
    assert "omitted by budget" in text
    assert "findings over" in text, "a cap must say what it dropped"


# --- Task 2b: severity discriminates --------------------------------------

def test_severity_is_not_a_per_kind_constant(ridx):
    """A fixed constant per check is what let one prior sweep the budget."""
    findings = audit_mod.audit(ridx)
    by_kind: dict[str, set] = {}
    for f in findings:
        by_kind.setdefault(f.kind, set()).add(f.severity)
    graded = [k for k, v in by_kind.items() if len(v) > 1]
    assert graded, "no check produced more than one severity value"
    for kind, vals in by_kind.items():
        floor, span = audit_mod.SEVERITY_BAND[kind]
        assert min(vals) >= floor - 1e-9
        assert max(vals) <= floor + span + 1e-9


def test_severity_tracks_the_evidence():
    """Two findings of one kind order by how badly the evidence is violated."""
    g = audit_mod._grade
    assert g("isolated-speck", 0.9) > g("isolated-speck", 0.1)
    assert g("missing-expected-context", 1.0, 1.0) > \
           g("missing-expected-context", 1.0, 0.0)


# --- Task 2c: the enclosed-component windows are scale-aware ---------------

def _two_tile_raster(gsd: float) -> LabelRaster:
    """A 40-cell DryGrassland square inside Rendzina. 40 cells is inside the
    speck window at any GSD; its ground area is not."""
    lab = np.full((120, 120), taxonomy.cid("Rendzina"), dtype=np.uint8)
    lab[50:56, 50:57] = taxonomy.cid("DryGrassland")     # 42 cells
    return LabelRaster(lab, gsd)


@pytest.mark.parametrize("gsd", [0.122, 0.3, 0.496])
def test_speck_detection_is_invariant_to_gsd(gsd):
    """The old window was 25-400 m2: 1,667-26,675 cells at 0.122 m/px and
    102-1,625 at 0.496, so the same object was a speck at one GSD and a
    building at another."""
    ridx = build_regions(_two_tile_raster(gsd), min_area_px=12)
    kinds = {f.kind for f in audit_mod.audit(ridx)}
    assert "isolated-speck" in kinds, f"speck check went silent at {gsd} m/px"


def test_speck_window_is_in_cells_not_ground_area():
    assert audit_mod.SPECK_MIN_PX < audit_mod.SPECK_MAX_PX
    # A component above the ceiling is an object the segmenter drew, whatever
    # its area in m2.
    lab = np.full((200, 200), taxonomy.cid("Rendzina"), dtype=np.uint8)
    lab[60:100, 60:100] = taxonomy.cid("DryGrassland")   # 1600 cells
    ridx = build_regions(LabelRaster(lab, 0.1), min_area_px=12)
    assert "isolated-speck" not in {f.kind for f in audit_mod.audit(ridx)}


def test_speck_check_is_not_gated_by_min_area_m2():
    """It is the check ABOUT components below that gate; gating it inverts it,
    which is how 300 m2 buildings became specks on aza."""
    ridx = build_regions(_two_tile_raster(0.122), min_area_px=12)
    kinds = {f.kind for f in audit_mod.audit(ridx, min_area_m2=1e6)}
    assert "isolated-speck" in kinds


# --- Task 3: coverage, and silence that says it is silence -----------------

def test_coverage_reports_checked_classes_and_area(ridx):
    cov = s1_audit.run(ridx).coverage
    assert cov is not None
    assert 0 <= cov.n_checked <= len(cov.classes_present)
    assert 0.0 <= cov.area_fraction <= 1.0
    assert abs(sum(cov.class_area.values()) - cov.total_area_m2) < 1.0


def test_no_dem_is_reported_as_abstention_not_as_a_pass():
    tile = synth.generate(size=256, seed=3)
    tile.dem = None
    rep = s1_audit.run(build_regions(tile))
    text = s1_audit.render(rep, budget=5)
    assert "ABSTAINED" in text and "no DEM" in text
    assert "slope-violation" not in rep.by_kind


def test_render_says_unchecked_is_not_clean(ridx):
    text = s1_audit.render(s1_audit.run(ridx), budget=5)
    assert "UNEXAMINED, NOT CLEAN" in text
    assert "coverage" in text.split("\n")[1]
    # The unchecked classes must be named, with their area share.
    assert "NONE -- UNEXAMINED" in text


def test_zero_findings_still_reports_coverage():
    """The aza failure mode: a short list read as a clean bill of health."""
    lab = np.full((64, 64), taxonomy.cid("Batha"), dtype=np.uint8)
    rep = s1_audit.run(build_regions(LabelRaster(lab, 0.5)))
    assert rep.findings == []
    text = s1_audit.render(rep, budget=5)
    assert "UNEXAMINED, NOT CLEAN" in text
    assert "none -- see the coverage block" in text
