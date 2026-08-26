"""Regression tests for the S6 change-reasoning fixes.

Each test names the finding it pins and was written RED against the pre-fix
s6_change.py. Finding numbers refer to the S6 review:

  1. compare() ignored LabelRaster.valid -- nodata garbage counted as change.
  2. multi-stage jumps along DEGRADATION_SERIES were typed as calm succession.
  3. the fallthrough why said "crosses superclasses" for same-superclass pairs.
  4. compare() rebuilt and ndi.label'ed a full-raster mask once per transition
     pair -- O(n_pairs x n_pixels), the blowup pattern of HANDOFF.md section 8.5.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from segmap_digest import synth
from segmap_digest.loader import LabelRaster
from segmap_digest.solutions import s6_change
from segmap_digest.taxonomy import N_CLASSES, SUPERCLASS_OF, cid, class_distance


def _raster(labels, valid=None, gsd=1.0):
    """gsd=1.0 so pixel counts are areas in m2 -- MIN_EVENT_AREA_M2=20 means
    a block needs >=20 px to become an event."""
    labels = np.asarray(labels, dtype=np.uint8)
    return LabelRaster(labels=labels, gsd=gsd,
                       valid=None if valid is None else np.asarray(valid, dtype=bool))


# --- finding 1: the nodata mask --------------------------------------------

def test_s6_nodata_is_coverage_change_not_class_change():
    """A pixel observed on exactly one date cannot have CHANGED CLASS. The
    garbage value under the other date's nodata must not surface as an event,
    and changed_frac's denominator is the co-valid count, not the extent."""
    h = w = 32
    t1 = _raster(np.full((h, w), cid("Batha")))          # valid=None: all data

    labels2 = np.full((h, w), cid("Batha"))
    valid2 = np.ones((h, w), dtype=bool)
    # 16x16 corner: t2 has no data there, and the array holds a garbage value
    # (Water) underneath -- exactly what a cropped GeoTIFF does (loader.py).
    labels2[:16, :16] = cid("Water")
    valid2[:16, :16] = False
    # One real change inside the co-valid area: 4x6 = 24 px Batha -> House.
    labels2[20:24, 20:26] = cid("House")
    t2 = _raster(labels2, valid=valid2)

    rep = s6_change.compare(t1, t2)

    assert [(e.from_class, e.to_class) for e in rep.events] == [("Batha", "House")], \
        "the nodata corner must not appear as a Batha->Water event"
    n_both = h * w - 16 * 16
    assert rep.changed_frac == pytest.approx(24 / n_both), \
        "changed_frac must be over pixels valid in BOTH dates"
    # The lost corner is a change in coverage, reported separately: 256 of the
    # 1024 observed-at-least-once pixels were seen on exactly one date.
    assert rep.coverage_changed_frac == pytest.approx(256 / 1024)
    assert "coverage" in s6_change.render(rep)


def test_s6_no_masks_means_no_coverage_change():
    """valid=None on both dates (the synthetic fixture) reports zero coverage
    change and behaves exactly as before."""
    tile = synth.generate(size=192, seed=5)
    rep = s6_change.compare(tile, tile)
    assert rep.coverage_changed_frac == 0.0
    assert rep.changed_frac == 0.0
    assert rep.events == []


def test_s6_pixels_valid_in_neither_date_count_nowhere():
    """Shared nodata is not class change and not coverage change either."""
    nod = np.zeros((8, 8), dtype=bool)
    nod[:4, :] = True
    t1 = _raster(np.full((8, 8), cid("Batha")), valid=nod)
    labels2 = np.full((8, 8), cid("Water"))              # garbage under nodata
    labels2[:4, :] = cid("Batha")
    t2 = _raster(labels2, valid=nod)
    rep = s6_change.compare(t1, t2)
    assert rep.changed_frac == 0.0
    assert rep.coverage_changed_frac == 0.0


# --- finding 2: multi-stage jumps along the degradation series --------------

def test_s6_multistage_drop_is_reportable_not_succession():
    """Maquis->DryGrassland is a 3-stage drop in one epoch: fire, clearance,
    or a label error -- the most reportable event on this landscape, not
    gradual succession."""
    cat, why = s6_change.classify_transition(cid("Maquis"), cid("DryGrassland"))
    assert cat == "real-change"
    assert "fire" in why and "succession" in why
    # 2 stages is already too fast for one epoch.
    cat2, _ = s6_change.classify_transition(cid("Maquis"), cid("Batha"))
    assert cat2 == "real-change"


def test_s6_multistage_regrowth_is_also_suspicious():
    """Regrowth takes years per stage; a >=2-stage jump up in one epoch is
    planting or a label error, not succession."""
    cat, why = s6_change.classify_transition(cid("DryGrassland"), cid("Maquis"))
    assert cat == "real-change"
    assert "regrowth" in why


def test_s6_single_step_stays_succession_and_roads_stay_infrastructure():
    """The reclassification must not leak: 1-step moves are still succession
    (either direction), and any road-series jump is still infrastructure --
    paving IS instantaneous."""
    assert s6_change.classify_transition(
        cid("Garigue"), cid("Batha"))[0] == "succession"
    assert s6_change.classify_transition(
        cid("Batha"), cid("Garigue"))[0] == "succession"
    assert s6_change.classify_transition(
        cid("DirtRoadB"), cid("PavedRoad"))[0] == "infrastructure"


# --- finding 3: the fallthrough why must be truthful -------------------------

def test_s6_fallthrough_why_names_the_real_relation():
    """GreenGrassland->Batha sits exactly on the 0.4 same-superclass distance:
    both are vegetation, so 'crosses superclasses' is simply false. It stays
    real-change (herbaceous -> woody is a formation shift, and the exactly-0.4
    boundary deliberately fails the strict < 0.4 noise gate -- see the comment
    in classify_transition), but the why must say what is actually true."""
    a, b = cid("GreenGrassland"), cid("Batha")
    assert SUPERCLASS_OF[a] == SUPERCLASS_OF[b] == "vegetation"
    assert class_distance(a, b) == pytest.approx(0.4)
    cat, why = s6_change.classify_transition(a, b)
    assert cat == "real-change"
    assert "crosses superclasses" not in why
    assert "vegetation" in why


def test_s6_true_cross_superclass_message_survives():
    """Soil -> rock genuinely crosses superclasses and should still say so."""
    a, b = cid("TerraRosa"), cid("LimestoneBoulder")
    assert SUPERCLASS_OF[a] != SUPERCLASS_OF[b]
    cat, why = s6_change.classify_transition(a, b)
    assert cat == "real-change"
    assert "crosses superclasses" in why


# --- finding 4: per-pair full-raster labelling -------------------------------

def _naive_ncomp_by_pair(t1, t2):
    """The pre-fix reference: one full-raster mask + ndi.label per (from,to)
    pair, restricted to pixels valid in both dates. Kept here, in the test, as
    the ground truth the fast path must match exactly."""
    a, b = t1.labels, t2.labels
    changed = a != b
    if t1.valid is not None:
        changed &= t1.valid
    if t2.valid is not None:
        changed &= t2.valid
    out = {}
    pairs = a[changed].astype(np.int32) * N_CLASSES + b[changed]
    for key in np.unique(pairs):
        ca, cb = divmod(int(key), N_CLASSES)
        mask = changed & (a == ca) & (b == cb)
        _, ncomp = ndi.label(mask, structure=np.ones((3, 3)))
        out[(ca, cb)] = int(ncomp)
    return out


def test_s6_component_counts_match_naive_on_adversarial_layout():
    """The layout is built to break a label-the-changed-mask-once shortcut:
    three touching transition blocks form ONE changed component holding TWO
    (from,to) pairs, and the same pair appears twice inside it, split by the
    other pair. Per-event semantics: components of pixels sharing the same
    (from,to) pair, 8-connectivity."""
    h = w = 40
    t1 = _raster(np.full((h, w), cid("Batha")))
    labels2 = np.full((h, w), cid("Batha"))
    labels2[0:4, 0:20] = cid("House")     # pair P1, component 1
    labels2[4:8, 0:20] = cid("Water")     # pair P2, touches both P1 blocks
    labels2[8:12, 0:20] = cid("House")    # pair P1 again -- separate component
    labels2[20:26, 20:28] = cid("House")  # pair P1, its own changed component
    t2 = _raster(labels2)

    naive = _naive_ncomp_by_pair(t1, t2)
    assert naive[(cid("Batha"), cid("House"))] == 3
    assert naive[(cid("Batha"), cid("Water"))] == 1

    rep = s6_change.compare(t1, t2)
    got = {(cid(e.from_class), cid(e.to_class)): e.n_components for e in rep.events}
    assert got == naive


def test_s6_component_counts_match_naive_on_fixture():
    """Same equivalence on the synthetic two-date fixture the CLI renders."""
    tile = synth.generate(size=512, seed=7)
    t2 = synth.second_date(tile, seed=99)
    naive = _naive_ncomp_by_pair(tile, t2)
    rep = s6_change.compare(tile, t2)
    assert len(rep.events) >= 3, "fixture should exercise several pairs"
    for e in rep.events:
        assert e.n_components == naive[(cid(e.from_class), cid(e.to_class))]


def test_s6_compare_does_not_label_the_full_raster_per_pair(monkeypatch):
    """compare() may label a full-raster mask at most ONCE, not once per
    transition pair -- the O(n_pairs x n_pixels) pattern of HANDOFF.md 8.5.
    (Sub-raster relabelling of a mixed component's bounding box is fine.)"""
    tile = synth.generate(size=512, seed=7)
    t2 = synth.second_date(tile, seed=99)

    full_calls = {"n": 0}
    real_label = ndi.label

    def counting_label(inp, *args, **kwargs):
        if getattr(inp, "shape", None) == tile.shape:
            full_calls["n"] += 1
        return real_label(inp, *args, **kwargs)

    monkeypatch.setattr(s6_change.ndi, "label", counting_label)
    rep = s6_change.compare(tile, t2)
    assert len(rep.events) >= 3, "fixture should exercise several pairs"
    assert full_calls["n"] <= 1, \
        f"labelled a full-raster mask {full_calls['n']} times for " \
        f"{len(rep.events)} pairs -- per-pair full-raster work is the blowup"
