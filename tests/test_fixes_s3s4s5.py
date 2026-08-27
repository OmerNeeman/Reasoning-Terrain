"""Regression tests for verified S3/S4/S5 findings.

Each test here was written RED against the bug it pins down:

  S3  select() rounded the control set to zero on small chip pools, while
      render() kept calling it "the only unbiased recall signal".
  S4  concealment() ran a Python callback per pixel (generic_filter + np.std);
      the vectorised local std must be numerically equivalent, edges included.
  S4  drainage() normalised elevation over the whole raster, so a valley at
      high absolute elevation read as non-accumulating. THAT PRODUCT WAS
      RETIRED (the owner removed drainage and fire_fuel as irrelevant to this
      project), so its test went with it -- but the bug it pinned was about
      LOCAL vs GLOBAL normalisation, and `built_fabric` inherited exactly that
      shape in its smoothed density term. The replacement tests below pin the
      same lesson on the product that now carries it.
  S5  _corridor()'s headline scalar disagreed with its own table (dropped
      pockets stayed in the scalar), and 8-connected labeling let two areas
      count as mutually reachable through a single diagonal pixel no vehicle
      fits through.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from segmap_digest import synth
from segmap_digest.index import build_chips, build_regions
from segmap_digest.loader import LabelRaster
from segmap_digest.solutions import s3_triage, s4_products, s5_query
from segmap_digest.taxonomy import cid


@pytest.fixture(scope="module")
def tile():
    return synth.generate(size=384, seed=5)


@pytest.fixture(scope="module")
def ridx(tile):
    return build_regions(tile)


# --- S3: control set must not round to zero on small pools ------------------

def test_s3_small_pool_still_gets_a_control_chip(tile):
    """16 chips at 20% budget rejects 13; 3% of 13 rounds to 0 -- and a zero
    control set silently deletes the only unbiased recall signal."""
    cidx = build_chips(tile, size=96)          # 384/96 -> 4x4 = 16 chips
    assert len(cidx.chips) == 16
    policy, sel = s3_triage.run(cidx, "vehicles", budget_frac=0.2)
    assert sel.rejected, "fixture must reject something at 20% budget"
    assert len(sel.control) >= 1, \
        "a non-empty rejected set with control_frac > 0 must yield >= 1 control chip"
    # A 1-chip control set is evidence, but barely: render must say so instead
    # of presenting it as a working recall estimate.
    text = s3_triage.render(sel, policy)
    assert len(sel.control) < s3_triage.CONTROL_MIN_INFORMATIVE
    assert "wide confidence" in text


def test_s3_control_frac_zero_is_respected(tile):
    """max(1, ...) must not manufacture a control set the caller turned off."""
    cidx = build_chips(tile, size=96)
    sel = s3_triage.select(
        s3_triage.score_chips(cidx, s3_triage.BUILTIN["vehicles"]),
        budget_frac=0.2, control_frac=0.0)
    assert sel.control == []


# --- S4: concealment roughness must be vectorised AND equivalent ------------

def test_s4_local_std_matches_generic_filter():
    """The uniform_filter formulation must reproduce the naive per-pixel
    np.std, including at the edges: both use 'nearest' border replication,
    so interior and edge semantics are identical, not merely close."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0.0, 45.0, size=(37, 41))          # slope-like magnitudes
    naive = ndi.generic_filter(a, np.std, size=5, mode="nearest")
    fast = s4_products._local_std(a, s4_products.ROUGHNESS_WINDOW_PX)
    np.testing.assert_allclose(fast, naive, atol=1e-5)


def test_s4_local_std_is_nonnegative_on_flat_input():
    """E[x^2] - E[x]^2 rounds a hair below zero on constant input; sqrt of
    that is NaN unless clamped."""
    flat = np.full((16, 16), 7.3)
    out = s4_products._local_std(flat, 5)
    assert np.isfinite(out).all() and (out >= 0).all()


# --- S4: built_fabric density must be local, not AOI-global -----------------

def test_s4_built_fabric_does_not_call_an_isolated_shed_a_city():
    """One building in open desert must not read as dense urban fabric.

    Same lesson the retired drainage test pinned: the density term has to be
    measured in a local window. A whole-raster normalisation makes the only
    built thing on the map its own maximum, and the score saturates on a shed.
    """
    h = w = 200
    labels = np.full((h, w), cid("DryGrassland"), dtype=np.uint8)
    labels[100:106, 100:106] = cid("House")            # ~6x6 px, one structure
    lone = s4_products.built_fabric(LabelRaster(labels=labels, gsd=1.0))

    dense = np.full((h, w), cid("House"), dtype=np.uint8)
    dense[::7, :] = cid("PavedRoad")                   # a street grid
    block = s4_products.built_fabric(LabelRaster(labels=dense, gsd=1.0))

    assert lone[103, 103] < block[103, 103], (
        f"an isolated shed scored {lone[103, 103]:.2f} against {block[103, 103]:.2f} "
        f"for dense urban fabric -- the density term is not local")
    assert lone[10, 10] < 0.2, "open grassland far from the shed should be near 0"


def test_s4_change_volatility_orders_phenology_above_masonry():
    """The ordering the change-detection work depends on.

    `change_volatility` exists so a detector can subtract what was always going
    to differ. If seasonal vegetation did not outrank masonry here, the whole
    subtraction would push in the wrong direction and a destroyed building would
    be forgiven while a field turning green got flagged.
    """
    def vol(name):
        labels = np.full((32, 32), cid(name), dtype=np.uint8)
        return float(s4_products.change_volatility(
            LabelRaster(labels=labels, gsd=1.0)).mean())

    assert vol("DryGrassland") > vol("Maquis") > vol("House")
    assert vol("Shadow") > vol("PavedRoad"), (
        "Shadow is an illumination artifact and moves with sun angle between "
        "two dates for reasons that are not change at all")
    assert vol("House") < 0.15, "masonry changing means something happened"


def test_s4_concealment_refuses_an_unknown_target():
    """The target is the question the old constant silently answered."""
    labels = np.full((32, 32), cid("Maquis"), dtype=np.uint8)
    r = LabelRaster(labels=labels, gsd=1.0)
    person = float(s4_products.concealment(r, target="person").mean())
    structure = float(s4_products.concealment(r, target="structure").mean())
    assert person > structure, "the same canopy hides a person, not a building"
    with pytest.raises(ValueError, match="person|target"):
        s4_products.concealment(r, target="banana")


# --- S5: corridor scalar must agree with its own table ----------------------

def _raster(labels: np.ndarray, gsd: float) -> LabelRaster:
    return LabelRaster(labels=labels.astype(np.uint8), gsd=gsd)


def test_s5_corridor_scalar_excludes_dropped_pockets():
    """A seed-touching pocket below CORRIDOR_MIN_AREA_M2 is dropped from the
    table as noise -- the headline scalar must drop it too, and the note must
    state the excluded total instead of folding it in silently."""
    gsd = 0.5
    labels = np.full((200, 200), cid("Water"), dtype=np.uint8)   # traffic 0
    labels[20:70, 20:70] = cid("PavedRoad")            # 2500 px = 625 m2, kept
    labels[150:153, 150:153] = cid("PavedRoad")        # 9 px = 2.25 m2, a pocket
    r = _raster(labels, gsd)
    res = s5_query.query("corridor PavedRoad vehicle foot", r, build_regions(r))

    table_total = sum(float(row[1]) for row in res.rows)
    assert res.scalar == pytest.approx(table_total), \
        "headline scalar and evidence table disagree"
    assert res.scalar == pytest.approx(625.0)
    assert "totalling" in res.note and "excluded" in res.note


def test_s5_corridor_diagonal_pixel_is_not_a_road():
    """Two 625 m2 areas joined only by a 1-px diagonal chain: mutually
    reachable on foot (pure percolation), but a wheeled vehicle (2.5 m wide)
    does not pass a half-metre gap -- the opening must split them."""
    gsd = 0.5
    labels = np.full((200, 200), cid("Water"), dtype=np.uint8)
    labels[20:70, 20:70] = cid("PavedRoad")
    labels[72:122, 72:122] = cid("PavedRoad")
    labels[70, 70] = cid("PavedRoad")                  # the diagonal chain:
    labels[71, 71] = cid("PavedRoad")                  # (69,69)-(70,70)-(71,71)-(72,72)
    r = _raster(labels, gsd)
    ridx = build_regions(r)

    on_foot = s5_query.query("corridor PavedRoad vehicle foot", r, ridx)
    assert len(on_foot.rows) == 1, "foot applies no width gate; percolation joins them"

    wheeled = s5_query.query("corridor PavedRoad vehicle wheeled", r, ridx)
    assert len(wheeled.rows) == 2, \
        "a 1-px diagonal must not connect components for a 2.5 m wide vehicle"
    assert "width" in wheeled.note


def test_s5_fixture_corridor_still_runs(tile, ridx):
    """The width opening must not erase the fixture's road network: the stock
    corridor query still finds reachable ground and states the width applied."""
    res = s5_query.query("corridor PavedRoad", tile, ridx)
    assert res.scalar is not None and res.scalar > 0
    assert "width" in res.note
