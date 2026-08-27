"""Regression tests for the tile-level change seam.

Every one of these was written against a defect an adversarial review found in
the first cut, and each pins the behaviour that defect broke:

  the top-N class cap was scored as if unlisted classes were absent, so a
    four-pixel rank swap read as a 20%-of-tile transition (204x overstated);
  the volatility prior defaulted to 0.0, the one value `s4_products` names as
    unsafe, so a tile with no product silently became "any difference is real";
  the structural rule was tested AFTER the noise floor and so could not fire
    below it, and its share halved a one-sided move, making the documented 5%
    bar an effective 10%;
  `structural_classes_from_taxonomy` read a table name that does not exist, so
    its silent fallback fired every time and called `Car` structural.

The null test at the bottom deliberately does NOT claim to validate the
detector: diffing an index against itself is arithmetic, and it can only ever
return all-quiet. It pins that the arithmetic is right, which is worth having
and is not evidence about real change.
"""

import pytest

from segmap_digest import tilechange as TC


def tile(i, classes, other=0.0, volatility=0.5, r=0, c=0):
    return {"i": i, "r": r, "c": c, "classes": classes, "classes_other": other,
            "s4": {"change_volatility": volatility}}


def grid(tiles, tile_px=128):
    return {"tile_px": tile_px, "cols": len(tiles), "rows": 1,
            "shape": [tile_px, tile_px * len(tiles)], "n_tiles": len(tiles),
            "tiles": tiles}


def one(a, b, **kw):
    return TC.diff_tile_indices(grid([a]), grid([b]), registered=True, **kw)["tiles"][0]


# --- truncation ------------------------------------------------------------

def test_a_rank_swap_below_the_cap_is_not_a_transition():
    """Five near-equal classes, four pixels move, the rank-4/5 tie flips.

    Scoring the class that fell off the list as zero reported shift 0.200 for a
    true 0.00098 -- and a `worth-a-look` verdict for a transition that did not
    happen. The residual mass is what makes that bound honest.
    """
    a = tile(0, [["A", .2506], ["B", .2506], ["C", .2506], ["D", .2506]], other=.1976)
    b = tile(0, [["A", .2506], ["B", .2506], ["C", .2506], ["E", .2494]], other=.1988)
    got = one(a, b)
    assert got["shift"] < 0.08, f"shift {got['shift']} -- truncation is inventing change"
    assert got["verdict"] == "quiet"


def test_total_variation_is_a_lower_bound_not_an_estimate():
    """A class missing from one side may hold up to that side's residual, so it
    can only be charged for the part the residual cannot explain."""
    a = {"X": 0.5}
    b = {}
    assert TC._total_variation(a, b, ra=0.5, rb=0.5) == pytest.approx(0.0)
    assert TC._total_variation(a, b, ra=0.5, rb=0.0) == pytest.approx(0.25)


# --- the volatility prior --------------------------------------------------

def test_a_missing_volatility_product_is_recorded_as_unknown():
    """`s4_products` says 0.0 means 'any difference here is real', which is the
    least safe thing to say about ground nobody has data for."""
    a = {"i": 0, "r": 0, "c": 0, "classes": [["House", 1.0]], "s4": {}}
    b = {"i": 0, "r": 0, "c": 0, "classes": [["Clutter", 1.0]], "s4": {}}
    got = TC.diff_tile_indices(grid([a]), grid([b]), registered=True)["tiles"][0]
    assert got["volatility_known"] is False


def test_phenology_is_discounted_and_masonry_is_not():
    """The whole point: a dry-grassland tile flipping green is the same ground
    in two seasons; a house block turning to rubble is not."""
    grass = one(tile(0, [["DryGrassland", 1.0]], volatility=0.9),
                tile(0, [["GreenGrassland", 1.0]], volatility=0.9))
    house = one(tile(0, [["House", 1.0]], volatility=0.05),
                tile(0, [["Clutter", 1.0]], volatility=0.2),
                structural_classes=("House",))
    assert grass["shift"] == pytest.approx(1.0)
    assert grass["verdict"] == "expected", "phenology must not surface as change"
    assert house["verdict"] == "structural"
    assert house["excess"] > grass["excess"]


# --- the structural rule ---------------------------------------------------

def test_new_construction_below_the_noise_floor_still_surfaces():
    """6% of a tile becoming House was reported `quiet` twice over: the
    structural test ran after the noise floor, and halving a one-sided move made
    the documented 5% bar an effective 10%."""
    got = one(tile(0, [["DryGrassland", 1.0]], volatility=0.9),
              tile(0, [["DryGrassland", .94], ["House", .06]], volatility=0.85),
              structural_classes=("House",))
    assert got["structural"] == pytest.approx(0.06, abs=1e-3)
    assert got["verdict"] == "structural"


def test_structural_classes_are_derived_from_the_volatility_table():
    """This read a table name that does not exist, so the fallback fired every
    time -- calling `Car` structural, whose volatility is 0.8 precisely because
    a parked car moving is not a change of the ground."""
    got = TC.structural_classes_from_taxonomy()
    assert "House" in got and "PavedRoad" in got
    assert "Car" not in got
    assert "DryGrassland" not in got
    assert any("Limestone" in n for n in got), "rock classes are structural too"


# --- refusals --------------------------------------------------------------

def test_mismatched_grids_raise_rather_than_compare():
    a, b = grid([tile(0, [["A", 1.0]])]), grid([tile(0, [["A", 1.0]])], tile_px=256)
    with pytest.raises(ValueError, match="not comparable"):
        TC.diff_tile_indices(a, b)


def test_an_unregistered_pair_is_refused_not_compared():
    a, b = grid([tile(0, [["A", 1.0]])]), grid([tile(0, [["B", 1.0]])])
    got = TC.diff_tile_indices(a, b, registered=False)
    assert got["tiles"] == [] and "co-registered" in got["note"]


def test_diffing_an_index_against_itself_is_all_quiet():
    """Arithmetic, not evidence. This CANNOT return anything else for any input,
    which is exactly why it is not validation of the detector -- the real null
    test re-runs the segmenter on one date and takes the 95th percentile, and
    has not been done."""
    g = grid([tile(0, [["House", .6], ["Batha", .4]], volatility=0.1),
              tile(1, [["DryGrassland", 1.0]], volatility=0.9, c=1)])
    got = TC.diff_tile_indices(g, g, registered=True)
    assert got["counts"] == {"quiet": 2}


def test_render_states_when_co_registration_was_never_checked():
    g = grid([tile(0, [["A", 1.0]])])
    txt = TC.render(TC.diff_tile_indices(g, g))
    assert "co-registration was not checked" in txt
