"""Per-tile aggregation, the map it renders, and the traversability seam.

All three shipped with no tests at all. These pin the things an adversarial
review found broken or unguarded -- most importantly the script-tag breakout,
because the data being embedded is OSM `name=*` text, which is to say arbitrary
strings authored by whoever last edited that street.
"""

import json

import numpy as np
import pytest

from segmap_digest import synth, tilemap, tilemap_ui, traversability
from segmap_digest.index import build_chips, build_regions
from segmap_digest.solutions import s1_audit


@pytest.fixture(scope="module")
def built():
    r = synth.generate(size=384, seed=5)
    ridx = build_regions(r)
    cidx = build_chips(r, size=128)
    rep = s1_audit.run(ridx)
    ti = tilemap.build_tile_index(r, ridx, cidx, None, rep, [], {}, tile_px=128)
    return r, ti


# --- the aggregation -------------------------------------------------------

def test_the_index_is_json_serialisable(built):
    """It is embedded in a web page; a numpy scalar anywhere raises at render."""
    _r, ti = built
    json.dumps(ti)


def test_the_grid_covers_the_raster(built):
    r, ti = built
    assert ti["n_tiles"] == ti["cols"] * ti["rows"]
    assert ti["cols"] * ti["tile_px"] >= r.shape[1] - ti["tile_px"]


def test_the_top_n_cap_records_what_it_dropped(built):
    """The repo enforces 'a cap must never read as this is everything'. Without
    the residual, `tilechange` scored a class that merely fell off the list as
    absent and invented transitions."""
    _r, ti = built
    for t in ti["tiles"]:
        listed = sum(c[1] for c in t["classes"])
        assert "classes_other" in t
        assert t["classes_other"] == pytest.approx(max(0.0, 1 - listed), abs=0.02)
        if t["n_classes"] > len(t["classes"]):
            assert t["classes_other"] > 0, "dropped classes must leave residual mass"


def test_product_keys_track_the_solution_not_a_private_copy():
    """A name added to `s4_products.PRODUCTS` and not here is a product that
    silently never reaches a tile -- which is how `drainage` and `fire_fuel`
    outlived their own deletion in six files."""
    from segmap_digest.solutions.s4_products import PRODUCTS

    assert set(PRODUCTS) <= set(tilemap.PRODUCT_KEYS)


def test_the_ui_metric_list_matches_what_a_tile_carries():
    """When these drift, `val()` returns 0 for every tile and the map paints a
    uniform heat surface under a 0-1 legend, silently."""
    keys = {m[0] for m in tilemap_ui.METRICS} - {"none", "s1", "s3"}
    assert keys <= set(tilemap.PRODUCT_KEYS), sorted(keys - set(tilemap.PRODUCT_KEYS))


# --- the map ---------------------------------------------------------------

def test_a_street_name_cannot_break_out_of_the_script_tag():
    """`json.dumps` does not escape `<`, and block labels are OSM `name=*` tags.
    Anyone who can edit the map could run JS in a report that is then written to
    out/playground/ and re-served."""
    ti = {"tile_px": 8, "cols": 1, "rows": 1, "gsd": 1.0, "shape": [8, 8],
          "n_tiles": 1, "tiles": [{
              "i": 0, "r": 0, "c": 0, "bbox": [0, 0, 8, 8], "classes": [],
              "classes_other": 0.0, "n_classes": 0,
              "s1": {"n": 0, "max_sev": 0.0, "causes": [], "rids": []},
              "s2": {"verdicts": [], "examples": []},
              "s3": {"score": 0.0, "selected": False}, "s4": {},
              "osm": {"block": 1, "n_junctions": 0, "road_frac": 0.0,
                      "building_frac": 0.0,
                      "block_label": "</script><img src=x onerror=alert(1)>"}}]}
    html = tilemap_ui.render_map_section(ti, "data:image/png;base64,iVBORw0KGgo=")
    assert "</script><img" not in html
    assert "u003c" in html


def test_the_map_renders_from_a_real_index(built):
    _r, ti = built
    html = tilemap_ui.render_map_section(ti, "data:image/png;base64,iVBORw0KGgo=")
    assert "<section" in html and "TILES" in html


# --- the provider seam -----------------------------------------------------

def test_the_builtin_abstains_when_there_is_no_terrain():
    """`has_terrain=False` is a legitimate answer downstream code respects, not
    a failure -- and it is the real situation for every AOI in `data/incoming`.

    The synthetic fixture DOES carry a DEM, so it tests the opposite branch;
    stripping it is what reproduces the shipped case.
    """
    r = synth.generate(size=128, seed=1)
    assert traversability.estimate(r).has_terrain is True, "fixture has a DEM"

    r.dem = None
    res = traversability.estimate(r, vehicle="wheeled")
    assert res.has_terrain is False
    assert res.score.dtype == np.float32
    assert 0.0 <= float(res.score.min()) and float(res.score.max()) <= 1.0
    assert any("DEM" in n or "slope" in n.lower() for n in res.notes)


@pytest.mark.parametrize("bad,why", [
    (lambda shape: np.full(shape, 2.0, "float32"), "out of [0,1]"),
    (lambda shape: np.full(shape, np.inf, "float32"), "+inf"),
    (lambda shape: np.full(shape, -np.inf, "float32"), "-inf"),
    (lambda shape: np.zeros((3, 3), "float32"), "wrong shape"),
])
def test_a_bad_provider_is_refused_at_the_seam(bad, why):
    """The whole reason the seam validates: an external model returning 0-255,
    +/-inf, or a mis-shaped raster must not reach a product as if it were a
    score. NaN is deliberately absent from this list -- it is now the
    contract's spelling of 'genuinely unmeasured', not a bug; see the NaN
    tests below."""
    r = synth.generate(size=64, seed=1)

    class Bad(traversability.TraversabilityProvider):
        def estimate(self, request):
            return traversability.TraversabilityResult(
                bad(request.labels.shape), "bad", True, [])

    traversability.register_provider("bad", Bad())
    with pytest.raises((ValueError, TypeError)):
        traversability.estimate(r, provider="bad")


def test_an_unknown_provider_raises_rather_than_falling_back():
    """Silently answering from the builtin after someone asked for their own
    model is the worst possible failure here."""
    r = synth.generate(size=64, seed=1)
    with pytest.raises((KeyError, ValueError)):
        traversability.estimate(r, provider="nope-not-registered")


# --- NaN: the unmeasured signal, not 0.0 ------------------------------------
#
# Review #4 in docs/handover.html: nodata scored 0.0 is indistinguishable from
# a genuine obstacle. The fix adds NaN to the contract as "genuinely
# unmeasured", distinct from 0.0's "measured and impassable".

def test_validate_accepts_partial_nan():
    """A provider marking some pixels unmeasured is not a bug at the seam."""
    r = synth.generate(size=64, seed=1)
    shape = r.labels.shape

    class PartiallyUnmeasured(traversability.TraversabilityProvider):
        def estimate(self, request):
            score = np.full(shape, 0.5, dtype=np.float32)
            score[0, 0] = np.nan
            return traversability.TraversabilityResult(score, "partial-nan", True, [])

    traversability.register_provider("partial-nan", PartiallyUnmeasured())
    res = traversability.estimate(r, provider="partial-nan")
    assert np.isnan(res.score[0, 0])
    assert res.score.dtype == np.float32
    assert float(res.score[0, 1]) == pytest.approx(0.5)


def test_range_check_still_fires_alongside_unrelated_nan():
    """NaN elsewhere in the raster must not blind `_validate` to a genuine
    out-of-range value -- the two checks are independent."""
    r = synth.generate(size=64, seed=1)
    shape = r.labels.shape

    class MixedBad(traversability.TraversabilityProvider):
        def estimate(self, request):
            score = np.full(shape, np.nan, dtype=np.float32)
            score[0, 0] = 1.7                  # the one non-NaN pixel is bad
            return traversability.TraversabilityResult(score, "mixed-bad", True, [])

    traversability.register_provider("mixed-bad", MixedBad())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        traversability.estimate(r, provider="mixed-bad")


def test_all_nan_provider_is_accepted():
    """A provider that could measure nothing at all is a legitimate answer,
    same spirit as `has_terrain=False`."""
    r = synth.generate(size=64, seed=1)
    shape = r.labels.shape

    class NothingMeasured(traversability.TraversabilityProvider):
        def estimate(self, request):
            return traversability.TraversabilityResult(
                np.full(shape, np.nan, dtype=np.float32), "blind", False, [])

    traversability.register_provider("blind", NothingMeasured())
    res = traversability.estimate(r, provider="blind")
    assert np.isnan(res.score).all()
    assert res.unmeasured_fraction == pytest.approx(1.0)


def test_unmeasured_fraction_reports_correctly():
    score = np.zeros((4, 4), dtype=np.float32)
    score[:2, :] = np.nan                      # exactly half the raster
    res = traversability.TraversabilityResult(score, "x", True, [])
    assert res.unmeasured_fraction == pytest.approx(0.5)

    block = res.caveat_block()
    assert "50.0%" in block and "unmeasured" in block

    fully_measured = traversability.TraversabilityResult(
        np.zeros((3, 3), "float32"), "x", True, [])
    assert fully_measured.unmeasured_fraction == 0.0
    assert "unmeasured" not in fully_measured.caveat_block()


def test_builtin_scores_nodata_as_nan_not_zero():
    """The bug itself: a sparse export's nodata must not read as impassable
    ground. `valid=False` -> NaN, not 0.0, from the builtin provider."""
    r = synth.generate(size=64, seed=3)
    valid = np.ones(r.shape, dtype=bool)
    valid[:16, :16] = False                    # a nodata corner, 1/16 of the tile
    r.valid = valid

    res = traversability.estimate(r, vehicle="wheeled")
    assert np.isnan(res.score[:16, :16]).all()
    assert np.isfinite(res.score[valid]).all()
    assert res.unmeasured_fraction == pytest.approx(valid[~valid].size / valid.size)


def test_builtin_mask_exclusion_stays_zero_not_nan():
    """`mask` is the caller narrowing scope on purpose, over ground that WAS
    measured -- that is a different claim from `valid`'s "no data here", so it
    keeps the old 0.0 convention rather than becoming NaN too."""
    r = synth.generate(size=64, seed=3)
    mask = np.ones(r.shape, dtype=bool)
    mask[:16, :16] = False

    res = traversability.estimate(r, vehicle="wheeled", mask=mask)
    assert (res.score[:16, :16] == 0.0).all()
    assert res.unmeasured_fraction == 0.0


def test_tile_aggregation_survives_partial_nan_traversability():
    """The consumer half of the fix: `build_tile_index` must not let a fully
    unmeasured tile's traversability round-trip through `_f` back into a
    misleading 0.0 with no signal attached (`s4_unmeasured` is that signal),
    and a partly unmeasured tile must still report the mean of what WAS
    measured, not NaN."""
    r = synth.generate(size=128, seed=2)
    ridx = build_regions(r)
    cidx = build_chips(r, size=128)
    rep = s1_audit.run(ridx)

    trav_score = np.full(r.shape, 0.6, dtype=np.float32)
    trav_score[:64, :64] = np.nan              # exactly the (0, 0) tile
    ti = tilemap.build_tile_index(r, ridx, cidx, None, rep, [],
                                  {"traversability": trav_score}, tile_px=64)
    by_rc = {(t["r"], t["c"]): t for t in ti["tiles"]}

    fully_unmeasured = by_rc[(0, 0)]
    assert fully_unmeasured["s4_unmeasured"]["traversability"] == 1.0
    assert fully_unmeasured["s4"]["traversability"] == 0.0

    fully_measured = by_rc[(1, 1)]
    assert fully_measured["s4"]["traversability"] == pytest.approx(0.6)
    assert fully_measured["s4_unmeasured"]["traversability"] == 0.0

    json.dumps(ti)                             # no raw NaN leaked into the payload
