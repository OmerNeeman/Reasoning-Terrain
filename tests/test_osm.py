"""The OSM layer: ingestion, the trust policy, and the invariants that matter.

Nothing here touches the network. The vectors are hand-built, which is also the
only way to test a tag table -- a real snapshot has whatever tags it has, and
the interesting cases (a `width` tag that contradicts `lanes`, a way tagged
`surface=unpaved` under a polygon labelled PavedRoad) are the rare ones.
"""

import numpy as np
import pytest

from segmap_digest import class_notes, synth
from segmap_digest.index import Region, build_regions
from segmap_digest.osm import tags as T
from segmap_digest.osm import trust
from segmap_digest.osm.burn import burn
from segmap_digest.osm.layer import OsmLayer
from segmap_digest.osm.partition import (
    blocks_in_window,
    build_blocks,
    build_road_graph,
)
from segmap_digest.osm.vectors import OsmVectors, OsmWay
from segmap_digest.solutions import s2_adjudicate, s3_triage
from segmap_digest.taxonomy import cid

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine                        # noqa: E402

GSD = 0.5
ORIGIN_LON, ORIGIN_LAT = 34.0, 31.0
# Degrees per pixel at this latitude, near enough for a fixture: the tests care
# that projection round-trips, not that the geodesy is survey grade.
DEG = GSD / 111_320.0


@pytest.fixture(scope="module")
def tile():
    r = synth.generate(size=256, gsd=GSD, seed=5)
    r.transform = Affine(DEG, 0.0, ORIGIN_LON, 0.0, -DEG, ORIGIN_LAT)
    r.crs = rasterio.crs.CRS.from_epsg(4326)
    return r


def _way(wid, tags, pix_pts):
    """A way from (row, col) pixel coordinates, so tests can say where it lands."""
    lon = np.array([ORIGIN_LON + c * DEG for _r, c in pix_pts])
    lat = np.array([ORIGIN_LAT - r * DEG for r, _c in pix_pts])
    return OsmWay(id=wid, tags=tags, lon=lon, lat=lat,
                  nodes=tuple(range(wid * 100, wid * 100 + len(pix_pts))))


@pytest.fixture(scope="module")
def vectors():
    """A cross: two streets meeting in the middle, plus one building."""
    return OsmVectors(
        ways=[
            OsmWay(id=1, tags={"highway": "residential", "name": "North Street",
                               "surface": "asphalt"},
                   lon=np.array([ORIGIN_LON + 128 * DEG] * 2),
                   lat=np.array([ORIGIN_LAT, ORIGIN_LAT - 255 * DEG]),
                   nodes=(10, 11)),
            OsmWay(id=2, tags={"highway": "residential", "name": "East Street",
                               "surface": "unpaved"},
                   lon=np.array([ORIGIN_LON, ORIGIN_LON + 255 * DEG]),
                   lat=np.array([ORIGIN_LAT - 128 * DEG] * 2),
                   nodes=(11, 12)),
        ],
        bbox=(ORIGIN_LAT - 256 * DEG, ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 256 * DEG),
    )


@pytest.fixture(scope="module")
def layer(tile, vectors):
    b = burn(tile, vectors, layers=("road",))
    g = build_road_graph(tile, vectors)
    return OsmLayer(vectors, b, g, build_blocks(tile, b, vectors, graph=g))


# --- tag tables ------------------------------------------------------------

def test_width_tag_beats_the_grade_table():
    """`width` is a measurement of THIS way; the table is a guess about its grade."""
    assert T.half_width_m({"highway": "residential"}) == T.HALF_WIDTH_M["residential"]
    assert T.half_width_m({"highway": "residential", "width": "6 m"}) == 3.0


def test_lanes_is_a_floor_not_the_answer():
    wide = T.half_width_m({"highway": "residential", "lanes": "4"})
    assert wide == pytest.approx(4 * T.LANE_WIDTH_M / 2)
    assert wide > T.HALF_WIDTH_M["residential"]


def test_a_surface_expectation_cites_its_evidence():
    """A finding that cites `surface=asphalt` is actionable; one that cites
    'residential roads are usually paved' is a prior with an unmeasured error
    rate, and the caller has to be able to tell them apart."""
    cls, why = T.expected_road_class({"highway": "residential", "surface": "asphalt"})
    assert cls == "PavedRoad" and why == "surface=asphalt"
    cls, why = T.expected_road_class({"highway": "residential"})
    assert cls == "PavedRoad" and why.endswith("(weak)")


def test_nonsense_width_is_ignored_rather_than_believed():
    assert T.half_width_m({"highway": "service", "width": "900"}) == \
        T.HALF_WIDTH_M["service"]


# --- projection and burn ---------------------------------------------------

def test_a_way_lands_where_it_was_put(tile, layer):
    """The vertical street was drawn down column 128; the corridor has to be
    there and not out at column 5 -- except where the OTHER street crosses it,
    which is why this looks away from row 128."""
    road = layer.burned.mask("road")
    assert road[:, 128].all()
    assert road[128, :].all()
    assert not road[:100, 5].any()


def test_burn_measures_st_under_every_way(layer):
    for fp in layer.burned.by_kind("road"):
        assert fp.n_px > 0
        assert fp.hist.sum() == fp.n_px


def test_an_ungeoreferenced_raster_refuses_rather_than_guesses(vectors):
    """A join against a raster whose position is unknown produces a plausible
    partition of the wrong ground, which is worse than no partition."""
    from segmap_digest.osm.fetch import bbox_of_raster

    plain = synth.generate(size=64, seed=1)
    with pytest.raises(ValueError, match="no affine transform"):
        bbox_of_raster(plain)


# --- partition -------------------------------------------------------------

def test_blocks_tile_the_ground_exactly(tile, layer):
    """Every valid pixel is in exactly one block or in the corridor. A partition
    with holes silently changes every area share computed from it."""
    lab = layer.blocks.label_array
    corridor = layer.burned.mask("road")
    valid = tile.valid if tile.valid is not None else np.ones(tile.shape, bool)
    in_block = lab > 0
    assert not (in_block & corridor).any()
    assert (in_block | corridor)[valid].all()


def test_a_cross_makes_four_blocks(layer):
    assert len(layer.blocks.blocks) == 4


def test_intersections_come_from_shared_node_ids(layer):
    assert layer.graph.node_source == "osm-node-ids"
    assert layer.graph.n_junctions == 1
    assert set(layer.graph.junctions[0].unique_names) == {"North Street",
                                                          "East Street"}


def test_blocks_are_named_by_their_streets(layer):
    assert all("North Street" in b.label or "East Street" in b.label
               for b in layer.blocks.blocks)


def test_a_window_reports_both_shares(tile, layer):
    """`share_of_window` describes the tile, `share_of_block` says whether any
    block-level figure may be quoted from it. Reporting one as the other is how
    a subset answer gets presented as an AOI answer."""
    rows = blocks_in_window(layer.blocks, (0, 0, 128, 128), raster=tile)
    assert len(rows) == 1
    wb = rows[0]
    # Not 100%: the last few rows and columns of the window are the corridor
    # itself, which belongs to no block.
    assert wb.share_of_window > 0.85
    assert wb.quotable                            # the window holds all of it
    rows = blocks_in_window(layer.blocks, (0, 0, 64, 64), raster=tile)
    assert rows[0].share_of_block < 0.5
    assert not rows[0].quotable
    assert rows[0].truncated


def test_degenerate_partition_says_so(tile):
    """Open ground with one track through it is one block, and the output has to
    say that rather than present it as a spatial analysis."""
    lone = OsmVectors(
        ways=[OsmWay(id=9, tags={"highway": "track"},
                     lon=np.array([ORIGIN_LON, ORIGIN_LON + 30 * DEG]),
                     lat=np.array([ORIGIN_LAT - 5 * DEG] * 2), nodes=(1, 2))],
        bbox=(0, 0, 0, 0))
    b = burn(tile, lone, layers=("road",))
    idx = build_blocks(tile, b, lone)
    assert idx.degenerate
    assert "not a partition" in idx.note or "no spatial meaning" in idx.note


# --- the trust policy (D2) -------------------------------------------------

def test_state_evidence_cannot_overturn_the_later_observation():
    """Owner policy: OSM is the reference for the land, ST is the latest word on
    what it looks like now. So a mapped footprint over bare ground argues for
    House and must not be able to flip the label by itself."""
    gap = trust.support(trust.AXIS_STATE) - trust.contradict(trust.AXIS_STATE)
    assert s2_adjudicate.W_REFERENCE * gap < s2_adjudicate.SWITCH_MARGIN


def test_existence_evidence_can_overturn_it():
    """...and the reverse, on the axis where OSM IS the authority: a polygon of
    soil sitting inside a mapped street corridor should be flippable."""
    gap = trust.support(trust.AXIS_EXISTENCE) - trust.contradict(trust.AXIS_EXISTENCE)
    assert s2_adjudicate.W_REFERENCE * gap >= s2_adjudicate.SWITCH_MARGIN


def test_every_check_family_declares_its_axis():
    from segmap_digest.osm import checks

    for kind in checks.CHECK_FAMILIES:
        assert kind in trust.AXIS, f"{kind} would fall back to DEFAULT"


# --- S2 stays what it was without a reference map --------------------------

def test_s2_is_bit_identical_without_osm(tile):
    """The acceptance criterion for the whole reference term: joining nothing
    changes nothing."""
    ridx = build_regions(tile)
    rid = ridx.regions[len(ridx.regions) // 2].id
    a = s2_adjudicate.adjudicate(ridx, rid)
    for name, ev in a.ranked:
        assert ev.reference is None
        assert ev.total == ev.own


def test_reference_term_needs_the_raster_it_was_burned_onto(tile, layer):
    ridx = build_regions(tile)
    with pytest.raises(ValueError, match="raster"):
        s2_adjudicate.adjudicate(ridx, ridx.regions[0].id, osm=layer)


# --- S2 renormalises when OSM covers only SOME of a ranking's candidates ---

def test_s2_renormalizes_the_whole_ranking_when_only_some_candidates_have_a_reference(
        tile, layer):
    """The bug: `reference_score` returns None per candidate, so a ranking
    where OSM has an opinion about SOME candidates and not others used to mix
    two scales -- a blended 0.7*own+0.3*ref for the OSM-covered candidates
    against a raw `own` for the ones OSM said nothing about. Region 99 on this
    fixture is exactly that case: a `TerraRosa` soil region 33% inside the
    mapped road corridor -- below STRONG_OVERLAP, so OSM stays silent about
    the soil candidates -- with `PavedRoad` on the shortlist (and WITH a real
    reference score) because of that same overlap."""
    ridx = build_regions(tile)
    adj = s2_adjudicate.adjudicate(ridx, 99, osm=layer, raster=tile)
    by_name = dict(adj.ranked)

    assert by_name["PavedRoad"].reference is not None, \
        "fixture drifted: PavedRoad was expected to get a real OSM reference"
    silent = {n: ev for n, ev in adj.ranked if ev.reference is None}
    assert silent, "fixture drifted: expected some candidates OSM said nothing about"

    # (a) every candidate in the ranking went through the SAME formula: a
    # silent candidate's total is the blend against the neutral stand-in, not
    # a fall-through to raw `own` the way it would have been pre-fix.
    for name, ev in adj.ranked:
        assert ev.ranking_has_reference
        expected = ((1 - s2_adjudicate.W_REFERENCE) * ev.own
                    + s2_adjudicate.W_REFERENCE
                    * (ev.reference if ev.reference is not None
                       else s2_adjudicate.NEUTRAL_REFERENCE))
        assert ev.total == pytest.approx(expected)

    # A concrete instance of (a): LimestoneStoneyTerrain's own score is not
    # 0.5, so if it had fallen through to raw `own` (the pre-fix behaviour for
    # a None reference) its total would equal its own -- it must not.
    lime = by_name["LimestoneStoneyTerrain"]
    assert lime.reference is None
    assert lime.own != 0.5
    assert lime.total != lime.own
    assert lime.total == pytest.approx(0.7 * lime.own + 0.3 * 0.5)


def test_s2_reference_penalty_no_longer_flips_a_stronger_candidate():
    """Direct construction of the case the owner's bug report describes: `A`
    has both a higher `own` AND real (if lukewarm) OSM support; sibling `B` in
    the same ranking is untouched by OSM. Pre-fix, `total` fell back to raw
    `own` for B while diluting A towards A's weak reference score -- so B's
    raw 0.60 beat A's blended 0.58 even though A was the better-supported
    candidate on every axis that had an opinion. Post-fix, B is diluted by the
    same formula (with a neutral stand-in for its missing reference) and A
    wins, as it should."""
    # context == morphology == geometry makes `own` equal that value exactly,
    # regardless of the term weights (they sum to 1.0) -- the cleanest way to
    # pin `own` for a test that is about the TOTAL formula, not the terms.
    a = s2_adjudicate.Evidence(context=0.70, morphology=0.70, geometry=0.70,
                               notes=[], reference=0.30,
                               ranking_has_reference=True)
    b = s2_adjudicate.Evidence(context=0.60, morphology=0.60, geometry=0.60,
                               notes=[], reference=None,
                               ranking_has_reference=True)
    assert a.own == pytest.approx(0.70) and b.own == pytest.approx(0.60)

    # What the OLD code did: A blended down to 0.58; B, with no reference,
    # fell straight through to its raw own of 0.60 -- B wins despite A being
    # better-supported everywhere OSM had an opinion.
    old_a_total = 0.7 * a.own + 0.3 * a.reference
    assert old_a_total == pytest.approx(0.58)
    assert b.own > old_a_total, "fixture no longer demonstrates the old penalty"

    # What the NEW code does: B is diluted by the same formula B's sibling A
    # was, using the neutral stand-in for its missing reference -- and A wins.
    assert b.total != b.own
    assert b.total == pytest.approx(0.7 * 0.60 + 0.3 * s2_adjudicate.NEUTRAL_REFERENCE)
    assert a.total == pytest.approx(old_a_total)     # unaffected: A had a real reference
    assert a.total > b.total, "A is still penalised for its weak OSM support"


# --- S3 --------------------------------------------------------------------

def test_a_policy_rule_the_unit_cannot_measure_is_reported(tile, layer):
    """A rule that silently stops firing is a policy that no longer means what
    its author approved."""
    _p, _sel, missing, _labels = s3_triage.run_blocks(tile, layer, "vehicles")
    assert any("dist_to." in m for m in missing)


def test_a_meaningless_feature_is_reported_not_faked(tile, layer):
    """`osm.dist_road` cannot mean anything on a block: a block is DEFINED as
    the complement of the road corridor, so every block is zero metres from a
    road. Rather than fill in that zero -- which would make the rule fire on
    everything -- the block unit does not provide the feature, and says so.
    The settlement policy's other two rules do fire."""
    _p, _sel, missing, _labels = s3_triage.run_blocks(tile, layer, "settlement")
    assert missing == ["osm.dist_road < 30"]
    from segmap_digest.osm import chipfeat

    assert "building_frac" in chipfeat.BLOCK_FEATURES
    assert "n_junctions" in chipfeat.BLOCK_FEATURES
    assert "dist_road" not in chipfeat.BLOCK_FEATURES


def test_block_unit_scores_the_same_way_chips_do(tile, layer):
    _p, sel, _m, labels = s3_triage.run_blocks(tile, layer, "settlement")
    assert sel.selected
    assert all(s.chip.id in labels for s in sel.selected)


# --- class notes -----------------------------------------------------------

def test_a_fresh_template_reads_as_undocumented():
    """The template seeds `what_it_is` from taxonomy.py so an expert has
    something to correct. If that counted as filled, a coverage report would say
    47 of 47 the moment the template was committed."""
    ns = class_notes.parse(class_notes.template())
    assert ns.documented == []
    assert not ns


def test_notes_parse_fields_and_keep_unknown_ones():
    ns = class_notes.parse(
        "## TerraRosa\n"
        "aka: terra rossa\n"
        "what_it_is: A red clay soil.\n"
        "  It continues on this line.\n"
        "confused_with: ClayeySoil — redder and stonier; Rendzina — paler\n"
        "smell: earthy\n"
    )
    note = ns.get("TerraRosa")
    assert note.get("aka") == "terra rossa"
    assert "continues on this line" in note.get("what_it_is")
    assert note.confused_with == [("ClayeySoil", "redder and stonier"),
                                  ("Rendzina", "paler")]
    assert note.extra == {"smell": "earthy"}


def test_a_typo_in_confused_with_is_reported_not_swallowed():
    ns = class_notes.parse("## TerraRosa\nconfused_with: ClayeySoiI — typo\n")
    assert ns.get("TerraRosa").confused_with == []
    assert any("ClayeySoiI" in p for p in class_notes.check(ns))


def test_confused_with_reaches_the_s2_shortlist(tile):
    """The point of the machine-read fields: an analyst's measured confusion
    beats the taxonomy's guess about which classes are alike."""
    ridx = build_regions(tile)
    r = next(x for x in ridx.regions if x.class_name == "Maquis")
    ns = class_notes.parse("## Maquis\nconfused_with: Water — deep shade "
                           "reads as water on a dark slope\n")
    plain = s2_adjudicate.candidates(ridx, r)
    with_notes = s2_adjudicate.candidates(ridx, r, notes=ns)
    assert cid("Water") not in plain
    assert cid("Water") in with_notes


def test_scale_note_overrides_the_guessed_area_band():
    """`scale` is documented as replacing S2's guessed area bands once wired
    -- this is that wiring. An analyst's parsed range beats the hardcoded
    `AREA_BAND` guess when it is present, and a class with no guess there at
    all still gets a real geometry constraint once an analyst writes one
    down."""
    def region(area_m2, class_name):
        return Region(id=1, class_id=cid(class_name), area_px=1,
                      area_m2=area_m2, perimeter_m=1.0, compactness=1.0,
                      elongation=1.0, bbox=(0, 0, 1, 1), centroid=(0.0, 0.0))

    car_id = cid("Car")
    # Car's hardcoded AREA_BAND is (4, 30) m2; 150 m2 is well outside it.
    r = region(150.0, "Car")
    plain_score, plain_notes = s2_adjudicate._geometry_score(r, car_id)
    assert plain_score == 0.05
    assert plain_notes  # the hardcoded guess flagged the miss

    ns = class_notes.parse("## Car\nscale: 100-200 m2\n")
    overridden_score, overridden_notes = s2_adjudicate._geometry_score(
        r, car_id, notes=ns)
    assert overridden_score == 1.0     # 150 m2 is inside the analyst's range
    assert not overridden_notes        # inside the band -- nothing to flag

    # A class with no AREA_BAND guess at all is unconstrained today...
    grass_id = cid("GreenGrassland")
    r2 = region(60.0, "GreenGrassland")
    unconstrained_score, _ = s2_adjudicate._geometry_score(r2, grass_id)
    assert unconstrained_score == 0.5

    # ...but a scale note alone can now constrain it, with no AREA_BAND entry
    # to fall back on.
    ns2 = class_notes.parse("## GreenGrassland\nscale: 5-20 m2\n")
    constrained_score, constrained_notes = s2_adjudicate._geometry_score(
        r2, grass_id, notes=ns2)
    assert constrained_score == 0.05   # 60 m2 is outside the analyst's 5-20 band
    assert any("analyst scale note" in n for n in constrained_notes)


def test_notes_render_says_which_classes_it_has_nothing_for():
    ns = class_notes.parse("## Maquis\nwhat_it_is: dense scrub\n")
    text = ns.render(["Maquis", "TerraRosa"])
    assert "dense scrub" in text
    assert "TerraRosa" in text and "no analyst notes on file" in text
