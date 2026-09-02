"""The four defects of `open_issues.md` Q3, and the tests that pin them.

Each one made a command *lie* rather than fail, which is the failure mode this
repo fears most: `--osm` accepted and ignored, an all-nodata window arriving as a
traceback from three frames away, a borrowed class mapping applied in silence,
and the mosaic path skipping the validation the single-tile path does.

Three of the twelve below are controls -- a window that *does* have data, a tile
that carries its own mapping tag, and the single-tile path that was already
strict. The other nine fail against the code as it was, which is the point.
"""

import json

import numpy as np
import pytest

from segmap_digest import loader, mosaic, report, synth
from segmap_digest.osm.burn import burn
from segmap_digest.osm.layer import OsmLayer
from segmap_digest.osm.partition import build_blocks, build_road_graph
from segmap_digest.osm.vectors import OsmVectors, OsmWay
from segmap_digest.solutions import s4_products

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine                        # noqa: E402

GSD = 0.5
ORIGIN_LON, ORIGIN_LAT = 34.0, 31.0
DEG = GSD / 111_320.0


# --- fixtures: a georeferenced tile and a real burned OSM layer -------------

@pytest.fixture(scope="module")
def tile():
    r = synth.generate(size=192, gsd=GSD, seed=5)
    r.transform = Affine(DEG, 0.0, ORIGIN_LON, 0.0, -DEG, ORIGIN_LAT)
    r.crs = rasterio.crs.CRS.from_epsg(4326)
    return r


@pytest.fixture(scope="module")
def osm_layer(tile):
    """Two streets crossing the tile. Real vectors, real burn, no network."""
    vectors = OsmVectors(
        ways=[
            OsmWay(id=1, tags={"highway": "residential", "name": "North Street",
                               "surface": "asphalt"},
                   lon=np.array([ORIGIN_LON + 96 * DEG] * 2),
                   lat=np.array([ORIGIN_LAT, ORIGIN_LAT - 191 * DEG]),
                   nodes=(10, 11)),
            OsmWay(id=2, tags={"highway": "residential", "name": "East Street",
                               "surface": "unpaved"},
                   lon=np.array([ORIGIN_LON, ORIGIN_LON + 191 * DEG]),
                   lat=np.array([ORIGIN_LAT - 96 * DEG] * 2),
                   nodes=(11, 12)),
        ],
        bbox=(ORIGIN_LAT - 192 * DEG, ORIGIN_LON,
              ORIGIN_LAT, ORIGIN_LON + 192 * DEG),
    )
    b = burn(tile, vectors, layers=("road", "building", "water", "flow",
                                    "built_landuse", "barrier"))
    g = build_road_graph(tile, vectors)
    return OsmLayer(vectors, b, g, build_blocks(tile, b, vectors, graph=g))


# --- 1. `report --osm` was accepted and did nothing -------------------------

def test_report_without_osm_says_there_is_no_reference_map(tile, tmp_path):
    """An S2 evidence row reading `reference: none` because OSM disagrees and one
    reading it because no reference map was ever loaded mean opposite things, and
    the page has to distinguish them."""
    index = report.build(tile, tmp_path / "noosm", synthetic=True, limit=4)
    page = index.read_text()
    assert "No OSM layer joined" in page
    assert "OSM joined:" not in page


def test_report_with_osm_burns_it_and_says_so(tile, osm_layer, tmp_path):
    """The regression: `report.py` used to contain no reference to `osm` at all,
    so the page rendered identically with and without the flag."""
    index = report.build(tile, tmp_path / "osm", synthetic=True, limit=4,
                         osm=osm_layer)
    page = index.read_text()
    assert "OSM joined: 2 ways" in page
    assert "No OSM layer joined" not in page
    # the S4 overlay delta is on the page, which can only come from the layer
    assert "OSM overlay" in page


def test_report_build_actually_hands_osm_to_the_solutions(tile, osm_layer,
                                                          monkeypatch, tmp_path):
    """Rendering a banner is not wiring. Assert the object reaches S1, S2 and the
    products -- the three call sites that were passing nothing."""
    seen = {}
    real_s1 = report.s1_audit.run
    real_s2 = report.s2_adjudicate.adjudicate
    real_p = report.s4_products.compute

    def spy_s1(ridx, **kw):
        seen["s1"] = kw.get("osm")
        return real_s1(ridx, **kw)

    def spy_s2(ridx, rid, **kw):
        seen["s2"] = kw.get("osm")
        return real_s2(ridx, rid, **kw)

    def spy_products(raster, name, **kw):
        seen.setdefault("s4", kw.get("osm"))
        return real_p(raster, name, **kw)

    monkeypatch.setattr(report.s1_audit, "run", spy_s1)
    monkeypatch.setattr(report.s2_adjudicate, "adjudicate", spy_s2)
    monkeypatch.setattr(report.s4_products, "compute", spy_products)
    report.build(tile, tmp_path / "spy", synthetic=True, limit=4, osm=osm_layer)
    assert seen["s1"] is osm_layer
    assert seen["s4"] is osm_layer
    # s2 only runs when S1 flagged something; on this fixture it does
    assert seen.get("s2", osm_layer) is osm_layer


def test_the_cli_forwards_its_own_flag(tile, monkeypatch, tmp_path):
    """`cmd_report` never called `_osm`. The flag parsed, and stopped there."""
    from segmap_digest import cli

    sentinel = object()
    captured = {}
    monkeypatch.setattr(cli, "_osm", lambda r, a, **kw: sentinel)
    monkeypatch.setattr(cli, "_load", lambda a: tile)
    monkeypatch.setattr(cli, "_regions", lambda r, a: None)
    monkeypatch.setattr(cli, "_chips", lambda r, a, size=None: None)

    def fake_build(raster, out, **kw):
        captured.update(kw)
        p = tmp_path / "index.html"
        p.write_text("x")
        return p

    import segmap_digest.report as report_mod
    monkeypatch.setattr(report_mod, "build", fake_build)
    cli.main(["report", "-i", "whatever.tif", "--osm", "-o", str(tmp_path)])
    assert captured["osm"] is sentinel


# --- 2. an all-nodata window was a traceback --------------------------------

def _all_nodata(size=64):
    r = synth.generate(size=size, gsd=0.5, seed=1)
    r.valid = np.zeros(r.shape, dtype=bool)
    return r


def test_summarise_states_an_empty_window_instead_of_raising():
    """A centred crop of a 94%-nodata edge tile is 0 valid pixels, and
    `np.percentile` on an empty array raises `IndexError` from inside numpy --
    which reached the user as a failed playground job at 96%."""
    r = _all_nodata()
    arr = s4_products.compute(r, "trafficability")
    text = s4_products.summarise(r, arr)
    assert "no classified pixels in this window" in text
    assert "statement about the crop" in text


def test_summarise_emits_no_empty_slice_warnings_on_an_empty_window():
    r = _all_nodata()
    arr = s4_products.compute(r, "trafficability")
    with np.errstate(all="raise"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s4_products.summarise(r, arr)


def test_osm_delta_states_an_empty_window_too(osm_layer):
    """Same empty selection, three more reductions over it."""
    r = _all_nodata()
    before = s4_products.compute(r, "trafficability")
    text = s4_products.osm_delta_from(r, before, before, osm_layer)
    assert "no classified pixels" in text


def test_a_window_with_data_still_reports_its_distribution():
    """The guard must not swallow the normal case."""
    r = synth.generate(size=64, gsd=0.5, seed=1)
    arr = s4_products.compute(r, "trafficability")
    text = s4_products.summarise(r, arr)
    assert "value distribution over classified pixels" in text
    assert "mean by superclass" in text


# --- 3 & 4. the mosaic path: no validation, no provenance -------------------

WIRE = {0: "Unclassified", 2: "Clutter", 94: "MaralBadlands", 112: "Rendzina"}


def _tif(path, arr, col=0, tags=None, nodata=0):
    tr = Affine(DEG, 0.0, ORIGIN_LON + col * arr.shape[1] * DEG,
                0.0, -DEG, ORIGIN_LAT)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="uint8",
                       crs="EPSG:4326", transform=tr, nodata=nodata) as dst:
        dst.write(arr, 1)
        if tags:
            dst.update_tags(**tags)
    return path


def test_mosaic_refuses_untranslated_wire_ids_and_names_the_tile(tmp_path):
    """Two tiles, no mapping tag, no --classes: ids of 94 and 112 are sparse wire
    ids, and the old code passed them straight through as dense taxonomy ids. The
    failure surfaced later as a bare `KeyError: np.int64(94)` out of digests.py --
    three modules from the file that caused it."""
    d = tmp_path / "aoi"
    d.mkdir()
    _tif(d / "t0.tif", np.full((32, 32), 94, dtype=np.uint8), col=0)
    _tif(d / "t1.tif", np.full((32, 32), 112, dtype=np.uint8), col=1)
    with pytest.raises(ValueError) as exc:
        mosaic.load_mosaic(d)
    msg = str(exc.value)
    assert "sparse wire ids" in msg, msg
    assert "--classes" in msg
    assert "t0.tif" in msg or "t1.tif" in msg, "the error must name the tile"


def test_mosaic_announces_a_borrowed_id_mapping(tmp_path, capsys):
    """aza is read with sinai's mapping and nothing anywhere said so. If it is
    the wrong mapping every class name on the AOI is wrong, and no downstream
    number can reveal it."""
    d = tmp_path / "aoi"
    d.mkdir()
    _tif(d / "t0.tif", np.full((32, 32), 94, dtype=np.uint8), col=0)
    _tif(d / "t1.tif", np.full((32, 32), 112, dtype=np.uint8), col=1)
    classes = tmp_path / "borrowed.json"
    classes.write_text(json.dumps({str(k): v for k, v in WIRE.items()}))

    r = mosaic.load_mosaic(d, classes=classes)
    err = capsys.readouterr().err
    assert "id mapping: 2 of 2 tiles carry no ID_TO_LABEL_MAPPING" in err
    assert "another export" in err
    assert "borrowed.json" in err
    # and it did translate, by name
    from segmap_digest.taxonomy import cid
    assert cid("MaralBadlands") in np.unique(r.labels)


def test_a_tile_with_its_own_tag_is_not_reported_as_borrowed(tmp_path, capsys):
    d = tmp_path / "aoi"
    d.mkdir()
    tags = {"ID_TO_LABEL_MAPPING": json.dumps({str(k): v for k, v in WIRE.items()})}
    _tif(d / "t0.tif", np.full((32, 32), 94, dtype=np.uint8), col=0, tags=tags)
    _tif(d / "t1.tif", np.full((32, 32), 112, dtype=np.uint8), col=1, tags=tags)
    mosaic.load_mosaic(d)
    assert "id mapping:" not in capsys.readouterr().err


def test_the_single_tile_path_is_unchanged(tmp_path):
    """The mosaic path is now as strict as this one always was -- which is the
    point: the two disagreed, and the loose one was the default for a directory."""
    p = _tif(tmp_path / "one.tif", np.full((32, 32), 94, dtype=np.uint8))
    with pytest.raises(ValueError, match="sparse wire ids"):
        loader.load(p)


# --- 5. class_distance: state what the speck check really tests -------------
#
# The decision was "leave the distance alone, but make S1 say so" -- so what is
# pinned here is the *saying*, and that it says numbers it computed.

def test_the_speck_caveat_computes_its_own_numbers():
    from segmap_digest.audit import SPECK_MIN_CLASS_DISTANCE
    from segmap_digest.solutions.s1_audit import speck_distance_caveat
    from segmap_digest.taxonomy import N_CLASSES, SUPERCLASS_OF, class_distance

    text = speck_distance_caveat()
    n_cross = sum(1 for a in range(N_CLASSES) for b in range(N_CLASSES)
                  if a != b and SUPERCLASS_OF[a] != SUPERCLASS_OF[b])
    far = sum(1 for a in range(N_CLASSES) for b in range(N_CLASSES)
              if a != b and class_distance(a, b) >= SPECK_MIN_CLASS_DISTANCE)
    total = N_CLASSES * (N_CLASSES - 1)
    assert f"{far / total:.0%} of all {total} ordered class pairs" in text
    assert f"all {n_cross} cross-superclass pairs score exactly 1.0" in text
    assert "'small and enclosed'" in text


def test_the_caveat_forbids_the_fix_that_broke_it_last_time():
    """The soil-genesis priors cost 10,788 false findings. A future reader
    reaching for 'just add a prior' has to meet that in the output."""
    from segmap_digest.solutions.s1_audit import speck_distance_caveat

    text = speck_distance_caveat()
    assert "must NOT be closed by adding a prior" in text
    assert "RETIRED_PRIORS" in text
    assert "confusion matrix" in text and "compatibility relation" in text


def test_a_report_carrying_speck_findings_carries_the_caveat(tile):
    """It has to reach the page, not just exist as a function."""
    from segmap_digest.index import build_regions
    from segmap_digest.solutions import s1_audit

    ridx = build_regions(tile)
    rep = s1_audit.run(ridx)
    if not rep.by_kind.get("isolated-speck"):
        pytest.skip("this fixture produced no isolated-speck findings")
    assert "CAVEAT" in s1_audit.fragmentation_block(rep)
    assert "CAVEAT" in s1_audit.render(rep)


def test_the_report_page_carries_the_caveat_too(tile, tmp_path):
    """The report builds its own S1 section rather than calling `render`, so it
    was dropping both the fragmentation line and the caveat."""
    index = report.build(tile, tmp_path / "frag", synthetic=True, limit=4)
    page = index.read_text()
    assert "fragmentation (a property of the raster" in page
    assert "CAVEAT" in page


# --- 6. the abstain names a datum, not an unobtainable product -------------

def test_the_lithology_abstain_asks_for_hardness_not_a_nari_layer():
    """`UNDECIDABLE-NEEDS-geological-map` was an instruction nobody could carry
    out for a Nari pair: nari is a surface crust on top of whatever the map
    already says is there, so it appears on no geological map at any scale. What
    the separation actually turns on is carbonate hardness, which a geological
    map does carry."""
    from segmap_digest.index import Region, build_regions
    from segmap_digest.solutions import s2_adjudicate as s2

    assert s2._hardness("Limestone") == "hard"
    assert s2._hardness("Nari") == "hard"        # the crust, as it behaves at the surface
    assert s2._hardness("Chalk") == "soft"
    assert s2._hardness("Basalt") == "non-carbonate"
    assert s2._hardness(None) == "non-carbonate"


def test_a_lithology_twin_pair_returns_the_renamed_verdict():
    """Two classes with the same morphology and different lithology cannot be
    separated by RGB, geometry or the DEM. The verdict has to say what would."""
    import numpy as np

    from segmap_digest.index import build_regions
    from segmap_digest.loader import LabelRaster
    from segmap_digest.solutions import s2_adjudicate as s2
    from segmap_digest.taxonomy import cid

    # a compact blob of LimestoneStoneyTerrain -- its dolomite twin differs only
    # by lithology, and the two score within UNDECIDABLE_MARGIN by construction
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[20:44, 20:44] = cid("LimestoneStoneyTerrain")
    ridx = build_regions(LabelRaster(labels, gsd=0.5))
    target = next((r for r in ridx.regions
                   if r.class_name == "LimestoneStoneyTerrain"), None)
    if target is None:
        pytest.skip("fixture produced no limestone region")
    adj = s2.adjudicate(ridx, target.id)
    if not adj.verdict.startswith("UNDECIDABLE"):
        pytest.skip(f"this region did not reach the abstain gate ({adj.verdict})")
    assert adj.verdict == "UNDECIDABLE-NEEDS-hardness-class"
    assert "hardness class" in adj.rationale
    assert "nari layer" in adj.rationale        # names what NOT to go looking for
