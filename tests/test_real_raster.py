"""Regressions for everything the synthetic fixture never exercised.

The fixture is a square 1024 px uint8 array, dense ids 0..N-1, no nodata, no
georeference. Real Smart Terrain exports are 8192 px, sparse wire ids up to 241,
nodata=0 in a geographic CRS, and arrive as a directory of adjacent tiles. Every
test below corresponds to something that was actually wrong when the first real
tile was pointed at this code.
"""

import json
import time

import numpy as np
import pytest

from segmap_digest import audit, digests, loader, taxonomy
from segmap_digest.index import build_chips, build_regions
from segmap_digest.loader import LabelRaster
from segmap_digest.solutions import s2_adjudicate, s4_products, s5_query

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine                       # noqa: E402


# The producer's own mapping, as read from the ID_TO_LABEL_MAPPING tag: sparse
# ids in 0..241, in taxonomy order.
WIRE = {0: "Unclassified", 2: "Clutter", 3: "Shadow", 5: "BrickWall",
        10: "House", 12: "GreenGrassland", 112: "Rendzina", 132: "DryGrassland",
        238: "ChalkSmoothRockSlopes", 241: "ChalkTerrace"}


def _write_tif(path, arr, transform, crs="EPSG:4326", nodata=0, tags=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="uint8",
                       crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)
        if tags:
            dst.update_tags(**tags)
    return path


# --- wire ids are not taxonomy ids -----------------------------------------

def test_wire_ids_are_remapped_by_name(tmp_path):
    """A real export's max id is 241, not 44. Reading those as class ids would
    have been an IndexError at best and silent nonsense at worst."""
    arr = np.array([[0, 112], [132, 241]], dtype=np.uint8)
    tif = _write_tif(tmp_path / "t.tif", arr, Affine(5e-6, 0, 34.0, 0, -5e-6, 31.0),
                     tags={"ID_TO_LABEL_MAPPING": json.dumps({str(k): v
                                                              for k, v in WIRE.items()})})
    r = loader.load(tif)
    assert r.labels[0, 1] == taxonomy.cid("Rendzina")
    assert r.labels[1, 0] == taxonomy.cid("DryGrassland")
    assert r.labels[1, 1] == taxonomy.cid("ChalkTerrace")


def test_unmapped_wire_ids_are_refused_not_folded(tmp_path):
    """Dropping an unknown class into Unclassified would delete real terrain
    from every number downstream. It has to raise."""
    with pytest.raises(ValueError, match="not in taxonomy"):
        loader.wire_lut({0: "Unclassified", 7: "SomethingNewTheModelLearned"})


def test_value_out_of_mapping_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no entry in its id mapping"):
        loader.remap(np.array([[0, 250]], dtype=np.uint8), WIRE)


def test_taxonomy_covers_the_producer_vocabulary():
    """The two classes the first draft of taxonomy.py was missing. Their absence
    shifted every id after LimestoneStoneyTerrain by one."""
    assert "LimestoneHardRockLineament" in taxonomy.BY_NAME
    assert "ChalkTerrace" in taxonomy.BY_NAME
    assert all(n in taxonomy.BY_NAME for n in WIRE.values())


# --- georeference ----------------------------------------------------------

def test_gsd_from_geographic_crs_is_metres(tmp_path):
    """The tiles are EPSG:4326 with a transform of ~5e-06 degrees. Taking
    abs(transform.a) at face value reported 5e-06 m/px, making every area in the
    report about ten orders of magnitude too small."""
    arr = np.ones((4, 4), dtype=np.uint8) * 112
    tif = _write_tif(tmp_path / "g.tif", arr, Affine(5e-6, 0, 34.0, 0, -5e-6, 31.0),
                     tags={"ID_TO_LABEL_MAPPING": json.dumps({str(k): v
                                                              for k, v in WIRE.items()})})
    r = loader.load(tif)
    assert 0.4 < r.gsd < 0.6, f"expected ~0.5 m/px, got {r.gsd}"


def test_gsd_from_projected_crs_is_untouched(tmp_path):
    arr = np.ones((4, 4), dtype=np.uint8)
    tif = _write_tif(tmp_path / "p.tif", arr, Affine(2.0, 0, 600000, 0, -2.0, 3450000),
                     crs="EPSG:32636", nodata=None)
    assert loader.load(tif).gsd == pytest.approx(2.0)


# --- nodata is not a class -------------------------------------------------

def _nodata_raster():
    """Half nodata, half Rendzina. Nodata value 0 is also Unclassified's id."""
    labels = np.zeros((40, 40), dtype=np.uint8)
    labels[:, 20:] = taxonomy.cid("Rendzina")
    valid = np.zeros((40, 40), dtype=bool)
    valid[:, 20:] = True
    return LabelRaster(labels=labels, gsd=0.5, valid=valid)


def test_nodata_excluded_from_composition():
    text = digests.l0_histogram(_nodata_raster())
    assert "nodata" in text, "the report must say how much of the extent is nodata"
    rows = [l.split("\t") for l in text.splitlines() if not l.startswith(("#", "class_id"))]
    assert len(rows) == 1 and rows[0][1] == "Rendzina"
    assert rows[0][3] == "1.0000", "fractions must be over classified pixels only"


def test_nodata_is_not_a_region():
    ridx = build_regions(_nodata_raster(), min_area_px=4)
    assert len(ridx.regions) == 1
    assert ridx.regions[0].class_name == "Rendzina"
    assert ridx.regions[0].area_px == 800


def test_nodata_excluded_from_chip_fractions():
    cidx = build_chips(_nodata_raster(), size=40)
    ch = cidx.chips[0]
    assert ch.frac(taxonomy.cid("Rendzina")) == pytest.approx(1.0)
    assert ch.frac(taxonomy.cid("Unclassified")) == 0.0


def test_nodata_is_not_trafficable():
    """Unclassified scores 0.5 traffic. Left in, a third of an arid crop reads as
    driveable open ground and corridors bridge straight through it."""
    r = _nodata_raster()
    traf = s4_products.compute(r, "trafficability")
    assert traf[:, :20].max() == 0.0
    assert traf[:, 20:].min() > 0.0

    res = s5_query.query("corridor Rendzina", r, build_regions(r, min_area_px=4))
    assert "100.0% of the classified area" in res.note, res.note


# --- no DEM means abstain, not "slope 0.0" ---------------------------------

def test_audit_abstains_on_slope_without_a_dem():
    """MaralBadlands requires slope 8-90 deg. With no DEM every region reports
    slope 0.0, so this fired on every badlands polygon at severity 0.80 and then
    dominated the review worklist -- a finding about the missing input dressed up
    as a finding about the terrain."""
    labels = np.full((40, 40), taxonomy.cid("MaralBadlands"), dtype=np.uint8)
    labels[:, 30:] = taxonomy.cid("Rendzina")
    r = LabelRaster(labels=labels, gsd=1.0)
    ridx = build_regions(r, min_area_px=4)
    assert not ridx.has_terrain
    kinds = {f.kind for f in audit.audit(ridx)}
    assert "slope-violation" not in kinds
    assert "aspect-incoherent" not in kinds

    r.dem = np.zeros((40, 40), dtype=np.float32)
    with_dem = build_regions(r, min_area_px=4)
    assert with_dem.has_terrain
    assert "slope-violation" in {f.kind for f in audit.audit(with_dem)}


def test_s2_morphology_is_neutral_without_a_dem():
    """Slope 0.0 scores Terrace perfectly and Boulder at zero for every region,
    which is a systematic push towards flat morphologies that reads as evidence."""
    flat, _ = s2_adjudicate._morphology_score(
        None, taxonomy.cid("LimestoneTerrace"), has_terrain=False)
    rough, notes = s2_adjudicate._morphology_score(
        None, taxonomy.cid("LimestoneBoulder"), has_terrain=False)
    assert flat == rough == 0.5
    assert any("no DEM" in n for n in notes)


# --- scale ------------------------------------------------------------------

def test_neighbour_indexing_does_not_rescan_the_array_per_pair():
    """`glob.max()` used to sit inside the per-pair loop, making adjacency
    O(n_pairs * n_pixels). Invisible on a 1024 px fixture with 3 classes; on a
    real 8192 px tile with thousands of regions it never returns.

    The bound is loose on purpose -- this is a complexity regression, not a
    benchmark."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, taxonomy.N_CLASSES, size=(1200, 1200), dtype=np.uint8)
    r = LabelRaster(labels=labels, gsd=1.0)
    t0 = time.perf_counter()
    ridx = build_regions(r, min_area_px=1)
    elapsed = time.perf_counter() - t0
    assert len(ridx.regions) > 10000
    assert elapsed < 120, f"adjacency took {elapsed:.0f}s for {len(ridx.regions)} regions"


def test_quadtree_histogram_is_chunked_and_exact():
    """np.bincount promotes to intp -- 8 bytes/px whatever the input dtype. The
    quadtree's root block on a mosaic is 65536^2, so one naive call there asks
    for 34 GB. Chunking must not change the answer."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, taxonomy.N_CLASSES + 1, size=(300, 700), dtype=np.uint8)
    reference = np.bincount(a.ravel(), minlength=taxonomy.N_CLASSES + 1)
    original = digests._HIST_CHUNK_PX
    try:
        digests._HIST_CHUNK_PX = 997          # force many chunks
        assert (digests._block_hist(a, taxonomy.N_CLASSES + 1) == reference).all()
    finally:
        digests._HIST_CHUNK_PX = original


def test_chip_index_does_not_retain_a_field_per_anchor():
    """Distance fields are float64 at full raster size. Holding all seven at once
    is 56 bytes/px; only one number per chip per anchor is ever read back."""
    r = _nodata_raster()
    cidx = build_chips(r, size=20)
    assert cidx.chips
    for ch in cidx.chips:
        assert set(ch.dist_to) == set(taxonomy.ANCHOR_CLASSES)
        assert all(isinstance(v, float) for v in ch.dist_to.values())


# --- mosaic -----------------------------------------------------------------

def test_mosaic_places_tiles_by_transform_and_crops_to_data(tmp_path):
    """Regions are connected components, so per-tile analysis cuts every region
    at the seam. Placement comes from the affine transform, not the filename."""
    from segmap_digest.mosaic import load_mosaic

    px = 5e-6
    tags = {"ID_TO_LABEL_MAPPING": json.dumps({str(k): v for k, v in WIRE.items()})}
    left = np.zeros((8, 8), dtype=np.uint8)
    left[2:6, 4:] = 112                      # touches its right edge
    right = np.zeros((8, 8), dtype=np.uint8)
    right[2:6, :4] = 112                     # touches its left edge
    _write_tif(tmp_path / "a.tif", left, Affine(px, 0, 34.0, 0, -px, 31.0), tags=tags)
    _write_tif(tmp_path / "b.tif", right,
               Affine(px, 0, 34.0 + 8 * px, 0, -px, 31.0), tags=tags)

    m = load_mosaic(tmp_path)
    # cropped to the data bbox: 4 rows, 8 columns spanning the seam
    assert m.shape == (4, 8)
    assert m.valid is None or m.valid.all()
    ridx = build_regions(m, min_area_px=1)
    assert len(ridx.regions) == 1, "the seam must not split the region in two"
    assert ridx.regions[0].area_px == 32


def test_mosaic_refuses_mixed_resolution(tmp_path):
    from segmap_digest.mosaic import load_mosaic

    a = np.ones((4, 4), dtype=np.uint8)
    _write_tif(tmp_path / "a.tif", a, Affine(5e-6, 0, 34.0, 0, -5e-6, 31.0), nodata=None)
    _write_tif(tmp_path / "b.tif", a, Affine(1e-6, 0, 34.1, 0, -1e-6, 31.0), nodata=None)
    with pytest.raises(ValueError, match="pixel size"):
        load_mosaic(tmp_path)


def test_empty_aoi_directory_is_a_clear_error_not_a_crash(tmp_path):
    """`leb/` exists but its data has not landed. It must fail legibly, and the
    CLI must not answer an operator question with a traceback."""
    from segmap_digest import cli
    from segmap_digest.mosaic import load_mosaic

    (tmp_path / "leb").mkdir()
    with pytest.raises(ValueError, match="no GeoTIFFs"):
        load_mosaic(tmp_path / "leb")

    with pytest.raises(SystemExit, match="no GeoTIFFs"):
        cli.main(["report", "-i", str(tmp_path / "leb"), "-o", str(tmp_path / "out")])
