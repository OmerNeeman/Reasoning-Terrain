import numpy as np
import pytest

from segmap_digest import audit, digests, loader, synth, taxonomy
from segmap_digest.index import build_chips, build_regions


@pytest.fixture(scope="module")
def tile():
    return synth.generate(size=256, seed=3)


def test_taxonomy_shape():
    # 47, not 45: the real export's ID_TO_LABEL_MAPPING carries
    # LimestoneHardRockLineament and ChalkTerrace, which the first draft of this
    # taxonomy omitted. Asserted against N_CLASSES rather than repeating a
    # literal, so dense ids stay dense whatever the count becomes.
    n = taxonomy.N_CLASSES
    assert n == 47
    assert len({c.name for c in taxonomy.CLASSES}) == n
    assert [c.id for c in taxonomy.CLASSES] == list(range(n))
    # every class has exactly one superclass
    assert len(taxonomy.SUPERCLASS_OF) == n


def test_lithology_grid_is_asymmetric():
    """The grid is sparse on purpose: chalk has two morphologies, limestone seven.
    A symmetric grid would mean the taxonomy had been flattened by mistake."""
    grid = taxonomy.LITHOLOGY_GRID
    assert set(grid["Chalk"]) == {"SmoothRockSlopes", "Terrace"}
    assert len(grid["Limestone"]) == 7
    assert len(grid["Basalt"]) == 3
    assert len({len(v) for v in grid.values()}) > 1


def test_class_distance_respects_ordinal_series():
    d = taxonomy.class_distance
    garigue, batha = taxonomy.cid("Garigue"), taxonomy.cid("Batha")
    house = taxonomy.cid("House")
    assert d(garigue, garigue) == 0.0
    assert d(garigue, batha) < d(garigue, house)
    # same morphology, different lithology -- the RGB-blind case
    assert 0 < d(taxonomy.cid("LimestoneBoulder"), taxonomy.cid("DolomiteBoulder")) < 1


def test_synth_is_deterministic():
    a = synth.generate(size=128, seed=11).labels
    b = synth.generate(size=128, seed=11).labels
    assert np.array_equal(a, b)
    assert not np.array_equal(a, synth.generate(size=128, seed=12).labels)


def test_synth_slope_is_plausible(tile):
    gy, gx = np.gradient(tile.dem.astype(float), tile.gsd)
    med = np.degrees(np.arctan(np.median(np.hypot(gx, gy))))
    assert 2 < med < 20, f"median slope {med:.1f} deg is not terrain"


def test_regions_and_attributes(tile):
    ridx = build_regions(tile)
    assert ridx.regions
    assert all(r.area_m2 > 0 and r.perimeter_m > 0 for r in ridx.regions)
    assert all(0 <= r.compactness <= 1.3 for r in ridx.regions)
    assert all(0 <= r.aspect_circvar <= 1.001 for r in ridx.regions)
    # region ids index the list
    assert all(ridx.get(r.id) is r for r in ridx.regions)
    # adjacency is symmetric
    for r in ridx.regions:
        for nid in r.neighbors:
            assert r.id in ridx.get(nid).neighbors


def test_chip_fractions_sum_to_one(tile):
    cidx = build_chips(tile, size=128)
    assert cidx.chips
    for ch in cidx.chips:
        assert ch.class_frac.shape == (taxonomy.N_CLASSES,)
        assert abs(ch.class_frac.sum() - 1.0) < 1e-9
        assert ch.entropy >= 0


def test_digest_ladder_is_monotone_in_size(tile):
    ridx = build_regions(tile)
    l0 = digests.l0_histogram(tile)
    l1 = digests.l1_grid(tile, n=8)
    l2 = digests.l2_regions(ridx)
    assert len(l0) < len(l1) < len(l2)
    # raw raster as text would be orders of magnitude bigger -- that's the point
    assert len(l2) < tile.labels.size * 3


def test_quadtree_roundtrip_is_parseable(tile):
    text = digests.l1q_quadtree(tile)
    body = text.split("\n", 1)[1]
    assert body.count("(") == body.count(")")
    for tok in body.replace("(", " ").replace(")", " ").split():
        assert tok == "." or 0 <= int(tok) < taxonomy.N_CLASSES


def test_limit_is_reported_not_silent(tile):
    ridx = build_regions(tile)
    text = digests.l2_regions(ridx, limit=3)
    assert "omitted by limit" in text, "a cap must never read as 'this is everything'"


def test_audit_returns_candidates(tile):
    ridx = build_regions(tile)
    findings = audit.audit(ridx)
    assert all(0 <= f.severity <= 1 for f in findings)
    assert "CANDIDATES" in audit.to_tsv(findings)


def test_loader_rejects_out_of_range(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.full((8, 8), 99, dtype=np.uint8))
    with pytest.raises(ValueError, match=f"0..{taxonomy.N_CLASSES - 1}"):
        loader.load(bad)


def test_loader_rejects_colourised_png(tmp_path):
    from PIL import Image

    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(rgb)
    with pytest.raises(ValueError, match="single-band"):
        loader.load(rgb)


def test_npy_roundtrip(tmp_path, tile):
    p = tmp_path / "t.npy"
    np.save(p, tile.labels)
    assert np.array_equal(loader.load(p, gsd=0.3).labels, tile.labels)
