"""The index cache, and the one honest way to make a big AOI smaller.

Two things here are load-bearing and everything else is detail:

  * A cached index must be *identical* to the one that was built, field for
    field. A cache that returns something almost right is worse than no cache,
    because nothing downstream will notice.
  * A cache must miss whenever anything that changed the answer changed. Every
    invalidation route below corresponds to something an operator actually does:
    re-export a tile, re-crop it, pass a different --min-px, bump the schema.
"""

import json
import os

import numpy as np
import pytest

from segmap_digest import ask, cache, cli, loader, synth
from segmap_digest.index import build_chips, build_regions


@pytest.fixture(scope="module")
def tile():
    return synth.generate(size=384, seed=5)


@pytest.fixture
def npy(tmp_path, tile):
    p = tmp_path / "tile.npy"
    np.save(p, tile.labels)
    return p


def _slot(kind, path, tmp_path, params, **fp):
    return cache.CacheSlot(kind, cache.source_fingerprint(path, **fp), params,
                           tmp_path / "cache")


# --- round-trip ------------------------------------------------------------

def test_region_roundtrip_is_exact(tile, npy, tmp_path):
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    built = build_regions(tile, min_area_px=12)
    cache.store(slot, built, 1.0)
    loaded, meta = cache.load(slot)

    assert len(loaded.regions) == len(built.regions) > 0
    # Region is a dataclass, so == compares every field including `neighbors`.
    assert loaded.regions == built.regions
    assert loaded.has_terrain == built.has_terrain
    assert np.array_equal(np.asarray(loaded.label_array), built.label_array)
    assert meta["n_regions"] == len(built.regions)


def test_chip_roundtrip_is_exact(tile, npy, tmp_path):
    slot = _slot("chips", npy, tmp_path, cache.chip_params(128))
    built = build_chips(tile, size=128)
    cache.store(slot, built, 1.0)
    loaded, _ = cache.load(slot)

    assert (loaded.size, loaded.stride, loaded.gsd) == (built.size, built.stride,
                                                        built.gsd)
    assert len(loaded.chips) == len(built.chips) > 0
    for a, b in zip(built.chips, loaded.chips):
        assert (a.id, a.row, a.col, a.bbox) == (b.id, b.row, b.col, b.bbox)
        assert np.array_equal(a.class_frac, b.class_frac)
        assert (a.entropy, a.n_classes, a.edge_density) == (b.entropy, b.n_classes,
                                                            b.edge_density)
        assert a.dist_to == b.dist_to
        assert a.interfaces == b.interfaces


def test_absent_anchor_stays_infinite(tmp_path):
    """`dist_to` is inf when an anchor class is not in the tile -- a real value,
    and one that JSON would have turned into null or a crash."""
    labels = np.zeros((64, 64), dtype=np.uint8)
    p = tmp_path / "flat.npy"
    np.save(p, labels)
    flat = loader.LabelRaster(labels, gsd=0.5)
    built = build_chips(flat, size=32)
    assert any(np.isinf(v) for v in built.chips[0].dist_to.values())

    slot = _slot("chips", p, tmp_path, cache.chip_params(32))
    cache.store(slot, built, 1.0)
    loaded, _ = cache.load(slot)
    assert loaded.chips[0].dist_to == built.chips[0].dist_to


def test_label_array_is_memory_mapped(tile, npy, tmp_path):
    """The point of keeping it out of the npz: on the mosaic it is 4.8 GB that
    only two callers ever read, so the warm path must not inflate it."""
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)
    loaded, _ = cache.load(slot)
    assert isinstance(loaded.label_array, np.memmap)


# --- invalidation ----------------------------------------------------------

def test_mtime_change_invalidates(tile, npy, tmp_path):
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)
    cache.load(slot)                                   # hits

    os.utime(npy, ns=(0, 0))
    with pytest.raises(cache.CacheMiss):
        cache.load(_slot("regions", npy, tmp_path, cache.region_params(12)))


def test_size_change_invalidates(tile, npy, tmp_path):
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)

    st = npy.stat()
    np.save(npy, np.pad(tile.labels, ((0, 8), (0, 0))))
    os.utime(npy, ns=(st.st_mtime_ns, st.st_mtime_ns))  # same mtime, new size
    with pytest.raises(cache.CacheMiss):
        cache.load(_slot("regions", npy, tmp_path, cache.region_params(12)))


@pytest.mark.parametrize("other", [cache.region_params(50)])
def test_param_change_invalidates(tile, npy, tmp_path, other):
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)
    with pytest.raises(cache.CacheMiss):
        cache.load(_slot("regions", npy, tmp_path, other))


def test_gsd_override_invalidates(tile, npy, tmp_path):
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)
    with pytest.raises(cache.CacheMiss):
        cache.load(_slot("regions", npy, tmp_path, cache.region_params(12), gsd=1.0))


def test_schema_bump_invalidates(tile, npy, tmp_path, monkeypatch):
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)
    monkeypatch.setattr(cache, "SCHEMA_VERSION", cache.SCHEMA_VERSION + 1)
    with pytest.raises(cache.CacheMiss):
        cache.load(_slot("regions", npy, tmp_path, cache.region_params(12)))


def test_a_half_written_entry_is_a_miss(tile, npy, tmp_path):
    """meta.json is written last and the directory is moved into place whole.
    An entry with a plausible meta and no arrays must not read as valid."""
    slot = _slot("regions", npy, tmp_path, cache.region_params(12))
    cache.store(slot, build_regions(tile, min_area_px=12), 1.0)
    (slot.path / "regions.npz").unlink()
    with pytest.raises(cache.CacheMiss):
        cache.load(slot)


# --- get() -----------------------------------------------------------------

def test_get_builds_once_then_reuses(tile, npy, tmp_path, monkeypatch, capsys):
    calls = []
    real = cache.build_regions
    monkeypatch.setattr(cache, "build_regions",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    kw = dict(source=cache.source_fingerprint(npy), params=cache.region_params(12),
              cache_dir=tmp_path / "cache")
    first = cache.get("regions", tile, **kw)
    second = cache.get("regions", tile, **kw)

    assert len(calls) == 1
    assert [r.id for r in second.regions] == [r.id for r in first.regions]
    err = capsys.readouterr().err
    assert "building regions index" in err
    assert "using cached regions index" in err


def test_refresh_rebuilds(tile, npy, tmp_path, monkeypatch):
    calls = []
    real = cache.build_regions
    monkeypatch.setattr(cache, "build_regions",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    kw = dict(source=cache.source_fingerprint(npy), params=cache.region_params(12),
              cache_dir=tmp_path / "cache")
    cache.get("regions", tile, **kw)
    cache.get("regions", tile, refresh=True, **kw)
    assert len(calls) == 2


def test_source_none_never_touches_disk(tile, tmp_path):
    root = tmp_path / "cache"
    cache.get("regions", tile, source=None, params=cache.region_params(12),
              cache_dir=root)
    assert not root.exists()


# --- the stitched mosaic ---------------------------------------------------

def _tile_dir(tmp_path):
    """Two adjacent GeoTIFFs, the smallest thing that is genuinely a mosaic."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    d = tmp_path / "aoi"
    d.mkdir()
    rng = np.random.default_rng(4)
    for i in range(2):
        arr = rng.integers(1, 6, size=(48, 48), dtype=np.uint8)
        arr[:4] = 0                                    # some real nodata
        with rasterio.open(d / f"t{i}.tif", "w", driver="GTiff", height=48,
                           width=48, count=1, dtype="uint8", crs="EPSG:32636",
                           transform=Affine(0.5, 0, 100 + 24 * i, 0, -0.5, 200),
                           nodata=0) as dst:
            dst.write(arr, 1)
    return d


def test_mosaic_roundtrip_is_exact(tmp_path):
    d = _tile_dir(tmp_path)
    src = cache.source_fingerprint(d)
    direct = loader.load(d)
    got = cache.mosaic_raster(d, source=src, cache_dir=tmp_path / "cache", quiet=True)
    again = cache.mosaic_raster(d, source=src, cache_dir=tmp_path / "cache",
                                quiet=True)

    for r in (got, again):
        assert np.array_equal(np.asarray(r.labels), direct.labels)
        assert np.array_equal(np.asarray(r.valid), direct.valid)
        assert r.gsd == pytest.approx(direct.gsd)
        assert str(r.crs) == str(direct.crs)
        assert list(r.transform)[:6] == pytest.approx(list(direct.transform)[:6])
    # The point: the warm one is mapped, not read.
    assert isinstance(again.labels, np.memmap)


def test_mosaic_cache_invalidates_on_a_new_tile(tmp_path):
    d = _tile_dir(tmp_path)
    root = tmp_path / "cache"
    cache.mosaic_raster(d, source=cache.source_fingerprint(d), cache_dir=root,
                        quiet=True)
    first = len(list(root.iterdir()))

    (d / "t1.tif").rename(d / "t2.tif")     # same pixels, different file set
    cache.mosaic_raster(d, source=cache.source_fingerprint(d), cache_dir=root,
                        quiet=True)
    assert len(list(root.iterdir())) == first + 1


def test_mosaic_cache_is_only_for_directories(tmp_path, npy):
    """A single raster re-reads in about a second; caching it would trade that
    for a gigabyte of disk and one more thing to keep valid."""
    root = tmp_path / "cache"
    cli.main(["solve", "s5", "-i", str(npy), "--cache", str(root),
              "--query", "describe"])
    assert not any(p.name.startswith("mosaic-") for p in root.iterdir())


# --- honest subsetting -----------------------------------------------------

def test_crop_keeps_pixels_verbatim(tile):
    """A crop is not a downsample: every pixel it keeps is the pixel that was
    there. This is the whole reason it crops instead of striding."""
    mpx = tile.labels.size / 4e6
    sub = loader.crop_to_max_mpx(tile, mpx)
    h, w = sub.shape
    assert h * w <= mpx * 1e6
    r0, c0 = (tile.shape[0] - h) // 2, (tile.shape[1] - w) // 2
    assert np.array_equal(sub.labels, tile.labels[r0:r0 + h, c0:c0 + w])


def test_crop_states_its_share_of_the_aoi():
    labels = np.zeros((1000, 1000), dtype=np.uint8)
    valid = np.zeros((1000, 1000), dtype=bool)
    valid[400:600, 400:600] = True          # all the data sits in the middle
    r = loader.LabelRaster(labels, gsd=1.0, valid=valid)
    sub = loader.crop_to_max_mpx(r, 0.25)   # 500x500 centred

    assert sub.subset_note.startswith("SUBSET:")
    assert "25.0% of the extent" in sub.subset_note
    # The centre crop happens to contain every classified pixel; the note has to
    # say that, not the extent fraction.
    assert "100.0% of the AOI's classified pixels" in sub.subset_note
    assert sub.n_valid == r.n_valid


def test_crop_is_a_noop_when_it_already_fits(tile):
    assert loader.crop_to_max_mpx(tile, 1000) is tile
    assert tile.subset_note == ""


def test_crop_changes_the_cache_key(npy):
    plain = cache.source_fingerprint(npy)
    cropped = cache.source_fingerprint(npy, subset=loader.crop_spec(4.0))
    assert cache.cache_key("regions", plain, {}) != \
        cache.cache_key("regions", cropped, {})
    assert cache.cache_key("regions", cropped, {}) == \
        cache.cache_key("regions", cache.source_fingerprint(
            npy, subset=loader.crop_spec(4.0)), {})


def test_crop_rejects_a_meaningless_budget(tile):
    with pytest.raises(ValueError):
        loader.crop_to_max_mpx(tile, 0)


# --- through the CLI -------------------------------------------------------

def test_cli_index_then_reuse(npy, tmp_path, capsys):
    root = str(tmp_path / "cache")
    cli.main(["index", "-i", str(npy), "-o", root])
    built = capsys.readouterr()
    assert "building regions index" in built.err
    assert "building chips index" in built.err

    cli.main(["solve", "s5", "-i", str(npy), "--cache", root, "--query", "describe"])
    warm = capsys.readouterr()
    assert "using cached regions index" in warm.err
    assert "describe" in warm.out

    listing = cache.entries(root)
    assert {e["kind"] for e in listing} == {"regions", "chips"}
    assert all(json.loads((tmp_path / "cache" / f"{e['kind']}-{e['key']}"
                           / "meta.json").read_text())["schema"]
               == cache.SCHEMA_VERSION for e in listing)


def test_cli_no_cache_writes_nothing(npy, tmp_path, capsys):
    root = tmp_path / "cache"
    cli.main(["solve", "s5", "-i", str(npy), "--cache", str(root), "--no-cache",
              "--query", "describe"])
    assert not root.exists()
    assert "using cached" not in capsys.readouterr().err


def test_cli_max_mpx_announces_the_subset(npy, tmp_path, capsys):
    cli.main(["solve", "s5", "-i", str(npy), "--cache", str(tmp_path / "c"),
              "--max-mpx", "0.05", "--query", "describe"])
    err = capsys.readouterr().err
    assert "SUBSET:" in err
    assert "must not be reported as an AOI-wide figure" in err


# --- the live path ---------------------------------------------------------

@pytest.mark.skipif(os.environ.get("ANTHROPIC_API_KEY"),
                    reason="credentials present; this asserts the no-key failure mode")
def test_ask_without_credentials_says_what_to_do():
    """Without this the SDK raises a TypeError about HTTP headers, which reads as
    a bug in this repo rather than a missing export."""
    pytest.importorskip("anthropic")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        ask.client()
