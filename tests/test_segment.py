"""The imagery step: `segmap segment`.

Nothing here loads the real half-gigabyte weights except the one test at the
bottom, which skips when they are absent. Everything above drives a stub session
instead, because what can actually go wrong in this file is not the network --
it is the bookkeeping around it:

  * a tile grid that leaves a seam, an overlap, or a strip of the raster unwritten
  * an interior extracted at the wrong offset, so the map is shifted by a halo
  * the producer's normalisation applied with the bands the wrong way round
  * an output that the rest of the repo then refuses to read

A stub whose "prediction" is a known function of its input catches all four; a
real forward pass catches none of them, because a plausible map of the wrong
ground looks exactly like a plausible map.
"""

import json

import numpy as np
import pytest

from segmap_digest import loader, segment, taxonomy

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine                       # noqa: E402


# --- a session that predicts what it was shown ------------------------------

class _Name:
    def __init__(self, name):
        self.name = name


class EchoSession:
    """Returns the R band of its input, un-normalised, as the class map.

    So `segment_array(rgb)` must reproduce `rgb[..., 0]` exactly -- which it can
    only do if the padding, the tiling and the interior offsets all agree.
    """

    def __init__(self):
        self.calls = 0
        self.shapes = []

    def get_inputs(self):
        return [_Name("input")]

    def get_outputs(self):
        return [_Name("preds")]

    def run(self, names, feed):
        x = feed["input"]
        self.calls += 1
        self.shapes.append(x.shape[-2:])
        r = x[0, 0] * segment.RGB_STD[0] + segment.RGB_MEAN[0]
        return [np.rint(r).astype(np.int64)[None]]


class ConstSession(EchoSession):
    def __init__(self, value=7):
        super().__init__()
        self.value = value

    def run(self, names, feed):
        x = feed["input"]
        self.calls += 1
        return [np.full((1,) + x.shape[-2:], self.value, dtype=np.int64)]


def _ramp(h, w, seed=3):
    rng = np.random.default_rng(seed)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = rng.integers(0, 200, size=(h, w), dtype=np.uint8)
    rgb[..., 1] = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    rgb[..., 2] = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    return rgb


# --- the tile grid ----------------------------------------------------------

@pytest.mark.parametrize("h,w", [(1024, 1024), (2000, 3000), (100, 100),
                                 (1025, 1023), (4096, 512)])
def test_plan_covers_every_pixel_exactly_once(h, w):
    """A gap is a strip of unwritten nodata down the middle of the map; an
    overlap is two answers for the same ground. Neither is visible in a preview
    of a 200 Mpx raster, so it is asserted here."""
    tiles, side = segment.plan(h, w, tile=1024, halo=128)
    hits = np.zeros((h, w), dtype=np.int32)
    for t in tiles:
        hits[t.row:t.row + t.rows, t.col:t.col + t.cols] += 1
    assert hits.min() == 1 and hits.max() == 1
    assert side == 1024 + 2 * 128


def test_plan_refuses_a_model_input_that_is_not_a_multiple_of_32():
    with pytest.raises(segment.SegmentError, match="multiple of 32"):
        segment.plan(500, 500, tile=1000, halo=10)


def test_plan_reads_a_halo_of_real_pixels_where_there_are_some():
    """The middle tile of a 3x3 grid must be read with context on all four
    sides; the top-left tile can only have context below and right."""
    tiles, side = segment.plan(3072, 3072, tile=1024, halo=128)
    mid = next(t for t in tiles if (t.row, t.col) == (1024, 1024))
    assert (mid.read_row, mid.read_col) == (896, 896)
    assert (mid.read_rows, mid.read_cols) == (side, side)
    assert (mid.top, mid.left) == (0, 0)
    tl = next(t for t in tiles if (t.row, t.col) == (0, 0))
    assert (tl.top, tl.left) == (128, 128)


def test_pad_to_replicates_the_edge_rather_than_inventing_black():
    a = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    out = segment.pad_to(a, 7, top=2, left=2)
    assert out.shape == (7, 7)
    assert np.array_equal(out[2:5, 2:5], a)          # the real pixels, in place
    assert np.array_equal(out[0], out[2])            # rows above replicate row 0
    assert np.array_equal(out[1], out[2])
    assert np.array_equal(out[6], out[4])            # and below, the last row
    assert np.array_equal(out[:, 0], out[:, 2])      # same on the left
    assert (out > 0).all()                           # nothing was zero-filled


# --- alignment --------------------------------------------------------------

@pytest.mark.parametrize("h,w,tile,halo", [(1024, 1024, 1024, 128),
                                           (1500, 1200, 512, 32),
                                           (300, 700, 256, 32),
                                           (64, 64, 512, 128)])
def test_segment_array_is_pixel_aligned(h, w, tile, halo):
    """The stub returns its input's R band, so the output must BE the R band.
    An interior taken at the wrong offset shifts the whole map by up to a halo
    and still produces a perfectly plausible-looking result."""
    rgb = _ramp(h, w)
    out = segment.segment_array(rgb, EchoSession(), tile=tile, halo=halo)
    assert out.shape == (h, w)
    assert np.array_equal(out, rgb[..., 0])


def test_every_tile_is_fed_the_same_shape():
    """One input shape for the whole grid, edge tiles included -- so onnxruntime
    plans memory once rather than per raster edge."""
    sess = EchoSession()
    segment.segment_array(_ramp(1500, 1200), sess, tile=512, halo=32)
    assert set(sess.shapes) == {(576, 576)}
    assert sess.calls == 3 * 3


# --- the producer's preprocessing -------------------------------------------

def test_preprocess_matches_the_producers_formula_band_for_band():
    """Mean/std in R,G,B order, HWC -> CHW, batch axis. Swapping the bands is a
    silent accuracy loss, not a crash, so the numbers are pinned."""
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    x = segment.preprocess(rgb)
    assert x.shape == (1, 3, 1, 1) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((10 - 123.675) / 58.395)
    assert x[0, 1, 0, 0] == pytest.approx((20 - 116.28) / 57.12)
    assert x[0, 2, 0, 0] == pytest.approx((30 - 103.53) / 57.375)


def test_preprocess_refuses_a_non_rgb_array():
    with pytest.raises(segment.SegmentError, match="RGB"):
        segment.preprocess(np.zeros((4, 4), dtype=np.uint8))


def test_infer_refuses_a_wire_id_that_does_not_fit_uint8():
    class Big(EchoSession):
        def run(self, names, feed):
            return [np.full((1, 4, 4), 300, dtype=np.int64)]

    with pytest.raises(segment.SegmentError, match="does not fit"):
        segment.infer(Big(), np.zeros((4, 4, 3), dtype=np.uint8))


def test_infer_takes_no_argmax():
    """The model already chose. If this ever starts arg-maxing a class map it
    will return zeros everywhere and look like a nodata bug."""
    sess = ConstSession(value=132)
    out = segment.infer(sess, np.zeros((32, 32, 3), dtype=np.uint8))
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) == {132}


# --- choosing the model -----------------------------------------------------

@pytest.fixture
def weights(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    for name in segment.MODELS.values():
        (d / name).write_bytes(b"not really onnx")
    monkeypatch.setenv("SEGMAP_ST_MODELS", str(d))
    return d


def test_closest_native_resolution_wins(weights):
    assert segment.pick_model(0.13)[1] == 0.125
    assert segment.pick_model(0.125)[1] == 0.125
    assert segment.pick_model(0.45)[1] == 0.5


def test_model_is_chosen_by_ratio_not_by_absolute_metres(weights):
    """0.3 m is 0.175 above 0.125 and 0.2 below 0.5, so absolute distance picks
    the 12.5 cm model -- at 2.4x its native scale. By ratio the 50 cm model is
    the less wrong one, and it is still far enough off to be refused."""
    with pytest.raises(segment.SegmentError, match="wrong scale"):
        segment.pick_model(0.3)
    path, native, note = segment.pick_model(0.3, allow_mismatch=True)
    assert native == 0.5


def test_a_mismatch_inside_the_refusal_band_is_stated_not_swallowed(weights):
    _, native, note = segment.pick_model(0.14)
    assert native == 0.125
    assert "WARNING" in note and "12%" in note


def test_explicit_model_gsd_overrides_the_header(weights):
    path, native, _ = segment.pick_model(0.125, native_gsd=0.5,
                                         allow_mismatch=True)
    assert native == 0.5 and path.name == segment.MODELS[0.5]


def test_an_unrecognised_model_filename_must_declare_its_resolution(weights):
    other = weights / "something_else.onnx"
    other.write_bytes(b"x")
    with pytest.raises(segment.SegmentError, match="--model-gsd"):
        segment.pick_model(0.125, model=other)
    _, native, _ = segment.pick_model(0.125, model=other, native_gsd=0.125)
    assert native == 0.125


def test_the_old_15cm_filename_still_resolves(tmp_path, monkeypatch):
    """It is the 12.5 cm model under a wrong name. Renaming ours must not break
    a models/ directory somebody else already populated."""
    d = tmp_path / "models"
    d.mkdir()
    (d / "ST_15cm_model.onnx").write_bytes(b"x")
    monkeypatch.setenv("SEGMAP_ST_MODELS", str(d))
    assert set(segment.available()) == {0.125}
    path, native, _ = segment.pick_model(0.125)
    assert native == 0.125 and path.name == "ST_15cm_model.onnx"
    # and named explicitly, it still declares its own resolution
    assert segment.pick_model(0.125, model=path)[1] == 0.125


def test_the_canonical_name_wins_over_the_legacy_one(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    (d / "ST_15cm_model.onnx").write_bytes(b"x")
    (d / "ST_12_5cm_model.onnx").write_bytes(b"x")
    monkeypatch.setenv("SEGMAP_ST_MODELS", str(d))
    assert segment.available()[0.125].name == "ST_12_5cm_model.onnx"


def test_missing_weights_say_where_they_were_looked_for(tmp_path, monkeypatch):
    monkeypatch.setenv("SEGMAP_ST_MODELS", str(tmp_path / "nope"))
    with pytest.raises(segment.SegmentError, match="no Smart Terrain weights"):
        segment.pick_model(0.125)


# --- the written file, and whether the rest of the repo can read it ---------

def _ortho(path, rgb, gsd=0.125, crs="EPSG:3857", nodata=None):
    h, w = rgb.shape[:2]
    tr = Affine(gsd, 0, 3921525.0, 0, -gsd, 3910136.0)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=3,
                       dtype="uint8", crs=crs, transform=tr, nodata=nodata) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    return path


def _wire_rgb(h, w, wire_ids):
    """RGB whose R band is a tile of real wire ids, so EchoSession's output is a
    label raster the loader should accept."""
    ids = np.array(sorted(wire_ids), dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = ids[np.arange(h * w).reshape(h, w) % len(ids)]
    return rgb


def test_a_supplied_session_needs_no_weights_on_this_machine(tmp_path, monkeypatch):
    """Every test above this line would otherwise pass only on a checkout that
    happens to have half a gigabyte of ONNX in ./models."""
    monkeypatch.setenv("SEGMAP_ST_MODELS", str(tmp_path / "definitely-not-here"))
    src = _ortho(tmp_path / "ortho.tif", _wire_rgb(64, 64, [132]))
    r = segment.segment_geotiff(src, tmp_path / "labels.tif",
                                sess=ConstSession(132), tile=64, halo=32,
                                native_gsd=0.125)
    assert r.counts == {132: 64 * 64}
    assert "caller-supplied session" in r.note


def test_a_supplied_session_is_still_scale_checked(tmp_path, monkeypatch):
    """The session cannot be interrogated, so --model-gsd is what the check has
    to go on -- and it still has to refuse."""
    monkeypatch.setenv("SEGMAP_ST_MODELS", str(tmp_path / "nope"))
    src = _ortho(tmp_path / "ortho.tif", _wire_rgb(64, 64, [132]), gsd=0.125)
    with pytest.raises(segment.SegmentError, match="wrong scale"):
        segment.segment_geotiff(src, tmp_path / "labels.tif",
                                sess=ConstSession(132), tile=64, halo=32,
                                native_gsd=0.5)


def test_output_is_readable_by_loader_with_no_arguments(tmp_path):
    """The point of the whole command: `segmap report -i <this>` must work with
    no --classes and no --gsd, because the file carries both."""
    rgb = _wire_rgb(300, 260, [2, 10, 112, 132, 241])
    src = _ortho(tmp_path / "ortho.tif", rgb)
    r = segment.segment_geotiff(src, tmp_path / "labels.tif", sess=EchoSession(),
                                tile=256, halo=32, native_gsd=0.125)
    assert r.shape == (300, 260)
    assert r.unmapped == []

    raster = loader.load(tmp_path / "labels.tif")
    assert raster.labels.shape == (300, 260)
    assert raster.gsd == pytest.approx(0.125, rel=1e-3)
    # wire -> dense taxonomy ids, by name, exactly as a product tile is read
    assert taxonomy.cid("Rendzina") in np.unique(raster.labels)
    assert taxonomy.cid("ChalkTerrace") in np.unique(raster.labels)


def test_output_carries_its_provenance(tmp_path):
    rgb = _wire_rgb(64, 64, [2, 132])
    src = _ortho(tmp_path / "ortho.tif", rgb)
    segment.segment_geotiff(src, tmp_path / "labels.tif", sess=EchoSession(),
                            tile=256, halo=32, native_gsd=0.125,
                            model=None)
    with rasterio.open(tmp_path / "labels.tif") as ds:
        tags = ds.tags()
        assert ds.count == 1 and ds.dtypes == ("uint8",) and ds.nodata == 0
        assert json.loads(tags["ID_TO_LABEL_MAPPING"])["112"] == "Rendzina"
        assert tags["ST_MODEL_NATIVE_GSD"] == "0.125"
        assert tags["ST_SOURCE"] == "ortho.tif"
        assert tags["ST_TILE"] == "256" and tags["ST_HALO"] == "32"


def test_the_written_labels_are_the_predictions(tmp_path):
    rgb = _wire_rgb(500, 300, [2, 10, 112, 132])
    src = _ortho(tmp_path / "ortho.tif", rgb)
    segment.segment_geotiff(src, tmp_path / "labels.tif", sess=EchoSession(),
                            tile=256, halo=32, native_gsd=0.125)
    with rasterio.open(tmp_path / "labels.tif") as ds:
        assert np.array_equal(ds.read(1), rgb[..., 0])


def test_source_nodata_is_not_filled_with_a_hallucinated_class(tmp_path):
    """A black border is a feature to a segmenter -- it predicts shadow and water
    along it. Where the source says nodata, the output says nodata."""
    rgb = _wire_rgb(128, 128, [132])
    rgb[:16, :, :] = 0                      # a nodata stripe, declared below
    src = _ortho(tmp_path / "ortho.tif", rgb, nodata=0)
    r = segment.segment_geotiff(src, tmp_path / "labels.tif", sess=ConstSession(132),
                                tile=64, halo=32, native_gsd=0.125)
    with rasterio.open(tmp_path / "labels.tif") as ds:
        out = ds.read(1)
    assert (out[:16] == 0).all()
    assert (out[16:] == 132).all()
    assert r.nodata_px == 16 * 128


def test_max_mpx_segments_a_stated_window_and_georeferences_it(tmp_path):
    rgb = _wire_rgb(600, 600, [132])
    src = _ortho(tmp_path / "ortho.tif", rgb)
    r = segment.segment_geotiff(src, tmp_path / "labels.tif", sess=ConstSession(132),
                                tile=128, halo=32, native_gsd=0.125, max_mpx=0.09)
    assert r.shape == (300, 300)
    assert "SUBSET" in r.subset_note
    with rasterio.open(src) as a, rasterio.open(tmp_path / "labels.tif") as b:
        assert b.tags()["ST_SUBSET"].startswith("SUBSET")
        # the window's own upper-left, not the ortho's
        assert b.transform.c == pytest.approx(a.transform.c + 150 * 0.125)
        assert b.transform.f == pytest.approx(a.transform.f - 150 * 0.125)


def test_a_single_band_input_is_refused_with_the_reason(tmp_path):
    arr = np.zeros((32, 32), dtype=np.uint8)
    path = tmp_path / "labels_in.tif"
    with rasterio.open(path, "w", driver="GTiff", height=32, width=32, count=1,
                       dtype="uint8", crs="EPSG:3857",
                       transform=Affine(0.125, 0, 0, 0, -0.125, 0)) as dst:
        dst.write(arr, 1)
    with pytest.raises(segment.SegmentError, match="already a label raster"):
        segment.segment_geotiff(path, tmp_path / "out.tif", sess=EchoSession(),
                                native_gsd=0.125)


def test_an_unusable_id_mapping_fails_before_the_hour_of_inference(tmp_path):
    """`wire_lut` refuses names the taxonomy does not define. Finding that out
    after the forward passes rather than before is the difference between a
    typo and a wasted afternoon."""
    bad = tmp_path / "classes.json"
    bad.write_text(json.dumps({"0": "Unclassified", "9": "NotATerrainClass"}))
    src = _ortho(tmp_path / "ortho.tif", _wire_rgb(64, 64, [0]))
    sess = ConstSession(0)
    with pytest.raises(ValueError, match="not in taxonomy"):
        segment.segment_geotiff(src, tmp_path / "out.tif", sess=sess,
                                classes=bad, native_gsd=0.125, tile=64, halo=32)
    assert sess.calls == 0


def test_summary_names_the_classes_it_found(tmp_path):
    rgb = _wire_rgb(128, 128, [132])
    src = _ortho(tmp_path / "ortho.tif", rgb)
    r = segment.segment_geotiff(src, tmp_path / "labels.tif",
                                sess=ConstSession(132), tile=64, halo=32,
                                native_gsd=0.125)
    text = r.summary(loader.class_map(segment.CLASSES_JSON))
    assert "DryGrassland" in text and "100.0%" in text


# --- the real weights, when they are here -----------------------------------

@pytest.mark.slow
def test_real_model_emits_only_ids_the_mapping_names():
    """The one test that runs the network. Its job is not accuracy -- it is that
    every id the model can emit has a name in the bundled mapping, because an
    unnamed id makes the output unreadable by every other command."""
    pytest.importorskip("onnxruntime")
    have = segment.available()
    if 0.125 not in have:
        pytest.skip(f"no 12.5 cm weights in {segment.model_dir()}")
    import glob
    orthos = sorted(glob.glob("data/raw/*.tif")) + sorted(
        glob.glob("data/raw/*.tiff"))
    if not orthos:
        pytest.skip("no RGB ortho in data/raw/ (gitignored real imagery)")

    with rasterio.open(orthos[0]) as src:
        rgb = src.read((1, 2, 3), window=rasterio.windows.Window(
            src.width // 2, src.height // 2, 320, 320)).transpose(1, 2, 0)
    sess = segment.session(have[0.125])
    wire = segment.infer(sess, rgb)
    mapping = loader.class_map(segment.CLASSES_JSON)
    present = sorted(int(i) for i in np.unique(wire))
    assert present, "the model returned nothing"
    assert [i for i in present if i not in mapping] == []
    # and the whole point: those ids survive the trip into taxonomy space
    assert loader.remap(wire, mapping).max() < taxonomy.N_CLASSES
