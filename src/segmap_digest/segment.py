"""Run the Smart Terrain segmenter over RGB imagery, and write the label raster
the rest of this repo digests.

Everything else in `segmap` starts from a *label* raster -- someone else ran the
model and handed us integers. This is the step before that, and it exists because
"the segmenter said this" and "the digest says that" are only the same claim if
the same code can produce both.

Three things about the models that shape this file:

  **They emit one class index per pixel, not a class-probability stack.** The
    `preds` output is int64 `(1, H, W)` and the integer is a *wire id* -- the
    sparse 0..241 product ids of `data/smart_terrain_class_ids.json`, not the
    dense taxonomy ids used inside this repo. So there is no argmax to take and
    no channel bookkeeping to get wrong: `preds` goes to disk verbatim, and
    `loader` does the wire -> taxonomy translation exactly as it does for a tile
    that arrived from the product. Two other outputs come along for free in the
    same forward pass -- `roads` (binary) and `dsem_landcover` (one float band)
    -- and nothing in RT consumes either, so they are dropped rather than
    written out to rot.

  **One model per ground resolution.** 12.5 cm and 50 cm are separate networks,
    and running the wrong one is a scale error the output cannot reveal: it
    reads as a plausible map of terrain at the wrong size. So the model is
    chosen from the raster's own metres-per-pixel and refused outright when the
    two are far apart (`--allow-gsd-mismatch` to insist).

  **The input is dynamic in H and W but the network is not scale-free.** Tiles
    are fed at one fixed size, edge-padded at the raster border, and only the
    interior of each prediction is kept. A convolutional segmenter's output is
    worst where its receptive field runs off the edge of its input, which is
    every tile seam -- keeping a `--halo` of padding and throwing it away is
    what stops the tile grid from printing itself into the map.

A note on metres. Pixel size comes from `loader.geotiff_gsd`, so this command
agrees with every other command in the repo about how big a pixel is. For a
web-mercator (EPSG:3857) raster that is a pseudo-metre and is smaller than a
true ground metre by cos(latitude) -- the whole repo takes the header at face
value here, and this command does not invent a different convention for itself.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The producer's own preprocessing, copied from `MaterialSegmentation` in
# MaterialSegmentationPackage: ImageNet mean/std in R,G,B band order, then
# HWC -> CHW, then a batch axis. Getting the band order backwards here is a
# silent 20-point accuracy loss, not a crash, so it is stated once and reused.
RGB_MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)
RGB_STD = np.array((58.395, 57.12, 57.375), dtype=np.float32)

# native metres/pixel -> the model trained at it.
MODELS: dict[float, str] = {0.125: "ST_12_5cm_model.onnx", 0.5: "ST_50cm_model.onnx"}

# The 12.5 cm weights arrived named `ST_15cm_model.onnx`, which they are not:
# that file is byte-identical (md5 88ba82f8...) to the product's
# `Merged_2025-02-13...` weights, and the product's own registry calls those
# `RESOLUTION_12_5`. The name was a trap for the next person -- 15 cm imagery
# would have been fed to it as if native -- so the canonical name says 12.5 and
# the old name is still recognised, because someone else's models/ directory
# should not break on our rename.
LEGACY_NAMES: dict[str, float] = {"ST_15cm_model.onnx": 0.125}

DEFAULT_TILE = 1024
DEFAULT_HALO = 128
# The network downsamples by 32; a padded tile that is not a multiple of it
# either fails in the graph or gets silently rounded. Checked, not assumed.
SIZE_MULTIPLE = 32

# How far the raster's metres-per-pixel may sit from the model's native
# resolution. Under WARN, silence; between WARN and REFUSE, a stated warning;
# above REFUSE, a refusal that names the escape hatch.
GSD_WARN = 0.05
GSD_REFUSE = 0.25

OUTPUTS = ("preds",)

# The model's id space, as a file, so the written raster carries the same
# mapping the playground and the loader already know how to read.
CLASSES_JSON: Path = importlib.resources.files("segmap_digest.data").joinpath(
    "smart_terrain_class_ids.json")


class SegmentError(RuntimeError):
    """Something the operator can act on: a missing model, a resolution
    mismatch, a raster that is not imagery."""


# --- where the weights live -------------------------------------------------
#
# Half a gigabyte each, gitignored, and not shipped in the wheel. So: an env var
# for anyone who keeps them elsewhere, and the repo's own `models/` otherwise.

def model_dir() -> Path:
    env = os.environ.get("SEGMAP_ST_MODELS")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "models"


def available() -> dict[float, Path]:
    """The native resolutions we actually have weights for, on this machine.

    The canonical name wins where both it and a legacy name are present, so a
    directory mid-rename does not resolve to whichever one os.listdir felt like.
    """
    d = model_dir()
    have = {gsd: d / name for gsd, name in MODELS.items() if (d / name).is_file()}
    for name, gsd in LEGACY_NAMES.items():
        if gsd not in have and (d / name).is_file():
            have[gsd] = d / name
    return have


def _native_from_name(path: Path) -> float | None:
    for gsd, name in MODELS.items():
        if path.name == name:
            return gsd
    return LEGACY_NAMES.get(path.name)


def pick_model(gsd: float, model: str | Path | None = None,
               native_gsd: float | None = None,
               allow_mismatch: bool = False) -> tuple[Path, float, str]:
    """(weights, the resolution they were trained at, a note for the operator).

    `gsd` is the raster's metres-per-pixel. With no `--model`/`--model-gsd` the
    closest native resolution wins, by ratio rather than by difference: 0.3 m is
    2.4x the 12.5 cm model and 0.6x the 50 cm one, and the second is the less
    wrong of the two even though 0.175 < 0.3 in absolute metres.
    """
    if model is not None:
        path = Path(model)
        if not path.is_file():
            raise SegmentError(f"no such model file: {path}")
        native = native_gsd if native_gsd is not None else _native_from_name(path)
        if native is None:
            raise SegmentError(
                f"cannot tell what resolution {path.name} was trained at. Pass "
                f"--model-gsd (one of {', '.join(f'{g:g}' for g in MODELS)}) so "
                f"the scale check can run instead of being skipped.")
        note = f"model: {path.name} (given), native {native:g} m/px"
    else:
        have = available()
        if not have:
            raise SegmentError(
                f"no Smart Terrain weights in {model_dir()}. Expected one of "
                f"{', '.join(MODELS.values())}, or set $SEGMAP_ST_MODELS to the "
                f"directory that holds them, or pass --model.")
        if native_gsd is not None:
            if native_gsd not in have:
                raise SegmentError(
                    f"--model-gsd {native_gsd:g} has no weights in {model_dir()}; "
                    f"have {', '.join(f'{g:g}' for g in sorted(have))}")
            native = native_gsd
        else:
            native = min(have, key=lambda g: abs(np.log(gsd / g)))
        path = have[native]
        note = f"model: {path.name}, native {native:g} m/px"

    return path, native, note + check_scale(gsd, native, allow_mismatch)


def check_scale(gsd: float, native: float, allow_mismatch: bool = False) -> str:
    """Empty when the imagery is at the scale the network expects, a stated
    warning when it is near the edge of it, and a refusal when it is past it.

    Separate from `pick_model` because it is the half that still has to run when
    the caller brought its own session and there is no file to inspect.
    """
    off = gsd / native - 1.0
    if abs(off) > GSD_REFUSE and not allow_mismatch:
        raise SegmentError(
            f"this raster is {gsd:.4g} m/px and the closest model was trained at "
            f"{native:g} m/px ({off:+.0%}). A segmenter run at the wrong scale "
            f"returns a plausible-looking map of terrain that is not there, and "
            f"nothing downstream can detect it. Resample the imagery to "
            f"{native:g} m/px, or pass --allow-gsd-mismatch to run it anyway and "
            f"own the result.")
    if abs(off) > GSD_WARN:
        return (f" -- WARNING: raster is {gsd:.4g} m/px, {off:+.0%} off native. "
                f"Every class boundary is scaled by that much.")
    return ""


# --- the forward pass -------------------------------------------------------

def session(model: str | Path, gpu: bool = False, threads: int | None = None):
    """An onnxruntime session, or a message saying how to get one.

    CUDA is asked for and not required: an onnxruntime built without it, or a
    box without the CUDA libraries, falls back to CPU with a stated warning
    rather than failing a 40-minute job at tile zero.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:                       # pragma: no cover - env
        raise SegmentError(
            "`segmap segment` needs onnxruntime: pip install 'reasoning-terrain[st]' "
            "(or pip install onnxruntime). Every other command runs without it."
        ) from exc

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
    want = ["CPUExecutionProvider"]
    if gpu:
        have = ort.get_available_providers()
        if "CUDAExecutionProvider" in have:
            want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("# --gpu: this onnxruntime has no CUDAExecutionProvider "
                  f"(providers: {', '.join(have)}); running on CPU",
                  file=sys.stderr)
    sess = ort.InferenceSession(str(model), sess_options=opts, providers=want)
    names = {o.name for o in sess.get_outputs()}
    missing = [n for n in OUTPUTS if n not in names]
    if missing:
        raise SegmentError(
            f"{Path(model).name} has no {', '.join(missing)} output (it has "
            f"{', '.join(sorted(names))}). This command wants the product model "
            f"that emits per-pixel class ids.")
    return sess


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32, the producer's normalisation."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SegmentError(f"expected an (H, W, 3) RGB array, got {rgb.shape}")
    x = (rgb.astype(np.float32) - RGB_MEAN) / RGB_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def infer(sess, rgb: np.ndarray) -> np.ndarray:
    """One tile of RGB -> (H, W) wire ids. No argmax: the model already chose."""
    (preds,) = sess.run(list(OUTPUTS), {sess.get_inputs()[0].name: preprocess(rgb)})
    preds = np.asarray(preds)
    preds = preds.reshape(preds.shape[-2:]) if preds.ndim > 2 else preds
    if preds.shape != rgb.shape[:2]:
        raise SegmentError(
            f"model returned a {preds.shape} class map for a {rgb.shape[:2]} tile")
    hi = int(preds.max(initial=0))
    if hi > 255:
        raise SegmentError(
            f"model emitted wire id {hi}, which does not fit the uint8 label "
            f"raster every other command in this repo reads.")
    return preds.astype(np.uint8)


# --- tiling -----------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    """One interior window, and the padded window fed to the model.

    `row`/`col`/`rows`/`cols` are the pixels this tile is responsible for.
    `pad_*` is the source rectangle read for it, `halo` px larger on every side
    and clipped at the raster edge; `top`/`left` are how much of the model input
    is replicated padding rather than pixels. The interior always sits at
    `[halo:halo+rows, halo:halo+cols]` of the model's output, because the padded
    window starts exactly `halo` px before the interior whether or not that
    lands outside the raster.
    """

    row: int
    col: int
    rows: int
    cols: int
    read_row: int
    read_col: int
    read_rows: int
    read_cols: int
    top: int
    left: int


def plan(height: int, width: int, tile: int = DEFAULT_TILE,
         halo: int = DEFAULT_HALO) -> tuple[list[Tile], int]:
    """The tile grid, plus the one fixed model input size used for all of them.

    A single input shape for every tile is deliberate: onnxruntime plans memory
    per shape, and a grid that feeds it a different rectangle at each edge pays
    that cost repeatedly for tiles that are mostly padding anyway.
    """
    if tile < 1 or halo < 0:
        raise SegmentError(f"--tile must be >= 1 and --halo >= 0, got {tile}/{halo}")
    side = tile + 2 * halo
    if side % SIZE_MULTIPLE:
        raise SegmentError(
            f"--tile + 2*--halo must be a multiple of {SIZE_MULTIPLE} (the "
            f"network's downsampling factor); {tile} + 2*{halo} = {side} is not.")

    out: list[Tile] = []
    for row in range(0, height, tile):
        rows = min(tile, height - row)
        for col in range(0, width, tile):
            cols = min(tile, width - col)
            r0, c0 = row - halo, col - halo
            rr0, cc0 = max(r0, 0), max(c0, 0)
            rr1, cc1 = min(r0 + side, height), min(c0 + side, width)
            out.append(Tile(row=row, col=col, rows=rows, cols=cols,
                            read_row=rr0, read_col=cc0,
                            read_rows=rr1 - rr0, read_cols=cc1 - cc0,
                            top=rr0 - r0, left=cc0 - c0))
    return out, side


def pad_to(arr: np.ndarray, side: int, top: int, left: int) -> np.ndarray:
    """Place `arr` at (top, left) in a `side`x`side` frame, replicating edges.

    Edge replication rather than zeros: a black border is a feature to a
    segmenter -- it predicts shadow and water along it -- while replicating the
    last real row says nothing new to the network about ground it cannot see.
    """
    h, w = arr.shape[:2]
    bottom, right = side - top - h, side - left - w
    if min(top, left, bottom, right) < 0:
        raise SegmentError(f"{arr.shape[:2]} + ({top},{left}) does not fit {side}")
    pads = [(top, bottom), (left, right)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pads, mode="edge")


@dataclass
class Result:
    """What a run did, in the numbers a report has to be able to quote."""

    out: Path
    shape: tuple[int, int]
    gsd: float
    model: str
    native_gsd: float
    tile: int
    halo: int
    side: int
    tiles: int
    seconds: float
    counts: dict[int, int] = field(default_factory=dict)
    nodata_px: int = 0
    unmapped: list[int] = field(default_factory=list)
    note: str = ""
    subset_note: str = ""

    def summary(self, mapping: dict[int, str] | None = None, top: int = 8) -> str:
        h, w = self.shape
        px = h * w
        lines = [
            f"wrote {self.out} -- {h}x{w} px @ {self.gsd:.4g} m/px, "
            f"{px / 1e6:.1f} Mpx, {1 - self.nodata_px / max(px, 1):.1%} classified",
            f"{self.model} at native {self.native_gsd:g} m/px, "
            f"{self.tiles} tiles of {self.tile} px + {self.halo} px halo "
            f"({self.side}x{self.side} model input), {self.seconds:.0f} s "
            f"= {px / 1e6 / max(self.seconds, 1e-9):.2f} Mpx/s",
        ]
        if self.subset_note:
            lines.append(self.subset_note)
        if self.unmapped:
            lines.append(f"UNMAPPED wire ids present: {self.unmapped} -- these have "
                         f"no name in the id mapping and `segmap` will refuse the "
                         f"file until they do")
        ranked = sorted(self.counts.items(), key=lambda kv: -kv[1])[:top]
        for wire, n in ranked:
            name = (mapping or {}).get(wire, f"wire {wire}")
            lines.append(f"  {n / px:6.1%}  {name}")
        return "\n".join(lines)


# --- an array in memory -----------------------------------------------------

def segment_array(rgb: np.ndarray, sess, tile: int = DEFAULT_TILE,
                  halo: int = DEFAULT_HALO) -> np.ndarray:
    """(H, W, 3) RGB -> (H, W) uint8 wire ids. The whole thing in memory.

    This is the same tiling `segment_geotiff` streams, minus the georeference,
    and it is what the tests drive with a stub session.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise SegmentError(f"expected (H, W, 3+) RGB imagery, got {rgb.shape}")
    h, w = rgb.shape[:2]
    tiles, side = plan(h, w, tile, halo)
    out = np.zeros((h, w), dtype=np.uint8)
    for t in tiles:
        window = rgb[t.read_row:t.read_row + t.read_rows,
                     t.read_col:t.read_col + t.read_cols, :3]
        preds = infer(sess, pad_to(window, side, t.top, t.left))
        out[t.row:t.row + t.rows, t.col:t.col + t.cols] = \
            preds[halo:halo + t.rows, halo:halo + t.cols]
    return out


# --- a GeoTIFF on disk, streamed --------------------------------------------
#
# A 200 Mpx ortho is 600 MB of RGB and would be another 200 MB of labels; both
# fit in RAM on this machine and neither needs to. Reading and writing by window
# keeps the footprint at one tile, which is also what makes `--max-mpx` free:
# the crop is just a different set of windows.

def _crop_window(height: int, width: int, max_mpx: float | None):
    """A centred window of at most `max_mpx` Mpx, and the note that must follow
    every number derived from it. `loader.crop_to_max_mpx` does this for labels;
    doing it here means the expensive part -- inference -- is never paid for
    pixels that were going to be cropped away anyway."""
    total = height * width
    if not max_mpx or total <= max_mpx * 1e6:
        return (0, 0, height, width), ""
    scale = (max_mpx * 1e6 / total) ** 0.5
    nh, nw = max(int(height * scale), 1), max(int(width * scale), 1)
    r0, c0 = (height - nh) // 2, (width - nw) // 2
    note = (f"SUBSET: --max-mpx {max_mpx:g} segmented a centred window only -- "
            f"rows {r0}:{r0 + nh}, cols {c0}:{c0 + nw} = {nh}x{nw} px "
            f"({nh * nw / total:.1%} of the ortho's {total / 1e6:.1f} Mpx), at "
            f"full resolution. The written raster covers that window and its "
            f"georeference says so.")
    return (r0, c0, nh, nw), note


def segment_geotiff(src_path: str | Path, out_path: str | Path, *,
                    sess=None, model: str | Path | None = None,
                    native_gsd: float | None = None, gsd: float | None = None,
                    tile: int = DEFAULT_TILE, halo: int = DEFAULT_HALO,
                    max_mpx: float | None = None, gpu: bool = False,
                    threads: int | None = None, allow_mismatch: bool = False,
                    classes: str | Path | None = None,
                    progress=None) -> Result:
    """RGB GeoTIFF in, single-band wire-id GeoTIFF out.

    The written file is what `segmap report -i <it>` reads with no further
    arguments: uint8, nodata 0 (which is `Unclassified`, exactly as the product
    writes it), the id -> name mapping in the `ID_TO_LABEL_MAPPING` tag, and the
    source georeference carried through.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:                       # pragma: no cover - env
        raise SegmentError(
            "`segmap segment` reads and writes GeoTIFFs, so it needs rasterio: "
            "pip install 'reasoning-terrain[geo]'") from exc

    from . import loader

    src_path, out_path = Path(src_path), Path(out_path)
    mapping = loader.class_map(classes or CLASSES_JSON)
    if not mapping:
        raise SegmentError(f"empty id -> name mapping ({classes or CLASSES_JSON})")
    # Fail here rather than after an hour of inference: an id mapping the
    # taxonomy does not accept makes the output unreadable by everything else.
    loader.wire_lut(mapping)

    with rasterio.open(src_path) as src:
        if src.count < 3:
            raise SegmentError(
                f"{src_path.name} has {src.count} band(s). This is the imagery "
                f"step: it wants RGB. A single-band file is already a label "
                f"raster -- point `segmap report` at it directly.")
        if src.count > 3:
            print(f"# {src_path.name} has {src.count} bands; using 1,2,3 as R,G,B",
                  file=sys.stderr)
        if set(src.dtypes[:3]) != {"uint8"}:
            # The producer's normalisation is (v - ImageNet mean) / ImageNet std,
            # which only means anything for 0..255. Handing it 16-bit imagery
            # produces a valid-looking map from inputs 250 standard deviations
            # out, so it is refused rather than scaled on a guess.
            raise SegmentError(
                f"{src_path.name} is {src.dtypes[0]}, not uint8. The model's "
                f"normalisation assumes 8-bit RGB -- convert the imagery to 8-bit "
                f"(and say how you scaled it) before segmenting.")
        px = gsd if gsd is not None else loader.geotiff_gsd(src)
        (r0, c0, height, width), subset_note = _crop_window(src.height, src.width,
                                                            max_mpx)
        tiles, side = plan(height, width, tile, halo)
        masked = any(f != rasterio.enums.MaskFlags.all_valid
                     for flags in src.mask_flag_enums for f in flags)

        if sess is None:
            weights, native, note = pick_model(px, model=model,
                                               native_gsd=native_gsd,
                                               allow_mismatch=allow_mismatch)
            model_name = weights.name
            sess = session(weights, gpu=gpu, threads=threads)
        else:
            # A caller-supplied session: there is no file to interrogate, so the
            # scale check runs on what the caller says the network is for, and
            # the weights are not required to be on this machine at all. This is
            # the path the tests take.
            native = native_gsd if native_gsd is not None else px
            note = f"model: caller-supplied session, native {native:g} m/px"
            note += check_scale(px, native, allow_mismatch)
            model_name = Path(model).name if model else "(session)"
        # Before the inference, not after it: a scale warning that arrives at the
        # end of a 30-minute run has arrived too late to act on.
        print(f"# {note}", file=sys.stderr)

        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0,
                       height=height, width=width, compress="lzw",
                       tiled=True, blockxsize=512, blockysize=512,
                       BIGTIFF="IF_SAFER")
        profile.pop("photometric", None)
        if src.transform is not None:
            profile["transform"] = src.transform * rasterio.Affine.translation(c0, r0)

        counts = np.zeros(256, dtype=np.int64)
        started = time.time()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # A stripe-organised ortho (blockysize=1, which is what these are) makes
        # every windowed read pull full-width stripes. A cache big enough to hold
        # one row of tiles' worth of them turns 20 redundant reads into 1.
        with rasterio.Env(GDAL_CACHEMAX=512), rasterio.open(out_path, "w", **profile) as dst:
            for n, t in enumerate(tiles, 1):
                win = Window(c0 + t.read_col, r0 + t.read_row,
                             t.read_cols, t.read_rows)
                rgb = src.read((1, 2, 3), window=win).transpose(1, 2, 0)
                preds = infer(sess, pad_to(rgb, side, t.top, t.left))
                out = preds[halo:halo + t.rows, halo:halo + t.cols]
                if masked:
                    # Pixels the source itself calls nodata get the product's
                    # own nodata id rather than whatever the network hallucinates
                    # over a black border.
                    inner = Window(c0 + t.col, r0 + t.row, t.cols, t.rows)
                    out = np.where(src.dataset_mask(window=inner) > 0, out, 0)
                counts += np.bincount(out.ravel(), minlength=256)
                dst.write(out, 1, window=Window(t.col, t.row, t.cols, t.rows))
                if progress:
                    progress(n, len(tiles), time.time() - started)
            dst.update_tags(
                ID_TO_LABEL_MAPPING=json.dumps({str(k): v for k, v in mapping.items()}),
                ST_MODEL=model_name,
                ST_MODEL_NATIVE_GSD=f"{native:g}",
                ST_SOURCE=src_path.name,
                ST_SOURCE_GSD=f"{px:.6g}",
                ST_TILE=str(tile), ST_HALO=str(halo),
                **({"ST_SUBSET": subset_note} if subset_note else {}))

    present = [int(i) for i in np.nonzero(counts)[0]]
    return Result(out=out_path, shape=(height, width), gsd=px,
                  model=model_name, native_gsd=native, tile=tile,
                  halo=halo, side=side, tiles=len(tiles),
                  seconds=time.time() - started,
                  counts={i: int(counts[i]) for i in present},
                  nodata_px=int(counts[0]),
                  unmapped=[i for i in present if i not in mapping],
                  note=note, subset_note=subset_note)


def stderr_progress(every: int = 10):
    """A progress line every `every` tiles, on stderr, with an ETA.

    Inference is minutes-to-an-hour of silence otherwise, and a silent hour is
    indistinguishable from a hang.
    """
    def report(n: int, total: int, elapsed: float) -> None:
        if n != total and n % every:
            return
        rate = n / max(elapsed, 1e-9)
        eta = (total - n) / max(rate, 1e-9)
        print(f"# tile {n}/{total} -- {elapsed / 60:.1f} min elapsed, "
              f"{rate * 60:.1f} tiles/min, {eta / 60:.1f} min left",
              file=sys.stderr, flush=True)
    return report
