"""Persist the Stage-0 indices, so asking a second question costs seconds.

`build_regions` and `build_chips` are query-independent by design -- and until
now every command rebuilt them from scratch. On the sinai mosaic that is half an
hour and tens of gigabytes *per question*, which makes the downstream end
impossible to poke at. This module writes the result once and reloads it.

Format: **`.npy`/`.npz` plus a `meta.json`**, not pickle. Three reasons.

  * A pickle of tens of thousands of `Region` dataclasses is slow both ways and
    breaks silently -- and differently on every Python -- the moment the
    dataclass changes shape. The columnar form is explicit: adding a field is a
    `SCHEMA_VERSION` bump and a missed cache, not a wrong index.
  * Nothing new to install. numpy is already a hard dependency.
  * The one genuinely large object, the `(H, W) int32` region-id raster, gets a
    plain `.npy` of its own so it can be **memory-mapped** on load rather than
    read. On the mosaic that is 4.8 GB that never enters RSS unless something
    actually touches it, and only two callers ever do (`s5 distance`, and
    `s4 --regions`).

The *class* raster is re-read rather than cached whenever re-reading is cheap,
which for a single GeoTIFF it is -- about a second. It is not cheap for a
directory: stitching the twenty sinai tiles is 19 s and 4 GB, and once the
indices are cached that is the entire remaining cost of asking a question. So a
**mosaic** is memoised too, as two memory-mapped `.npy` files, and a single tile
is not.

Invalidation is the part that has to be right, because a stale cache silently
returning the wrong index is worse than no cache at all. The key is a hash over
every input file's path, mtime and size, the georeferencing overrides, the
subset window, the parameters that change the result, and `SCHEMA_VERSION`. It
is checked twice: the key names the cache directory, and the full fingerprint is
re-compared against `meta.json` after loading. Anything that does not match is a
miss, and a miss rebuilds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .index import Chip, ChipIndex, Region, RegionIndex, build_chips, build_regions
from .loader import LabelRaster
from .taxonomy import ANCHOR_CLASSES, N_CLASSES

# --- cache identity --------------------------------------------------------

# Bump on ANY change to what is written below -- a new Region field, a different
# array dtype, a different meaning for an existing one. Every existing entry
# then misses and rebuilds, which is the cheap failure. Reading a v1 layout back
# as v2 is the expensive one.
SCHEMA_VERSION = 1

# Where indices live when nobody says otherwise. Relative to the working
# directory, so a checkout's caches stay with the checkout.
DEFAULT_CACHE_DIR = Path(os.environ.get("SEGMAP_CACHE", ".segmap_cache"))

# Characters of the key hash used to name a directory. 16 hex = 64 bits; the
# full fingerprint is re-checked out of meta.json after loading anyway, so this
# only has to be short enough to read and long enough not to collide by accident.
KEY_CHARS = 16

_KINDS = ("regions", "chips", "mosaic")


class CacheMiss(Exception):
    """No valid entry for this key. Not an error -- the caller builds."""


# --- fingerprinting the input ----------------------------------------------

def file_stamp(path: str | Path) -> list:
    """`[path, mtime_ns, size]` -- what makes a file *this* file.

    Content hashing would be stricter and would also mean reading 38 GB of
    GeoTIFF to decide whether to skip a 30-minute build. mtime plus size catches
    every way these files actually change (a re-export, a re-crop, a rsync).
    """
    p = Path(path).resolve()
    st = p.stat()
    return [str(p), st.st_mtime_ns, st.st_size]


def source_fingerprint(
    input_path: str | Path | None,
    *,
    gsd: float | None = None,
    dem: str | Path | None = None,
    classes: str | Path | None = None,
    subset: dict | None = None,
    synthetic: dict | None = None,
) -> dict:
    """Everything about the *input* that can change the index.

    Cheap on purpose: it stats files, it does not open them. Deciding whether the
    cache is valid must not cost a fraction of what building it costs.
    """
    if synthetic is not None:
        return {"synthetic": dict(synthetic)}
    if input_path is None:
        raise ValueError("source_fingerprint needs an input path or a synthetic spec")

    p = Path(input_path)
    if p.is_dir():
        from .mosaic import tile_paths

        files = [file_stamp(t) for t in tile_paths(p)]
        if not files:
            raise ValueError(f"{p} contains no GeoTIFFs")
    else:
        files = [file_stamp(p)]
    return {
        "files": files,
        "gsd": gsd,
        "dem": file_stamp(dem) if dem else None,
        "classes": file_stamp(classes) if classes else None,
        "subset": dict(subset) if subset else None,
    }


def cache_key(kind: str, source: dict, params: dict) -> str:
    if kind not in _KINDS:
        raise ValueError(f"unknown index kind {kind!r}; expected one of {_KINDS}")
    blob = json.dumps(
        {"schema": SCHEMA_VERSION, "kind": kind, "source": source, "params": params},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:KEY_CHARS]


@dataclass(frozen=True)
class CacheSlot:
    """One (kind, source, params) triple and the directory it lives in."""

    kind: str
    source: dict
    params: dict
    root: Path

    @property
    def key(self) -> str:
        return cache_key(self.kind, self.source, self.params)

    @property
    def path(self) -> Path:
        return Path(self.root) / f"{self.kind}-{self.key}"

    def meta(self) -> dict | None:
        f = self.path / "meta.json"
        if not f.is_file():
            return None
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        # Belt and braces: the key already names the directory, but a hand-copied
        # or hash-collided entry would still be caught here.
        if (m.get("schema") != SCHEMA_VERSION or m.get("kind") != self.kind
                or m.get("source") != self.source or m.get("params") != self.params):
            return None
        return m


def describe_age(meta: dict) -> str:
    """`built <when>` in words, for the one line this prints to stderr."""
    try:
        then = datetime.fromisoformat(meta["built_at"])
    except (KeyError, ValueError):
        return "unknown age"
    secs = max((datetime.now(timezone.utc) - then).total_seconds(), 0.0)
    for div, unit in ((86400.0, "d"), (3600.0, "h"), (60.0, "min")):
        if secs >= div:
            return f"{secs / div:.0f}{unit} ago"
    return f"{secs:.0f}s ago"


# --- region index ----------------------------------------------------------

# Region scalar fields, in the order they are packed. Name -> dtype. Kept as one
# table so writing and reading cannot drift apart.
_REGION_COLS: dict[str, str] = {
    "id": "int64",
    "class_id": "int16",
    "area_px": "int64",
    "area_m2": "float64",
    "perimeter_m": "float64",
    "compactness": "float64",
    "elongation": "float64",
    "mean_slope": "float64",
    "std_slope": "float64",
    "mean_elev": "float64",
    "aspect_circvar": "float64",
}


def _write_regions(slot: CacheSlot, ridx: RegionIndex, build_seconds: float) -> None:
    tmp = slot.path.with_name(slot.path.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    regs = ridx.regions

    cols = {k: np.asarray([getattr(r, k) for r in regs], dtype=dt)
            for k, dt in _REGION_COLS.items()}
    cols["bbox"] = np.asarray([r.bbox for r in regs],
                              dtype="int64").reshape(len(regs), 4)
    cols["centroid"] = np.asarray([r.centroid for r in regs],
                                  dtype="float64").reshape(len(regs), 2)

    # Neighbours as CSR. A dict per region pickles or JSONs badly at scale; three
    # flat arrays do not.
    indptr = np.zeros(len(regs) + 1, dtype="int64")
    nb_ids: list[int] = []
    nb_shared: list[float] = []
    for i, r in enumerate(regs):
        for nid, shared in r.neighbors.items():
            nb_ids.append(nid)
            nb_shared.append(shared)
        indptr[i + 1] = len(nb_ids)
    cols["nb_indptr"] = indptr
    cols["nb_ids"] = np.asarray(nb_ids, dtype="int64")
    cols["nb_shared"] = np.asarray(nb_shared, dtype="float64")

    np.savez_compressed(tmp / "regions.npz", **cols)
    # Uncompressed and on its own, so the warm path can mmap it instead of
    # inflating several GB it will probably never read.
    np.save(tmp / "region_labels.npy", ridx.label_array)

    _finish(slot, tmp, build_seconds, {
        "has_terrain": bool(ridx.has_terrain),
        "n_regions": len(regs),
        "shape": list(ridx.label_array.shape),
    })


def _read_regions(slot: CacheSlot, meta: dict) -> RegionIndex:
    d = slot.path
    with np.load(d / "regions.npz") as z:
        cols = {k: z[k] for k in z.files}

    bbox = cols["bbox"]
    cent = cols["centroid"]
    indptr = cols["nb_indptr"]
    nb_ids = cols["nb_ids"]
    nb_shared = cols["nb_shared"]
    n = len(cols["id"])

    scalars = {k: cols[k].tolist() for k in _REGION_COLS}
    regions = [
        Region(
            id=int(scalars["id"][i]),
            class_id=int(scalars["class_id"][i]),
            area_px=int(scalars["area_px"][i]),
            area_m2=float(scalars["area_m2"][i]),
            perimeter_m=float(scalars["perimeter_m"][i]),
            compactness=float(scalars["compactness"][i]),
            elongation=float(scalars["elongation"][i]),
            bbox=(int(bbox[i, 0]), int(bbox[i, 1]), int(bbox[i, 2]), int(bbox[i, 3])),
            centroid=(float(cent[i, 0]), float(cent[i, 1])),
            mean_slope=float(scalars["mean_slope"][i]),
            std_slope=float(scalars["std_slope"][i]),
            mean_elev=float(scalars["mean_elev"][i]),
            aspect_circvar=float(scalars["aspect_circvar"][i]),
            neighbors={int(nb_ids[j]): float(nb_shared[j])
                       for j in range(int(indptr[i]), int(indptr[i + 1]))},
        )
        for i in range(n)
    ]
    labels = np.load(d / "region_labels.npy", mmap_mode="r")
    return RegionIndex(regions, labels, has_terrain=bool(meta["has_terrain"]))


# --- chip index ------------------------------------------------------------

_CHIP_COLS: dict[str, str] = {
    "id": "int64",
    "row": "int64",
    "col": "int64",
    "entropy": "float64",
    "n_classes": "int64",
    "edge_density": "float64",
}


def _write_chips(slot: CacheSlot, cidx: ChipIndex, build_seconds: float) -> None:
    tmp = slot.path.with_name(slot.path.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    chips = cidx.chips
    anchors = list(chips[0].dist_to) if chips else list(ANCHOR_CLASSES)

    cols = {k: np.asarray([getattr(c, k) for c in chips], dtype=dt)
            for k, dt in _CHIP_COLS.items()}
    cols["bbox"] = np.asarray([c.bbox for c in chips],
                              dtype="int64").reshape(len(chips), 4)
    cols["class_frac"] = (np.stack([c.class_frac for c in chips])
                          if chips else np.zeros((0, N_CLASSES)))
    # inf is a real value here (anchor class absent from the tile) and survives
    # npz round-trip as float64.
    cols["dist_to"] = np.asarray([[c.dist_to[a] for a in anchors] for c in chips],
                                 dtype="float64").reshape(len(chips), len(anchors))

    indptr = np.zeros(len(chips) + 1, dtype="int64")
    keys: list[int] = []
    vals: list[float] = []
    for i, c in enumerate(chips):
        for (a, b), v in c.interfaces.items():
            keys.append(a * N_CLASSES + b)
            vals.append(v)
        indptr[i + 1] = len(keys)
    cols["if_indptr"] = indptr
    cols["if_key"] = np.asarray(keys, dtype="int64")
    cols["if_val"] = np.asarray(vals, dtype="float64")

    np.savez_compressed(tmp / "chips.npz", **cols)
    _finish(slot, tmp, build_seconds, {
        "n_chips": len(chips), "size": cidx.size, "stride": cidx.stride,
        "gsd": cidx.gsd, "anchors": anchors,
    })


def _read_chips(slot: CacheSlot, meta: dict) -> ChipIndex:
    with np.load(slot.path / "chips.npz") as z:
        cols = {k: z[k] for k in z.files}

    anchors = list(meta["anchors"])
    bbox, frac, dist = cols["bbox"], cols["class_frac"], cols["dist_to"]
    indptr, if_key, if_val = cols["if_indptr"], cols["if_key"], cols["if_val"]
    scalars = {k: cols[k].tolist() for k in _CHIP_COLS}

    chips = [
        Chip(
            id=int(scalars["id"][i]),
            row=int(scalars["row"][i]),
            col=int(scalars["col"][i]),
            bbox=(int(bbox[i, 0]), int(bbox[i, 1]), int(bbox[i, 2]), int(bbox[i, 3])),
            class_frac=frac[i],
            entropy=float(scalars["entropy"][i]),
            n_classes=int(scalars["n_classes"][i]),
            edge_density=float(scalars["edge_density"][i]),
            dist_to={a: float(dist[i, j]) for j, a in enumerate(anchors)},
            interfaces={divmod(int(if_key[j]), N_CLASSES): float(if_val[j])
                        for j in range(int(indptr[i]), int(indptr[i + 1]))},
        )
        for i in range(len(scalars["id"]))
    ]
    return ChipIndex(chips, int(meta["size"]), int(meta["stride"]), float(meta["gsd"]))


# --- the stitched mosaic ---------------------------------------------------

def _write_mosaic(slot: CacheSlot, raster: LabelRaster, build_seconds: float) -> None:
    tmp = slot.path.with_name(slot.path.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    # Plain .npy, uncompressed, for the same reason as the region-id raster: the
    # warm path maps these rather than reading them.
    np.save(tmp / "labels.npy", raster.labels)
    if raster.valid is not None:
        np.save(tmp / "valid.npy", raster.valid)
    tr = raster.transform
    _finish(slot, tmp, build_seconds, {
        "gsd": float(raster.gsd),
        "shape": list(raster.shape),
        "has_valid": raster.valid is not None,
        "transform": list(tr)[:6] if tr is not None else None,
        "crs": str(raster.crs) if raster.crs is not None else None,
    })


def _read_mosaic(slot: CacheSlot, meta: dict) -> LabelRaster:
    labels = np.load(slot.path / "labels.npy", mmap_mode="r")
    valid = (np.load(slot.path / "valid.npy", mmap_mode="r")
             if meta["has_valid"] else None)
    transform = crs = None
    if meta.get("transform"):
        try:
            import rasterio
            from rasterio.transform import Affine

            transform = Affine(*meta["transform"])
            crs = rasterio.crs.CRS.from_string(meta["crs"]) if meta.get("crs") else None
        except (ImportError, ValueError):  # pragma: no cover - provenance only
            transform = crs = None
    return LabelRaster(labels, float(meta["gsd"]), transform, crs, valid=valid)


def mosaic_raster(
    input_path,
    *,
    gsd=None,
    dem=None,
    classes=None,
    source: dict | None,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    quiet: bool = False,
) -> LabelRaster:
    """`loader.load()` on a directory, with the stitched arrays memoised.

    Only for directories. One GeoTIFF re-reads in about a second and caching it
    would trade that for a gigabyte of disk and one more thing to invalidate.
    """
    from . import loader

    if source is None or cache_dir is None:
        return loader.load(input_path, gsd=gsd, dem=dem, classes=classes)

    slot = CacheSlot("mosaic", source, {}, Path(cache_dir))
    raster = None
    if not refresh:
        try:
            raster, meta = load(slot)
            _note(f"# using cached mosaic ({meta['shape'][0]}x{meta['shape'][1]} px, "
                  f"stitched {describe_age(meta)}, {slot.path})", quiet)
        except CacheMiss:
            raster = None

    if raster is None:
        _note("# stitching the mosaic, this will take a while "
              f"(cached afterwards in {slot.path})", quiet)
        t0 = time.perf_counter()
        raster = loader.load(input_path, gsd=gsd, classes=classes)
        elapsed = time.perf_counter() - t0
        try:
            store(slot, raster, elapsed)
        except OSError as exc:
            _note(f"# WARNING: stitched the mosaic in {elapsed:.1f}s but could not "
                  f"cache it: {exc}", quiet)

    if dem is not None:
        raster.dem = loader.load_dem(dem, raster.shape)
    return raster


# --- write / read ----------------------------------------------------------

def _finish(slot: CacheSlot, tmp: Path, build_seconds: float, extra: dict) -> None:
    """Write meta.json last and move the directory into place atomically.

    A half-written cache that reads as valid is the one bug this module must not
    have, so nothing is visible under the real name until everything is there.
    """
    meta = {
        "schema": SCHEMA_VERSION,
        "kind": slot.kind,
        "key": slot.key,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(build_seconds, 2),
        "source": slot.source,
        "params": slot.params,
        **extra,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if slot.path.exists():
        shutil.rmtree(slot.path.with_name(slot.path.name + ".old"), ignore_errors=True)
        slot.path.rename(slot.path.with_name(slot.path.name + ".old"))
        tmp.rename(slot.path)
        shutil.rmtree(slot.path.with_name(slot.path.name + ".old"), ignore_errors=True)
    else:
        tmp.rename(slot.path)


def load(slot: CacheSlot):
    """The cached index, or raise `CacheMiss`."""
    meta = slot.meta()
    if meta is None:
        raise CacheMiss(str(slot.path))
    read = {"regions": _read_regions, "chips": _read_chips,
            "mosaic": _read_mosaic}[slot.kind]
    try:
        return read(slot, meta), meta
    except (OSError, KeyError, ValueError) as exc:
        raise CacheMiss(f"{slot.path}: {exc}") from exc


def store(slot: CacheSlot, index, build_seconds: float) -> None:
    writers = {"regions": _write_regions, "chips": _write_chips,
               "mosaic": _write_mosaic}
    writers[slot.kind](slot, index, build_seconds)


# --- the entry points the CLI uses -----------------------------------------

def _note(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def get(
    kind: str,
    raster: LabelRaster,
    *,
    source: dict | None,
    params: dict,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    quiet: bool = False,
):
    """Load `kind` from cache, or build it and store it.

    `source=None` disables caching entirely (`--no-cache`, and the synthetic
    fixture, which builds in under a second -- caching it would buy nothing and
    add a staleness failure mode).
    """
    build = (lambda: build_regions(raster, min_area_px=params["min_area_px"])) \
        if kind == "regions" else \
        (lambda: build_chips(raster, size=params["size"], stride=params["stride"],
                             anchors=tuple(params["anchors"])))

    if source is None or cache_dir is None:
        return build()

    slot = CacheSlot(kind, source, params, Path(cache_dir))
    if not refresh:
        try:
            index, meta = load(slot)
            _note(f"# using cached {kind} index (built {describe_age(meta)}, "
                  f"{meta['build_seconds']:.0f}s to build, {slot.path})", quiet)
            return index
        except CacheMiss:
            pass

    _note(f"# building {kind} index, this will take a while "
          f"(cached afterwards in {slot.path})", quiet)
    t0 = time.perf_counter()
    index = build()
    elapsed = time.perf_counter() - t0
    try:
        store(slot, index, elapsed)
        _note(f"# built {kind} index in {elapsed:.1f}s and cached it", quiet)
    except OSError as exc:
        # A full or read-only cache directory must not lose the answer the user
        # already paid for.
        _note(f"# WARNING: built the {kind} index in {elapsed:.1f}s but could not "
              f"cache it: {exc}", quiet)
    return index


def region_params(min_area_px: int) -> dict:
    return {"min_area_px": int(min_area_px)}


def chip_params(size: int, stride: int | None = None,
                anchors: tuple[str, ...] = ANCHOR_CLASSES) -> dict:
    return {"size": int(size), "stride": int(stride or size),
            "anchors": list(anchors)}


def entries(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[dict]:
    """Every readable entry, for `segmap index --list`."""
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        f = d / "meta.json"
        if not f.is_file():
            continue
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        m["path"] = str(d)
        m["bytes"] = sum(p.stat().st_size for p in d.iterdir() if p.is_file())
        out.append(m)
    return out
