"""The playground: drop in a segmentation raster, get all four no-LLM solutions.

`segmap ui` drives one preloaded AOI. This is the other shape: **upload a tile,
and see what RT makes of ground it has never seen** -- with the current OSM data
for that exact footprint fetched while you wait.

The whole point is that nothing here is a demo of a demo. The same
`loader -> index -> osm -> S1..S4` path the CLI runs is what runs behind the
form, and every panel is the CLI's own renderer. If a number appears on this
page, `segmap solve` prints the same number.

Four things it insists on, each because the alternative is a page that lies:

  **Georeference or refuse.** OSM can only be joined to a raster whose position
    on Earth is known. A GeoTIFF carries it; a PNG or an .npy does not, and the
    form asks for a bounding box rather than guessing one. A join to the wrong
    ground produces a beautiful partition of somewhere else.
  **Latest OSM, and say when.** The snapshot is refetched from Overpass for the
    uploaded footprint by default, and the page states the fetch time and the
    element counts. RT's own cache would otherwise silently answer with whatever
    was fetched last week.
  **The registration gate runs first.** If the two maps do not line up, the
    reference checks are not run at all and the page says so, rather than
    printing a thousand confident findings about a transform bug.
  **Bounded work.** An upload is cropped to a megapixel budget before indexing,
    and the crop is stated in the same breath as every number derived from it.

**Why it runs as a background job.** The first version did the whole pipeline
inside the POST and wrote the page at the end. On a real upload that is two to
five minutes of silence, and every browser gave up first -- the server log filled
with `BrokenPipeError` raised from `wfile.write(body)`, i.e. the work had
*succeeded* and the answer had nowhere to go. So: the POST starts a thread and
redirects immediately, the page polls for weighted progress, and **every finished
run is also written to disk** under `out/playground/`. A closed browser can no
longer cost you a run.

Stdlib only, one pipeline at a time, single user. This is a local tool, not a
deployment: it accepts a file from whoever can reach the port and runs a
scientific-Python pipeline over it, so bind it to localhost and leave it there.
"""

from __future__ import annotations

import base64
import html
import importlib.resources
import io
import json
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import cache as cache_mod
from . import loader

# One pipeline at a time. The indices are not reentrant, the solutions are not
# reentrant, and two people uploading 25 Mpx tiles at once is not a load profile
# worth engineering for. A second job waits here and its page says so.
_LOCK = threading.Lock()

# Ceiling on one run, and on how long a queued job waits for `_LOCK` before it
# gives up. The STEPS comment below measures region-index 16s + burn 7s +
# blocks 10s + the four S4 products ~20s + Overpass 3-40s on a 4 Mpx crop --
# call it under 100s worst case. DEFAULT_MAX_MPX is 12 Mpx, three times that
# crop; even tripling every measured step for headroom (~280s) plus whatever
# reading/positioning/indexing/S1-S3/rendering cost on top stays comfortably
# under ten minutes. 600s also isn't so tight that a legitimate large upload
# gets killed while it is simply working -- the failure this guards against is
# a blocking call that never returns at all (see `_run_job`), not a slow one.
#
# A Python thread cannot be forced to stop from outside once it is inside a
# blocking call like `rasterio.open()` on a hung FIFO -- there is no safe way
# to reclaim it. So this is two separate, honest half-measures rather than one
# complete fix: `_run_job` runs a watchdog timer that marks a job "failed" if
# it is still "running" past this deadline (the PAGE recovers), and a queued
# job calls `_LOCK.acquire(timeout=JOB_TIMEOUT_S)` instead of blocking forever
# (the QUEUE recovers). Neither one gets `_LOCK` released if the thread holding
# it is truly stuck inside a syscall that never returns -- that thread, and
# whatever OS resource it is blocked on, leaks for the life of the process.
# What this buys back is visibility and a bounded wait, not a guarantee that a
# wedged job stops occupying `_LOCK`.
JOB_TIMEOUT_S = 600

# Jobs live for the life of the process; the rendered page also lives on disk, so
# losing this dict costs you the progress history and nothing else.
_JOBS: dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()

# Past this many jobs, the oldest DONE/FAILED ones have `Job.body` (the full
# rendered HTML report, ~3.2 MB measured) dropped from memory -- `_persist()`
# already wrote the same bytes to `out_dir/index.html`, and `_report_bytes()`
# reads them back from there. The `Job` record itself (id, status, timings,
# `out_dir`) is kept indefinitely: it is a few hundred bytes, `/jobs` wants it
# for history, and `/result/<id>` needs `out_dir` to find the file on disk. A
# job that is still "waiting" or "running" is never touched by eviction.
MAX_JOBS_RETAINED = 50

# Where finished runs are written. One directory per run, so an upload is never
# lost to a closed tab.
OUT_ROOT = Path("out/playground")

# The pipeline's steps and the percentage each one is DONE at. Weighted from
# measured timings on a 4 Mpx crop (region index 16 s, burn 7 s, blocks 10 s,
# the four S4 products ~20 s, a live Overpass call 3-40 s) rather than spaced
# evenly -- an evenly spaced bar that sits at 25% for ninety seconds is worse
# than no bar, because it reads as stuck.
STEPS: tuple[tuple[str, int], ...] = (
    ("reading the raster", 5),
    ("positioning it on the Earth", 8),
    ("building the region index", 28),
    ("building the chip index", 40),
    ("fetching current OSM for this footprint", 50),
    ("burning OSM onto the raster grid", 58),
    ("checking co-registration", 62),
    ("partitioning into road-bounded blocks", 68),
    ("S1 -- consistency audit", 77),
    ("S2 -- confusion adjudication", 82),
    ("S3 -- detection triage", 87),
    ("S4 -- derived products", 96),
    ("rendering the page", 100),
)


@dataclass
class Job:
    id: str
    name: str
    status: str = "waiting"          # waiting | running | done | failed
    pct: int = 0
    step: str = "queued"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""
    detail: str = ""
    out_dir: Path | None = None
    body: bytes | None = None
    log: list[tuple[int, str]] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def advance(self, step: str, pct: int) -> None:
        self.step, self.pct = step, pct
        self.log.append((pct, step))
        print(f"  [{self.id}] {pct:3d}%  {step}", file=sys.stderr)

    def as_json(self) -> dict:
        return {"id": self.id, "status": self.status, "pct": self.pct,
                "step": self.step, "elapsed": round(self.elapsed, 1),
                "error": self.error, "name": self.name,
                "out": str(self.out_dir) if self.out_dir else "",
                "log": [{"pct": p, "step": t} for p, t in self.log]}


def _new_job(name: str) -> Job:
    # Time-ordered ids: `/jobs` then reads newest-last without a sort key, and a
    # directory listing under out/playground/ is chronological.
    jid = time.strftime("%Y%m%d-%H%M%S") + f"-{len(_JOBS) + 1:02d}"
    job = Job(id=jid, name=name)
    with _JOBS_LOCK:
        _JOBS[jid] = job
    return job

# Upload ceiling. A 200 MB GeoTIFF over a form POST is not a good experience for
# anybody and the pipeline would not finish while the browser waited.
MAX_UPLOAD_BYTES = 220 * 1024 * 1024

# Megapixel budget for one run, before indexing. `crop_to_max_mpx` states the
# window in every downstream output, so this bounds the wait without hiding
# what was left out.
DEFAULT_MAX_MPX = 12.0

# Longest side of an inline preview image. Everything on the page is a data:
# URI, so this is a page-weight decision as much as a legibility one.
PREVIEW_SIDE = 1100

ACCEPT = ".tif,.tiff,.npy,.png"

# The class mapping that ships with the package, read off the sinai drop. A real
# Smart Terrain export writes SPARSE wire ids (0..241) and is supposed to carry
# the translation as an `ID_TO_LABEL_MAPPING` GeoTIFF tag -- but several real
# drops do not, and then `loader.load` refuses rather than guessing. Asking every
# user to hunt down a JSON before they can see anything is the wrong default when
# a standard mapping exists, so the playground tries the file's own tag first and
# falls back to this, saying which it used.
#
# Located via `importlib.resources` rather than a repo-relative path, so this
# resolves under an editable install, a wheel install, or a zipped one alike --
# a plain `pip install .` no longer makes this default silently disappear.
DEFAULT_CLASSES: Path = importlib.resources.files("segmap_digest.data").joinpath(
    "smart_terrain_class_ids.json")


# --- multipart -------------------------------------------------------------
#
# Hand-rolled because `cgi` was removed in Python 3.13 and the alternatives are
# a dependency. The form is ours, so the parser only has to handle what the form
# sends: one file part, one optional file part, and a handful of text fields.

@dataclass
class Part:
    name: str
    filename: str = ""
    data: bytes = b""

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", "replace").strip()


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, Part], list[Part]]:
    """Returns (fields, files).

    Files come back as a LIST because the form accepts several rasters at once
    and a dict keyed on the field name would silently keep only the last one --
    which, with `multiple` on the input, means quietly analysing one tile of six.
    """
    m = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
    if not m:
        raise ValueError("malformed multipart request: no boundary")
    boundary = (m.group(1) or m.group(2)).strip().encode()
    sep = b"--" + boundary
    out: dict[str, Part] = {}
    files: list[Part] = []
    for chunk in body.split(sep):
        if chunk in (b"", b"--\r\n", b"--", b"\r\n"):
            continue
        chunk = chunk.lstrip(b"\r\n")
        head, _, data = chunk.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = head.decode("utf-8", "replace")
        nm = re.search(r'name="([^"]*)"', headers)
        if not nm:
            continue
        fn = re.search(r'filename="([^"]*)"', headers)
        # `data.rstrip(b"\r\n")` strips a byte SET, not one CRLF: a payload whose
        # own last bytes are 0x0A or 0x0D loses them silently. A GeoTIFF ending
        # in either is then corrupt before `loader.load` ever sees it, with no
        # error anywhere. Remove exactly the one delimiter CRLF.
        if data.endswith(b"\r\n"):
            data = data[:-2]
        part = Part(nm.group(1), fn.group(1) if fn else "", data)
        out[nm.group(1)] = part
        if part.filename and part.data and nm.group(1) == "file":
            files.append(part)
    return out, files


# --- the run ---------------------------------------------------------------

@dataclass
class Panel:
    key: str
    title: str
    blurb: str
    body: str = ""
    images: list[tuple[str, str]] = field(default_factory=list)  # (caption, data-uri)
    note: str = ""


@dataclass
class Run:
    source: str
    facts: list[tuple[str, str]] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    # The interactive tile map, as ready-to-insert HTML. Empty when it could not
    # be built -- a failure there must never cost the user the run itself.
    map_html: str = ""


def _grid_px(shape: tuple[int, int]) -> int:
    """Tile size that puts the grid in a range a human can actually point at.

    Too few and a tile is not a place; too many and the overlay is confetti and
    the page carries thousands of records. Aim for roughly 150-500 tiles.
    """
    longest = max(shape)
    for px in (64, 128, 256, 512, 1024):
        if (longest / px) ** 2 <= 500:
            return px
    return 1024


def _png_data_uri(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _downsample(arr: np.ndarray, side: int = PREVIEW_SIDE) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(int(np.ceil(max(h, w) / side)), 1)
    return arr[::step, ::step]


def _label_preview(raster) -> str:
    rgb = loader.colourise(_downsample(raster.labels))
    if raster.valid is not None:
        rgb[~_downsample(raster.valid)] = (30, 30, 30)
    return _png_data_uri(rgb)


def _product_preview(arr: np.ndarray) -> str:
    """False-colour preview. `arr` may carry NaN for ground a provider never
    measured (see `traversability.TraversabilityResult`) -- those pixels are
    painted flat grey rather than run through the colour ramp, where a NaN
    would otherwise silently cast to an arbitrary uint8 and could not be told
    apart from a real, measured value."""
    small = _downsample(arr)
    unmeasured = np.isnan(small)
    x = np.clip(np.where(unmeasured, 0.0, small), 0, 1)
    rgb = np.stack([
        np.clip(1.6 * x - 0.35, 0, 1) * 255,
        np.clip(1.2 * x + 0.05, 0, 1) * 255,
        np.clip(1.1 - 1.3 * x, 0, 1) * 255,
    ], axis=-1).astype(np.uint8)
    rgb[unmeasured] = (60, 60, 60)
    return _png_data_uri(rgb)


def run_pipeline(path: Path, opts: dict, cache_dir: Path, job=None) -> Run:
    """Everything, in the order the pipeline actually runs it.

    `job` is advanced through `STEPS` as it goes. The steps are the real phase
    boundaries, not a timer: if a phase is slow the bar sits still, which is
    honest, and the step label says which one.
    """
    steps = iter(STEPS)

    def tick(skipped: bool = False):
        """Advance one step. `skipped=True` still consumes it -- so the bar
        reaches 100% -- but says the phase did not run rather than ticking it
        green. A progress bar that reports work it did not do is worse than one
        that stalls."""
        if job is not None:
            label, pct = next(steps)
            job.advance(f"{label} — SKIPPED, no OSM layer" if skipped else label,
                        pct)

    from . import digests
    from .osm import chipfeat, preview as osm_preview
    from .osm import layer as osm_layer
    from .solutions import s1_audit, s2_adjudicate, s3_triage, s4_products

    t0 = time.perf_counter()
    run = Run(source=path.name)

    tick()                                            # reading the raster
    classes = opts.get("classes_path")
    gsd = opts.get("gsd")
    raster, class_note = _load_source(path, gsd, classes)

    tick()                                            # positioning
    bbox = opts.get("bbox")
    if raster.transform is None:
        if not bbox:
            raise ValueError(
                f"{path.name} carries no georeference, so OSM cannot be joined "
                f"to it: there is no way to know which ground it covers. Either "
                f"upload the GeoTIFF (which carries its own transform) or give "
                f"the bounding box the raster covers, as south,west,north,east "
                f"in degrees.")
        loader.georeference(raster, bbox)
        run.warnings.append(
            f"This raster had no georeference; it was positioned from the "
            f"bounding box you gave ({', '.join(f'{v:g}' for v in bbox)}) and "
            f"assumed north-up and covering it exactly. Nothing downstream can "
            f"check that assumption.")

    max_mpx = opts.get("max_mpx") or DEFAULT_MAX_MPX
    if raster.shape[0] * raster.shape[1] > max_mpx * 1e6:
        raster = loader.crop_to_max_mpx(raster, max_mpx)
        run.warnings.append(raster.subset_note)

    h, w = raster.shape
    if path.is_dir():
        tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        src_line = (f"{path.name}/ — {len(tifs)} tiles mosaicked "
                    f"({sum(t.stat().st_size for t in tifs) / 1e6:.1f} MB)")
    else:
        src_line = f"{path.name} ({path.stat().st_size / 1e6:.1f} MB)"
    run.facts += [
        ("source", src_line),
        ("class mapping", class_note),
        ("raster", f"{h} x {w} px = {h * w / 1e6:.2f} Mpx"),
        ("resolution", f"{raster.gsd:.3f} m/px"),
        ("ground", f"{h * w * raster.gsd ** 2 / 1e6:.3f} km²"),
        ("classified", f"{1 - raster.nodata_frac:.1%} of the extent"),
        ("classes present", str(int((np.bincount(
            raster.labels[raster.valid] if raster.valid is not None
            else raster.labels.ravel()).astype(bool)).sum()))),
    ]

    tick()                                            # region index
    ridx = cache_mod.get("regions", raster, source=None,
                         params=cache_mod.region_params(opts.get("min_px", 12)),
                         cache_dir=None, quiet=True)
    tick()                                            # chip index
    chip_px = opts.get("chip", 256)
    cidx = cache_mod.get("chips", raster, source=None,
                         params=cache_mod.chip_params(chip_px),
                         cache_dir=None, quiet=True)
    run.facts.append(("regions", f"{len(ridx.regions):,} connected components "
                                 f"at min {opts.get('min_px', 12)} px"))
    run.facts.append(("chips", f"{len(cidx.chips):,} of {chip_px} px"))

    # --- OSM, live ---------------------------------------------------------
    # Three ticks around one call, because `osm_layer.build` is three phases and
    # the network one is the only step whose duration is out of our hands.
    # It is allowed to fail. Overpass is a free service under permanent load and
    # answers 500 or 504 when busy; a run that has already spent two minutes
    # indexing must not be thrown away because a third-party server was tired.
    # Everything downstream already handles `osm=None` -- S1 drops its reference
    # checks, S2's reference term goes silent, S4 skips its overlays -- so the
    # honest degradation is to carry on and say so loudly.
    tick()                                            # fetching OSM
    osm = None
    try:
        osm = osm_layer.build(
            raster, cache_dir=cache_dir,
            layers=("road", "building", "water", "flow", "barrier",
                    "built_landuse"),
            cut_layers=("road",), refresh_osm=opts.get("refresh_osm", True),
            allow_network=True, quiet=True,
        )
    except (ConnectionError, ValueError, OSError) as exc:
        run.warnings.append(
            f"NO OSM LAYER — {exc}  Everything below ran on the segmentation "
            f"alone: no reference checks in S1, no reference term in S2, no "
            f"road or building overlay in S4, and no block partition. Re-run to "
            f"retry the fetch.")
        run.facts.append(("OSM", "UNAVAILABLE — see the warning above"))
    if osm is not None:
        tick()                                        # burning
        tick()                                        # co-registration
        tick()                                        # blocks
    else:
        # These three sat OUTSIDE the guard, so a failed Overpass fetch still
        # ticked "burning OSM ✓", "checking co-registration ✓" and
        # "partitioning into blocks ✓" while the report's own warning said NO
        # OSM LAYER. They still consume their steps -- the bar must reach 100%
        # -- but they say so.
        tick(skipped=True)
        tick(skipped=True)
        tick(skipped=True)

    if osm is not None:
        counts = ", ".join(f"{k} {v}" for k, v in osm.vectors.counts().items())
        run.facts += [
            ("OSM ways",
             f"{len(osm.vectors.ways):,} -- {counts or 'nothing mapped here'}"),
            ("OSM fetched",
             osm.vectors.provenance.get("fetched_utc", "from a local file") + " UTC"),
            ("blocks", f"{len(osm.blocks.blocks):,} road-bounded"),
            ("junctions", f"{osm.graph.n_junctions:,} on this raster"),
        ]
        if osm.align is not None:
            run.facts.append(("co-registration",
                              f"{osm.align.verdict} -- best shift "
                              f"{osm.align.best_shift_m:.2f} m"))
            if osm.align.verdict != "REGISTERED":
                run.warnings.append(osm.align.note)
        if osm.blocks.degenerate:
            run.warnings.append("Block partition is degenerate: " + osm.blocks.note)

    # --- panel 0: the join -------------------------------------------------
    p0 = Panel("osm", "The join — OSM as a second map of this ground",
               "Not a solution: the ingestion step all four sit on. The road "
               "corridor is buffered from each way's own tags, the blocks are "
               "what the network encloses, and the class histogram inside each "
               "block is what ST says is there.")
    label_uri = _label_preview(raster)
    overlay_uri = ""
    p0.images = [("The segmentation, colourised by class", label_uri)]
    if osm is not None:
        overlay_uri = _png_data_uri(_read_png(osm_preview.write(
            raster, osm, str(cache_dir / "_pg_preview.png"),
            max_side=PREVIEW_SIDE)))
        p0.images.append(
            ("OSM road corridor (dark) and block edges over the same ground",
             overlay_uri))
        p0.body = osm.summary() + "\n\n" + digests.block_table(osm.blocks, limit=15)
    else:
        p0.body = ("# no OSM layer for this run -- the join was unavailable.\n"
                   "# Everything below is the segmentation reasoning about itself.")
    run.panels.append(p0)

    # --- S1 ----------------------------------------------------------------
    tick()
    rep = s1_audit.run(ridx, min_area_m2=25.0, osm=osm, raster=raster)
    n_osm = sum(1 for f in rep.findings if f.kind.startswith("osm-"))
    p1 = Panel("s1", "S1 — consistency audit",
               "Which labels contradict their own geometry, their neighbours, or "
               "the reference map. Rows are root causes, not findings: one row "
               "is one thing to decide about, however many regions it covers.")
    p1.note = (f"{len(rep.findings):,} findings over {len(rep.causes)} distinct "
               f"causes; {n_osm:,} of them came from the OSM join, which is the "
               f"only check family that can contradict a label using something "
               f"other than the label raster itself.")
    p1.body = s1_audit.render(rep, budget=opts.get("budget", 15))
    run.panels.append(p1)

    # --- S2 ----------------------------------------------------------------
    tick()
    rids, seen = [], set()
    for f in rep.findings:
        if f.region_id and f.region_id not in seen:
            seen.add(f.region_id)
            rids.append(f.region_id)
        if len(rids) >= opts.get("n_adjudicate", 4):
            break
    adjs = [s2_adjudicate.adjudicate(ridx, rid, osm=osm, raster=raster)
            for rid in rids]
    p2 = Panel("s2", "S2 — confusion adjudication",
               "S1 says a region is wrong; S2 answers what it is, or refuses. "
               "The `osm` column is the reference term: OSM is the authority on "
               "whether something is there and what it is, ST on what it looks "
               "like now, so a mapped footprint over bare ground argues for "
               "House without being allowed to overrule the later observation.")
    p2.body = ("\n\n".join(s2_adjudicate.render(a) for a in adjs)
               + "\n\n" + s2_adjudicate.ledger(adjs)) if adjs else \
        "# S1 raised nothing region-shaped to adjudicate on this tile."
    run.panels.append(p2)

    # --- S3 ----------------------------------------------------------------
    tick()
    if osm is not None:
        chipfeat.attach(cidx, raster, osm)
    policy_name = opts.get("policy", "settlement")
    pol, sel = s3_triage.run(cidx, policy_name,
                             budget_frac=opts.get("budget_frac", 0.2))
    # Hand the triage verdict back to the chips so the tile map can show it.
    # `score_chips` returns ScoredChip wrappers rather than mutating the index,
    # which is right for the CLI and leaves the map with nothing to read.
    for bucket, flag in ((sel.rejected, 0.0), (sel.control, 0.0),
                         (sel.selected, 1.0)):
        for sc in bucket:
            sc.chip.osm["score"] = float(sc.score)
            sc.chip.osm["selected"] = flag
    missing = chipfeat.missing_features(pol, chipfeat.CHIP_FEATURES,
                                        has_dist=True, has_interfaces=True)
    p3 = Panel("s3", "S3 — detection triage",
               "Which units are worth an expensive detector. The OSM features "
               "(`osm.dist_road`, `osm.building_frac`, `osm.n_junctions`) are "
               "usable in a policy rule exactly like the class fractions, which "
               "gives the scorer a settlement prior that is a fact rather than "
               "an inference from the same map being triaged.")
    p3.body = s3_triage.render(sel, pol, missing=missing)
    if osm is not None:
        try:
            bpol, bsel, bmissing, blabels = s3_triage.run_blocks(
                raster, osm, policy_name, budget_frac=opts.get("budget_frac", 0.2))
            p3.body += "\n\n" + s3_triage.render(bsel, bpol, unit="block",
                                                 missing=bmissing, labels=blabels)
        except (ValueError, IndexError) as exc:
            p3.body += f"\n\n# block unit unavailable on this tile: {exc}"
    run.panels.append(p3)

    # --- S4 ----------------------------------------------------------------
    tick()
    p4 = Panel("s4", "S4 — derived products",
               "47 classes plus slope plus season become the raster somebody "
               "asked for. OSM enters as named overlay terms — a mapped street "
               "is passable even where the segmenter saw only shadow — and each "
               "product states what the overlay moved before showing a figure.")
    bodies = []
    product_rasters: dict[str, np.ndarray] = {}
    # Read the registry rather than a hard-coded list: the product set is the
    # solution's to decide, and a name added there but not here is a product
    # that silently never renders.
    for product in sorted(s4_products.PRODUCTS):
        kw: dict = {}
        if product == "trafficability":
            kw = {"vehicle": opts.get("vehicle", "wheeled"),
                  "wet": bool(opts.get("wet"))}
        elif product == "concealment":
            kw = {"target": opts.get("target", "person")}
        # Both rasters, once each. `osm_delta` would recompute them -- eight
        # full-raster passes for four products, which was a third of the wait.
        before = s4_products.compute(raster, product, osm=None, **kw)
        after = s4_products.compute(raster, product, osm=osm, **kw)
        title = product + (f" ({kw['vehicle']}{', wet' if kw.get('wet') else ''})"
                           if product == "trafficability" else
                           (f" ({kw['target']})" if product == "concealment" else ""))
        bodies.append(f"# {title}\n"
                      + s4_products.osm_delta_from(raster, before, after, osm)
                      + "\n" + s4_products.summarise(raster, after, None))
        p4.images.append((title, _product_preview(after)))
        product_rasters[product] = after
    # Traversability comes from a provider the owner supplies -- this repo only
    # owns the seam. Wrapped, because a provider that is not installed yet must
    # cost the traversability row and nothing else.
    try:
        from . import traversability as trav

        tr = trav.estimate(raster, vehicle=opts.get("vehicle", "wheeled"))
        product_rasters["traversability"] = tr.score
        p4.images.append((f"traversability ({tr.provider})",
                          _product_preview(tr.score)))
        unmeasured = tr.unmeasured_fraction
        bodies.append(
            f"# traversability -- provider `{tr.provider}`, "
            f"terrain {'measured' if tr.has_terrain else 'UNMEASURED'}\n"
            + (f"# {unmeasured:.1%} of this raster is unmeasured (NaN) -- treat "
               f"that ground as unknown, not impassable\n" if unmeasured else "")
            + "\n".join(f"# {n}" for n in tr.notes)
            + "\n" + s4_products.summarise(raster, tr.score, None))
        run.facts.append(("traversability", f"{tr.provider}"
                          + ("" if tr.has_terrain else " — no DEM/DSM yet, "
                             "slope unmeasured")))
    except Exception as exc:                          # noqa: BLE001
        run.warnings.append(f"traversability provider unavailable: "
                            f"{type(exc).__name__}: {exc}")

    p4.body = "\n\n".join(bodies)
    run.panels.append(p4)

    tick()                                            # rendering
    # --- the interactive map ----------------------------------------------
    # Wrapped: a bug in the map must cost the map, never the four solutions the
    # user actually waited for.
    try:
        from . import tilemap, tilemap_ui

        tile_px = _grid_px(raster.shape)
        ti = tilemap.build_tile_index(raster, ridx, cidx, osm, rep, adjs,
                                      product_rasters, tile_px=tile_px)
        run.map_html = tilemap_ui.render_map_section(ti, label_uri, overlay_uri)
        run.facts.append(("tile grid",
                          f"{ti['cols']}x{ti['rows']} = {ti['n_tiles']} tiles of "
                          f"{tile_px} px ({tile_px * raster.gsd:.0f} m)"))
    except Exception as exc:                          # noqa: BLE001
        run.warnings.append(
            f"The interactive tile map could not be built ({type(exc).__name__}: "
            f"{exc}). Everything below is unaffected.")

    run.seconds = time.perf_counter() - t0
    return run


# Upper bound on a form-supplied megapixel budget. Beyond this the pipeline is
# not slow, it is a way to exhaust the machine.
MAX_MPX_CEILING = 400.0


def _positive(value, default, name):
    """A finite, positive number, or a clear refusal.

    `float("nan")`, `inf` and a huge value all reached the pipeline: NaN and inf
    disabled the megapixel crop outright, because every comparison against NaN
    is False.
    """
    import math

    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number; got {value!r}")
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}")
    if name == "max_mpx" and v > MAX_MPX_CEILING:
        raise ValueError(f"max_mpx {v:g} is above this server's ceiling of "
                         f"{MAX_MPX_CEILING:g} Mpx. Crop first, or use the CLI.")
    return v


def _load_source(path: Path, gsd, classes):
    """Load a single raster or a whole directory, and say which mapping was used.

    A directory is stitched into one mosaic before anything else looks at it,
    because per-tile analysis cuts every region at the seam: one wadi across four
    tiles becomes four regions with four wrong areas and four wrong neighbour
    lists. The stitch is the expensive part and is memoised exactly as the CLI
    memoises it -- 19 s for twenty tiles, paid once per directory.
    """
    if not path.is_dir():
        return _load_with_fallback(path, gsd, classes)

    from .mosaic import tile_paths

    tiles = tile_paths(path)
    if not tiles:
        raise ValueError(
            f"{path} contains no GeoTIFFs. A directory input is mosaicked, so it "
            f"needs .tif/.tiff files that carry their own affine transforms.")
    try:
        source = cache_mod.source_fingerprint(str(path), gsd=gsd, classes=classes)
    except (OSError, ValueError):
        source = None
    try:
        raster = cache_mod.mosaic_raster(
            str(path), gsd=gsd, classes=classes, source=source,
            cache_dir=cache_mod.DEFAULT_CACHE_DIR, quiet=True)
        return raster, ("the tiles' own ID_TO_LABEL_MAPPING tags" if classes is None
                        else "supplied with the upload")
    except ValueError as exc:
        if "0..46" not in str(exc) and "wire ids" not in str(exc):
            raise
        if not DEFAULT_CLASSES.is_file():
            raise
        try:
            source = cache_mod.source_fingerprint(str(path), gsd=gsd,
                                                  classes=str(DEFAULT_CLASSES))
        except (OSError, ValueError):
            source = None
        raster = cache_mod.mosaic_raster(
            str(path), gsd=gsd, classes=str(DEFAULT_CLASSES), source=source,
            cache_dir=cache_mod.DEFAULT_CACHE_DIR, quiet=True)
        return raster, (f"the bundled default ({DEFAULT_CLASSES.name}) -- these "
                        f"tiles carry no ID_TO_LABEL_MAPPING tag")


def _load_with_fallback(path: Path, gsd, classes):
    """Load the raster, falling back to the bundled mapping when ids are sparse.

    Returns `(raster, note)` where the note says which mapping was used -- that
    line goes in the report, because reading a raster under the wrong id mapping
    produces confident labels for the wrong classes and nothing downstream can
    tell.
    """
    if classes:
        return loader.load(path, gsd=gsd, classes=classes), "supplied with the upload"
    try:
        return loader.load(path, gsd=gsd), "the file's own ID_TO_LABEL_MAPPING tag"
    except ValueError as exc:
        # Only the sparse-wire-id refusal is retryable. A mixed-CRS mosaic or an
        # unreadable file must still fail loudly.
        if "0..46" not in str(exc) and "wire ids" not in str(exc):
            raise
        if not DEFAULT_CLASSES.is_file():
            raise ValueError(
                f"{path.name} uses sparse wire ids and carries no "
                f"ID_TO_LABEL_MAPPING tag, and the bundled default mapping is "
                f"missing from {DEFAULT_CLASSES}. Upload the class-mapping JSON.")
        return (loader.load(path, gsd=gsd, classes=str(DEFAULT_CLASSES)),
                f"the bundled default ({DEFAULT_CLASSES.name}) -- this file "
                f"carries no ID_TO_LABEL_MAPPING tag")


def _read_png(path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


# --- pages -----------------------------------------------------------------

CSS = """
:root{
  --bg:#f7f8f9; --panel:#ffffff; --ink:#14181d; --dim:#5d6772; --line:#dde2e7;
  --accent:#2f5d7c; --soft:#eaeff3; --ok:#2c7a52; --warn:#8a6320; --bad:#a33b2a;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1418; --panel:#161d23; --ink:#e6ecf2; --dim:#8b97a3; --line:#26303a;
  --accent:#7fb3d5; --soft:#1b242c; --ok:#6fc79a; --warn:#d9ae62; --bad:#e89078;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);margin:0 0 28px;max-width:70ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:20px 22px;margin:0 0 20px}
label{display:block;font-size:13px;color:var(--dim);margin:0 0 4px}
input,select{width:100%;padding:8px 10px;border:1px solid var(--line);
  border-radius:6px;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px}
input[type=checkbox]{width:auto;margin-right:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
      gap:14px 18px;margin:14px 0}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;
  padding:11px 22px;font-family:var(--sans);font-size:15px;font-weight:600;
  cursor:pointer}
button:disabled{opacity:.55;cursor:progress}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;
  padding:14px;overflow-x:auto;font-family:var(--mono);font-size:12.5px;
  line-height:1.5;margin:14px 0 0;white-space:pre;max-height:560px}
table.facts{border-collapse:collapse;width:100%;font-size:14px}
table.facts td{padding:5px 10px 5px 0;border-bottom:1px solid var(--line);
  vertical-align:top}
table.facts td:first-child{color:var(--dim);width:170px;white-space:nowrap}
.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
      gap:16px;margin:16px 0 0}
.imgs figure{margin:0}
.imgs img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.imgs figcaption{font-size:12.5px;color:var(--dim);margin-top:6px}
.warn{border-left:3px solid var(--warn);background:var(--soft);padding:10px 14px;
  border-radius:0 6px 6px 0;margin:0 0 12px;font-size:14px}
.err{border-left:3px solid var(--bad)}
.note{color:var(--dim);font-size:14px;margin:8px 0 0}
.blurb{color:var(--dim);max-width:80ch;margin:2px 0 0}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 7px;
  border-radius:999px;background:var(--soft);color:var(--dim);
  border:1px solid var(--line);margin-right:8px;vertical-align:2px}
a{color:var(--accent)}
.foot{color:var(--dim);font-size:13px;margin-top:36px}

/* progress */
.bar{height:10px;background:var(--soft);border:1px solid var(--line);
  border-radius:999px;overflow:hidden;margin:18px 0 10px}
.bar i{display:block;height:100%;width:0;background:var(--accent);
  border-radius:999px;transition:width .45s ease}
.pct{font-family:var(--mono);font-size:34px;font-weight:600;letter-spacing:-.02em;
  line-height:1;font-variant-numeric:tabular-nums}
.stepline{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
  flex-wrap:wrap}
.stepname{font-size:16px}
.elapsed{font-family:var(--mono);font-size:13px;color:var(--dim);
  font-variant-numeric:tabular-nums}
ol.steps{list-style:none;padding:0;margin:20px 0 0;display:flex;
  flex-direction:column;gap:2px}
ol.steps li{display:flex;gap:10px;align-items:baseline;font-size:13.5px;
  color:var(--dim);font-family:var(--mono)}
ol.steps li .m{width:14px;flex:none;text-align:center}
ol.steps li.now{color:var(--ink);font-weight:600}
ol.steps li.todo{opacity:.45}
.done{border-left:3px solid var(--ok);background:var(--soft);padding:14px 18px;
  border-radius:0 6px 6px 0;margin:0 0 16px}
.done b{color:var(--ok)}
.pathline{font-family:var(--mono);font-size:12.5px;color:var(--dim);
  word-break:break-all}
table.jobs{border-collapse:collapse;width:100%;font-size:14px}
table.jobs td,table.jobs th{padding:8px 12px 8px 0;border-bottom:1px solid var(--line);
  text-align:left}
table.jobs th{font-family:var(--mono);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--dim);font-weight:400}
"""

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>%s</style></head><body><div class="wrap">"""


def page_upload(message: str = "", is_error: bool = False) -> bytes:
    from .solutions import s3_triage

    policies = "".join(f'<option value="{p}"{" selected" if p == "settlement" else ""}>'
                       f'{p}</option>' for p in s3_triage.BUILTIN)
    banner = (f'<div class="warn{" err" if is_error else ""}">{html.escape(message)}'
              f'</div>' if message else "")
    return ((HEAD % ("RT playground", CSS)) + f"""
<h1>RT playground</h1>
<p class="sub">Drop in a Smart Terrain segmentation raster. RT indexes it, pulls
the <strong>current OpenStreetMap data for that exact footprint</strong>, and runs
all four solutions that need no model: the consistency audit, confusion
adjudication, detection triage and the derived products.</p>
{banner}
<form class="card" method="post" action="/run" enctype="multipart/form-data"
      onsubmit="go()">
  <div class="grid">
    <div style="grid-column:1/-1">
      <label for="f">Segmentation raster — one file, or <strong>select several
        tiles</strong> and they are mosaicked before anything looks at them.
        GeoTIFF keeps its georeference; .npy and .png need a bounding box below.</label>
      <input id="f" type="file" name="file" accept="{ACCEPT}" multiple>
    </div>
    <div style="grid-column:1/-1">
      <label for="d">…or a <strong>directory on this machine</strong> — every
        GeoTIFF inside it is mosaicked. Faster than uploading, since the server
        is local.</label>
      <input id="d" name="dirpath" placeholder="data/incoming/leb">
    </div>
    <div style="grid-column:1/-1">
      <label for="c">Class mapping — <strong>leave empty</strong> unless your
        export uses a non-standard one. The file's own
        <code>ID_TO_LABEL_MAPPING</code> tag is used when present, and the
        standard Smart Terrain mapping otherwise.</label>
      <input id="c" type="file" name="classes" accept=".json">
    </div>
    <div><label for="b">Bounding box, S,W,N,E in degrees</label>
      <input id="b" name="bbox" placeholder="31.494,34.473,31.512,34.494"></div>
    <div><label for="g">Metres per pixel (blank = from the header)</label>
      <input id="g" name="gsd" placeholder="0.5"></div>
    <div><label for="m">Megapixel budget</label>
      <input id="m" name="max_mpx" value="{DEFAULT_MAX_MPX:g}"></div>
    <div><label for="p">Triage policy</label>
      <select id="p" name="policy">{policies}</select></div>
    <div><label for="v">Vehicle</label><select id="v" name="vehicle">
      <option>wheeled</option><option>tracked</option><option>foot</option>
    </select></div>
    <div><label for="tg">Concealing what?</label><select id="tg" name="target">
      <option>person</option><option>vehicle</option><option>structure</option>
    </select></div>
    <div><label for="mp">Minimum region size (px)</label>
      <input id="mp" name="min_px" value="12"></div>
    <div><label>&nbsp;</label>
      <label><input type="checkbox" name="refresh_osm" checked> refetch OSM now
        (uncheck to reuse a cached snapshot)</label>
      <label><input type="checkbox" name="wet"> wet season</label></div>
  </div>
  <button id="go" type="submit">Run it</button>
  <p class="note" id="wait" style="display:none">Starting the run — you will be
    taken to a progress page in a moment.</p>
</form>
<div class="card">
  <h2>What it will do</h2>
  <p class="blurb">Thirteen steps, reported as they happen: load the raster ·
  build the region and chip indices · derive the WGS84 footprint from the affine ·
  query Overpass for the roads, buildings, water and barriers inside it · burn
  them onto the raster's own grid · check the two maps are co-registered before
  comparing them · partition the ground into road-bounded blocks · then S1, S2,
  S3 and S4 over both maps together. Every panel is the CLI's own renderer, so
  anything you see is reproducible with <code>segmap solve</code>.</p>
</div>
<p class="foot">The run happens in the background: this page hands you a progress
bar, and the finished report is written to <code>{OUT_ROOT}/</code> whether or not
the browser is still open. <a href="/jobs">Past runs →</a><br>
Local tool. It runs a scientific-Python pipeline over whatever file it is given,
so keep it bound to localhost.</p>
<script>function go(){{document.getElementById('go').disabled=true;
document.getElementById('go').textContent='Starting…';
document.getElementById('wait').style.display='block';}}</script>
</div></body></html>""").encode()


def page_results(run: Run) -> bytes:
    facts = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
                    for k, v in run.facts)
    warns = "".join(f'<div class="warn">{html.escape(w)}</div>'
                    for w in run.warnings)
    panels = []
    for p in run.panels:
        imgs = ""
        if p.images:
            imgs = '<div class="imgs">' + "".join(
                f'<figure><img alt="{html.escape(c)}" src="{u}">'
                f'<figcaption>{html.escape(c)}</figcaption></figure>'
                for c, u in p.images) + "</div>"
        note = f'<p class="note">{html.escape(p.note)}</p>' if p.note else ""
        panels.append(
            f'<div class="card"><h2><span class="tag">{p.key}</span>'
            f'{html.escape(p.title)}</h2>'
            f'<p class="blurb">{html.escape(p.blurb)}</p>{note}{imgs}'
            f'<pre>{html.escape(p.body)}</pre></div>')
    return ((HEAD % (f"RT — {html.escape(run.source)}", CSS)) + f"""
<h1>{html.escape(run.source)}</h1>
<p class="sub">Four solutions, no model in the loop, over the segmentation and
the current OSM data for the same ground. {run.seconds:.1f} s end to end.</p>
{warns}
<div class="card"><h2>What was read</h2><table class="facts">{facts}</table></div>
{run.map_html}
{''.join(panels)}
<p class="foot"><a href="/">← run another</a> · every panel is the output of
<code>segmap solve … --osm</code>, unchanged.</p>
</div></body></html>""").encode()


def page_progress(job: Job) -> bytes:
    """The page the browser lands on immediately after the POST.

    It polls rather than streaming: a poll survives a reload, a laptop lid, and a
    proxy that buffers, and the job keeps running regardless of whether anyone is
    watching it.
    """
    steps = "".join(
        f'<li data-pct="{pct}" class="todo"><span class="m">·</span>'
        f'<span>{html.escape(label)}</span></li>'
        for label, pct in STEPS)
    return ((HEAD % (f"RT — running {html.escape(job.name)}", CSS)) + f"""
<h1>Running</h1>
<p class="sub">{html.escape(job.name)} — indexing, fetching current OSM for its
footprint, then S1 through S4. This page updates itself; you can leave it and come
back to <code>/job/{job.id}</code>, and the finished page is written to disk either
way.</p>
<div class="card">
  <div class="stepline">
    <span class="pct" id="pct">0%</span>
    <span class="elapsed" id="el">0.0 s</span>
  </div>
  <div class="bar"><i id="fill"></i></div>
  <div class="stepline"><span class="stepname" id="step">starting…</span></div>
  <ol class="steps" id="steps">{steps}</ol>
</div>
<div class="card" id="result" style="display:none">
  <h2>Done</h2>
  <div class="done" id="donebox"></div>
  <p><a id="link" href="#">Open the report →</a></p>
  <p class="note">Also written to disk: <span class="pathline" id="outpath"></span></p>
</div>
<div class="card" id="failed" style="display:none">
  <h2>That did not run</h2>
  <div class="warn err" id="errbox"></div>
  <p><a href="/">← back to the form</a></p>
</div>
<p class="foot"><a href="/jobs">every run this server has done →</a></p>
<script>
var JOB = {job.id!r};
var lastPct = -1;
function paint(d){{
  document.getElementById('pct').textContent = d.pct + '%';
  document.getElementById('fill').style.width = d.pct + '%';
  document.getElementById('step').textContent = d.step;
  document.getElementById('el').textContent = d.elapsed.toFixed(1) + ' s';
  if (d.pct !== lastPct) {{
    lastPct = d.pct;
    document.querySelectorAll('#steps li').forEach(function(li){{
      var p = parseInt(li.dataset.pct, 10);
      li.className = p < d.pct ? 'done' : (p === d.pct ? 'now' : 'todo');
      li.querySelector('.m').textContent =
        p < d.pct ? '✓' : (p === d.pct ? '▸' : '·');
    }});
  }}
  if (d.status === 'done') {{
    document.getElementById('result').style.display = 'block';
    document.getElementById('donebox').innerHTML =
      '<b>Finished</b> in ' + d.elapsed.toFixed(1) + ' s — all four solutions rendered.';
    var a = document.getElementById('link');
    a.href = '/result/' + JOB;
    document.getElementById('outpath').textContent = d.out;
    document.getElementById('step').textContent = 'complete';
    document.title = '✓ RT — ' + d.name;
    return true;
  }}
  if (d.status === 'failed') {{
    document.getElementById('failed').style.display = 'block';
    document.getElementById('errbox').textContent = d.error;
    document.getElementById('step').textContent = 'failed';
    document.title = '✗ RT — ' + d.name;
    return true;
  }}
  return false;
}}
function poll(){{
  fetch('/api/job/' + JOB, {{cache: 'no-store'}})
    .then(function(r){{ return r.json(); }})
    .then(function(d){{ if (!paint(d)) setTimeout(poll, 700); }})
    .catch(function(){{ setTimeout(poll, 2000); }});
}}
poll();
</script>
</div></body></html>""").encode()


def page_jobs() -> bytes:
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())[::-1]
    if not jobs:
        rows = '<tr><td colspan="5">nothing yet</td></tr>'
    else:
        rows = "".join(
            f'<tr><td><code>{j.id}</code></td><td>{html.escape(j.name)}</td>'
            f'<td>{j.status}</td><td>{j.elapsed:.0f} s</td>'
            f'<td>' + (f'<a href="/result/{j.id}">report</a>'
                       if j.status == "done" else
                       (f'<a href="/job/{j.id}">progress</a>'
                        if j.status in ("waiting", "running") else "—"))
            + '</td></tr>'
            for j in jobs)
    return ((HEAD % ("RT playground — runs", CSS)) + f"""
<h1>Runs</h1>
<p class="sub">Every pipeline this server has executed since it started. Finished
reports are also on disk under <code>{OUT_ROOT}/</code>, so they outlive the
process.</p>
<div class="card"><table class="jobs">
<tr><th>id</th><th>upload</th><th>status</th><th>elapsed</th><th></th></tr>
{rows}</table></div>
<p class="foot"><a href="/">← new run</a></p>
</div></body></html>""").encode()


def page_error(exc: Exception, detail: str = "") -> bytes:
    return ((HEAD % ("RT playground — failed", CSS)) + f"""
<h1>That did not run</h1>
<div class="warn err">{html.escape(str(exc))}</div>
<div class="card"><h2>What to try</h2><p class="blurb">A GeoTIFF carries its own
georeference and needs nothing else. A <code>.npy</code> or <code>.png</code>
needs the bounding box it covers. A real Smart Terrain export whose ids are
sparse wire ids needs the class mapping JSON. If OSM could not be reached, untick
"refetch OSM now" to use a cached snapshot, or try again — Overpass is a free
service and returns 504 under load.</p>
{f'<pre>{html.escape(detail)}</pre>' if detail else ''}</div>
<p class="foot"><a href="/">← back</a></p>
</div></body></html>""").encode()


# --- server ----------------------------------------------------------------

def _mark_timed_out(job: Job, timeout: float) -> None:
    """Watchdog callback: fires once, `timeout` seconds after a job started
    running `run_pipeline`. Only touches the job if it is still "running" --
    if the pipeline finished (or failed on its own) first, this is a no-op.

    This reports the job as failed so `/job/<id>` and `/api/job/<id>` stop
    showing "running" forever. It does NOT reclaim the worker thread or
    release `_LOCK` -- if the thread is truly stuck inside a call that never
    returns to Python (e.g. `rasterio.open()` blocked on a hung FIFO), there
    is no safe way to do that from here. See the JOB_TIMEOUT_S comment.
    """
    if job.status == "running":
        job.status = "failed"
        job.error = (
            f"timed out after {timeout:.0f}s. The run is still occupying the "
            f"pipeline lock and may never release it -- if uploads keep "
            f"queuing behind this one, the server needs a restart."
        )
        print(f"  [{job.id}] TIMED OUT after {timeout:.0f}s (thread may "
              f"still be running; _LOCK is not released by this)",
              file=sys.stderr)


def _evict_old_jobs() -> None:
    """Bound the memory `_JOBS` holds via `Job.body`.

    `_JOBS` itself is never shrunk -- `/jobs` shows every run's history for
    the life of the process, and a `Job` without its body is a few hundred
    bytes. What actually costs memory is `body`: the full rendered HTML
    report, ~3.2 MB measured per run, and `_persist()` has already written
    the same bytes to `out_dir/index.html` by the time a job reaches this
    function. Past MAX_JOBS_RETAINED jobs, the oldest DONE/FAILED ones (by
    the time-ordered id -- see `_new_job`) have `body` dropped; `_report_bytes`
    reads the file back off disk for those. A "waiting" or "running" job is
    never touched.
    """
    with _JOBS_LOCK:
        if len(_JOBS) <= MAX_JOBS_RETAINED:
            return
        evictable = sorted(
            jid for jid, j in _JOBS.items()
            if j.status in ("done", "failed") and j.body is not None)
        overflow = len(_JOBS) - MAX_JOBS_RETAINED
        for jid in evictable[:overflow]:
            _JOBS[jid].body = None


def _run_job(job: Job, path: Path, opts: dict, cache_dir: Path,
             tmp: Path) -> None:
    """The worker thread. Never raises -- a failure is a job state, not a crash."""
    timeout = JOB_TIMEOUT_S  # read fresh each call so tests can monkeypatch it
    watchdog: threading.Timer | None = None
    try:
        if not _LOCK.acquire(blocking=False):
            job.step = "waiting for the current run to finish"
            print(f"  [{job.id}] queued behind another run", file=sys.stderr)
            # Block for at most `timeout` rather than forever: if whoever is
            # holding `_LOCK` is wedged, every job behind it would otherwise
            # queue in silence with no error and no way to tell what
            # happened. This job gives up and reports itself failed instead;
            # it does not free `_LOCK` for the ones still stuck behind it.
            if not _LOCK.acquire(timeout=timeout):
                job.status = "failed"
                job.error = (
                    f"gave up after waiting {timeout:.0f}s for the current "
                    f"run to finish -- it has been active longer than the "
                    f"{timeout:.0f}s timeout and is likely stuck. This job "
                    f"never started; try again once the stuck run clears, "
                    f"or restart the server."
                )
                print(f"  [{job.id}] FAILED: {job.error}", file=sys.stderr)
                return
        try:
            job.status = "running"
            # A second, independent safety net for the job that IS running:
            # if `run_pipeline` itself hangs, this marks the job failed at
            # the deadline even though the thread never returns. See
            # `_mark_timed_out` and the JOB_TIMEOUT_S comment for what this
            # does and does not fix.
            watchdog = threading.Timer(timeout, _mark_timed_out,
                                        args=(job, timeout))
            watchdog.daemon = True
            watchdog.start()
            run = run_pipeline(path, opts, cache_dir, job=job)
            body = page_results(run)
            job.body = body
            job.out_dir = _persist(job, run, body)
            # A watchdog may have already declared this job failed (it ran
            # past JOB_TIMEOUT_S but the call eventually returned anyway).
            # The report is still written to disk above either way; the
            # status the user was already shown is left as reported rather
            # than flipped back to "done" behind their back.
            if job.status != "failed":
                job.status = "done"
            print(f"  [{job.id}] done in {job.elapsed:.1f}s -> {job.out_dir}",
                  file=sys.stderr)
        finally:
            _LOCK.release()
    except (ValueError, FileNotFoundError, KeyError, ConnectionError) as exc:
        job.status, job.error = "failed", str(exc)
        print(f"  [{job.id}] FAILED: {exc}", file=sys.stderr)
    except Exception as exc:                          # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.detail = traceback.format_exc()
        print(f"  [{job.id}] CRASHED:\n{job.detail}", file=sys.stderr)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        job.finished = time.time()
        shutil.rmtree(tmp, ignore_errors=True)
        _evict_old_jobs()


def _persist(job: Job, run: Run, body: bytes) -> Path:
    """Write the finished report next to a plain-text copy of every panel.

    The whole reason this exists: the first version's only copy of a two-minute
    run was an HTTP response body, and browsers kept timing out before it was
    written. Now the response is a convenience.
    """
    out = OUT_ROOT / f"{job.id}-{re.sub(r'[^A-Za-z0-9._-]+', '_', job.name)[:60]}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_bytes(body)
    txt = [f"# {run.source} -- {run.seconds:.1f}s",
           *(f"{k}: {v}" for k, v in run.facts), ""]
    for w in run.warnings:
        txt.append(f"! {w}")
    for panel in run.panels:
        txt += ["", f"{'=' * 70}", f"{panel.key.upper()}  {panel.title}", "",
                panel.body]
    (out / "report.txt").write_text("\n".join(txt), encoding="utf-8")
    return out


def _report_bytes(job: Job) -> bytes | None:
    """The rendered report for a job, in memory if still resident, else read
    back from the copy `_persist()` wrote to disk.

    `_evict_old_jobs()` drops `job.body` for old finished jobs to bound
    memory; this is the read-side counterpart, so `/result/<id>` keeps
    serving a legitimately finished run instead of 404ing once its body has
    been evicted. Returns None only when there is truly nothing to serve
    (job never finished, or its files are gone).
    """
    if job.body is not None:
        return job.body
    if job.out_dir is not None:
        report = job.out_dir / "index.html"
        if report.exists():
            return report.read_bytes()
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cache_dir = Path(cache_mod.DEFAULT_CACHE_DIR)

    def log_message(self, fmt, *a):
        if not self.path.startswith("/api/"):        # polling is not news
            print(f"  {self.command} {self.path}", file=sys.stderr)

    def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8",
              code: int = 200):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The reader left. Nothing is lost -- the job and its files stand.
            pass

    def _redirect(self, where: str):
        try:
            self.send_response(303)
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _job(self, path_prefix: str) -> Job | None:
        jid = self.path[len(path_prefix):].strip("/").split("?")[0]
        with _JOBS_LOCK:
            return _JOBS.get(jid)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(page_upload())
        if self.path.startswith("/jobs"):
            return self._send(page_jobs())
        if self.path.startswith("/api/job/"):
            job = self._job("/api/job/")
            if job is None:
                return self._send(b'{"error":"no such job"}',
                                  "application/json", 404)
            return self._send(json.dumps(job.as_json()).encode(),
                              "application/json")
        if self.path.startswith("/job/"):
            job = self._job("/job/")
            if job is None:
                return self._send(page_upload("No such run.", True), code=404)
            if job.status == "done":
                return self._redirect(f"/result/{job.id}")
            return self._send(page_progress(job))
        if self.path.startswith("/result/"):
            job = self._job("/result/")
            body = _report_bytes(job) if job is not None else None
            if body is None:
                return self._send(page_upload(
                    "That run has no report yet.", True), code=404)
            return self._send(body)
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if not self.path.startswith("/run"):
            return self._send(b"not found", "text/plain", 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        # A negative Content-Length passed the `> MAX_UPLOAD_BYTES` test and then
        # `rfile.read(-1)` drained the socket to EOF -- the ceiling was the only
        # memory bound in the process, and one header removed it.
        if length < 0:
            return self._send(page_upload(
                "That request has no usable Content-Length.", True), code=411)
        if length > MAX_UPLOAD_BYTES:
            return self._send(page_upload(
                f"That upload is {length / 1e6:.0f} MB; the ceiling here is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB. Crop it, or point the CLI at "
                f"it directly -- `segmap solve s1 -i tile.tif --osm`.", True),
                code=413)
        body = self.rfile.read(length)
        try:
            parts, files = parse_multipart(body, self.headers.get("Content-Type", ""))
        except ValueError as exc:
            return self._send(page_error(exc), code=400)

        tmp = Path(tempfile.mkdtemp(prefix="rt-playground-"))
        try:
            return self._start(parts, files, tmp)
        except Exception as exc:                       # noqa: BLE001
            # Anything raising between mkdtemp and Thread.start() used to leak
            # the temp dir AND return no HTTP response at all -- socketserver
            # just closed the connection, so the browser showed "empty reply".
            # Reproduced with filename "..", an over-long dirpath, and
            # min_px=inf.
            shutil.rmtree(tmp, ignore_errors=True)
            return self._send(page_error(exc, traceback.format_exc()), code=400)

    def _start(self, parts, files, tmp: Path):
        dir_field = parts.get("dirpath")
        dir_text = dir_field.text if dir_field is not None else ""

        if files:
            # One upload is a tile; several are a mosaic. Both land in the same
            # temp directory and the directory becomes the input, so there is
            # exactly one downstream path.
            for n, f in enumerate(files):
                # `Path(x).name` stops real traversal, but ".." and "." survive
                # it and then IsADirectoryError escapes with no response.
                safe = Path(f.filename).name
                if safe in ("", ".", ".."):
                    safe = f"upload_{n}.tif"
                (tmp / safe).write_bytes(f.data)
            path = (tmp / Path(files[0].filename).name) if len(files) == 1 else tmp
            label = files[0].filename if len(files) == 1 else f"{len(files)} tiles"
        elif dir_text:
            path = Path(dir_text).expanduser()
            if not path.exists():
                shutil.rmtree(tmp, ignore_errors=True)
                return self._send(page_upload(
                    f"No such path on this machine: {path}", True), code=400)
            if not path.is_dir():
                shutil.rmtree(tmp, ignore_errors=True)
                return self._send(page_upload(
                    f"{path} is a file, not a directory. Upload it instead, or "
                    f"give the directory that contains it.", True), code=400)
            label = path.name + "/"
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            return self._send(page_upload(
                "Choose one or more rasters, or give a directory path.", True),
                code=400)
        try:
            opts = self._options(parts, tmp)
        except ValueError as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            return self._send(page_upload(str(exc), True), code=400)

        job = _new_job(label)
        threading.Thread(target=_run_job, name=f"rt-{job.id}",
                         args=(job, path, opts, self.cache_dir, tmp),
                         daemon=True).start()
        # Redirect at once. The pipeline outliving the request is the whole point:
        # a synchronous POST here is what produced BrokenPipeError six times.
        self._redirect(f"/job/{job.id}")

    def _options(self, parts: dict[str, Part], tmp: Path) -> dict:
        def num(name, default, cast=float):
            part = parts.get(name)
            if part is None or not part.text:
                return default
            try:
                return cast(part.text)
            except ValueError:
                return default

        bbox = None
        raw = parts.get("bbox")
        if raw is not None and raw.text:
            try:
                vals = [float(x) for x in re.split(r"[,\s]+", raw.text) if x]
                if len(vals) != 4:
                    raise ValueError
                bbox = tuple(vals)
            except ValueError:
                raise ValueError(
                    f"the bounding box needs four numbers, south,west,north,east "
                    f"in degrees -- got {raw.text!r}")

        classes_path = None
        cp = parts.get("classes")
        if cp is not None and cp.filename and cp.data:
            classes_path = tmp / "classes.json"
            classes_path.write_bytes(cp.data)

        return {
            "bbox": bbox,
            "gsd": num("gsd", None),
            # NaN is truthy and every comparison against it is False, so
            # `max_mpx=nan` removed the crop entirely -- the only bound on how
            # much raster the pipeline pulls into RAM.
            "max_mpx": _positive(num("max_mpx", DEFAULT_MAX_MPX),
                                 DEFAULT_MAX_MPX, "max_mpx"),
            "min_px": int(_positive(num("min_px", 12), 12, "min_px")),
            "chip": int(_positive(num("chip", 256), 256, "chip")),
            "policy": (parts["policy"].text if "policy" in parts else "settlement"),
            "vehicle": (parts["vehicle"].text if "vehicle" in parts else "wheeled"),
            "wet": "wet" in parts,
            "target": (parts["target"].text if "target" in parts else "person"),
            "refresh_osm": "refresh_osm" in parts,
            "budget": 15,
            "n_adjudicate": 4,
            "budget_frac": 0.2,
            "classes_path": str(classes_path) if classes_path else None,
        }


def serve(host: str = "127.0.0.1", port: int = 8011,
          cache_dir=None, open_browser: bool = False) -> None:
    Handler.cache_dir = Path(cache_dir or cache_mod.DEFAULT_CACHE_DIR)
    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"RT playground on {url}\n"
          f"  upload a segmentation raster; OSM is fetched live for its footprint\n"
          f"  runs happen in the background -- progress at {url}job/<id>, "
          f"history at {url}jobs\n"
          f"  every finished report is also written to {OUT_ROOT}/\n"
          f"  ctrl-c to stop", file=sys.stderr)
    if open_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        srv.server_close()
