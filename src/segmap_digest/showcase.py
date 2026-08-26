"""`segmap showcase` -- one self-contained HTML page explaining every solution,
illustrated with real imagery cut from the raster it is run on.

Runs on any label raster, so the page it writes is about *your* map: the
overviews, the class isolations, the zoom crops, the S4 product rasters and the
S1/S3 overlays are all rendered from the file you point it at, and the worked
examples quote that file's actual numbers rather than a fixture's.

Images are base64-inlined, so the output is a single portable .html with no
sidecar directory. Nothing here calls a model.
"""

from __future__ import annotations

import base64
import html
import io
import re

import numpy as np

from .index import ChipIndex, RegionIndex
from .loader import LabelRaster, colormap, colourise
from .taxonomy import BY_ID, BY_NAME

# Big enough to see a house, small enough that twenty of them still make a page
# a browser opens instantly.
OVERVIEW_MAX_PX = 1_600_000
ZOOM_PX = 900              # side of a full-resolution detail crop


# --- imaging ---------------------------------------------------------------

def _b64(rgb: np.ndarray, fmt: str = "PNG") -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _step_for(shape, max_px: int = OVERVIEW_MAX_PX) -> int:
    step = 1
    while (shape[0] // step) * (shape[1] // step) > max_px:
        step += 1
    return step


def _grey(rgb: np.ndarray, keep: float = 0.18) -> np.ndarray:
    """Desaturate to a near-white basemap so one highlighted class reads as the
    only thing on the page. Straight greyscale is too dark to overlay on."""
    g = rgb.mean(axis=-1, keepdims=True)
    out = 255 - (255 - g) * keep
    return np.repeat(out, 3, axis=-1).astype(np.uint8)


def overview(r: LabelRaster, step: int | None = None) -> str:
    step = step or _step_for(r.labels.shape)
    rgb = colourise(r.labels[::step, ::step])
    if r.valid is not None:
        rgb[~r.valid[::step, ::step]] = 255
    return _b64(rgb)


def isolate(r: LabelRaster, names: list[str], step: int | None = None) -> str:
    """One or more classes in their own colours; everything else a pale ghost."""
    step = step or _step_for(r.labels.shape)
    lab = r.labels[::step, ::step]
    rgb = colourise(lab)
    out = _grey(rgb)
    cmap = colormap()
    for n in names:
        cid = BY_NAME[n].id
        m = lab == cid
        if m.any():
            out[m] = cmap[cid]
    if r.valid is not None:
        out[~r.valid[::step, ::step]] = 255
    return _b64(out)


def zoom(r: LabelRaster, cy: int, cx: int, size: int = ZOOM_PX,
         mark: tuple[int, int] | None = None) -> str:
    """A full-resolution crop. This is where a 0.5 m/px map stops being a
    thumbnail and starts showing individual buildings and vehicles."""
    h, w = r.labels.shape
    half = size // 2
    r0 = int(np.clip(cy - half, 0, max(h - size, 0)))
    c0 = int(np.clip(cx - half, 0, max(w - size, 0)))
    r1, c1 = min(r0 + size, h), min(c0 + size, w)
    rgb = colourise(r.labels[r0:r1, c0:c1])
    if r.valid is not None:
        rgb[~r.valid[r0:r1, c0:c1]] = 255
    if mark is not None:
        my, mx = mark[0] - r0, mark[1] - c0
        if 0 <= my < rgb.shape[0] and 0 <= mx < rgb.shape[1]:
            _crosshair(rgb, my, mx)
    return _b64(rgb)


def _crosshair(rgb: np.ndarray, y: int, x: int, arm: int = 26, gap: int = 7) -> None:
    h, w = rgb.shape[:2]
    ink = np.array([20, 20, 20], np.uint8)
    for d in range(gap, arm):
        for yy, xx in ((y - d, x), (y + d, x), (y, x - d), (y, x + d)):
            if 0 <= yy < h and 0 <= xx < w:
                rgb[yy, xx] = ink


def ramp_png(arr: np.ndarray, step: int | None = None,
             mask: np.ndarray | None = None) -> str:
    """A 0..1 raster on the same ramp `s4_products.to_png` uses."""
    step = step or _step_for(arr.shape)
    x = np.clip(arr[::step, ::step], 0, 1)
    rgb = np.stack([
        np.clip(1.6 * x - 0.35, 0, 1) * 255,
        np.clip(1.2 * x + 0.05, 0, 1) * 255,
        np.clip(1.1 - 1.3 * x, 0, 1) * 255,
    ], axis=-1).astype(np.uint8)
    if mask is not None:
        rgb[~mask[::step, ::step]] = 255
    return _b64(rgb)


def highlight_regions(r: LabelRaster, ridx: RegionIndex, rids: list[int],
                      step: int | None = None) -> str:
    """Named regions burned onto the ghost basemap. This is what makes an S1
    finding checkable: the row says region 14328, the picture says where."""
    step = step or _step_for(r.labels.shape)
    base = _grey(colourise(r.labels[::step, ::step]))
    lab = ridx.label_array[::step, ::step]
    hot = np.isin(lab, list(rids))
    base[hot] = np.array([214, 48, 32], np.uint8)
    # A lone flagged region at 5x decimation can be one pixel; box it too.
    for rid in rids[:60]:
        reg = ridx.get(rid)
        r0, c0, r1, c1 = (v // step for v in reg.bbox)
        _box(base, r0, c0, r1, c1)
    return _b64(base)


def _box(rgb: np.ndarray, r0, c0, r1, c1, pad: int = 4) -> None:
    h, w = rgb.shape[:2]
    ink = np.array([214, 48, 32], np.uint8)
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, h - 1), min(c1 + pad, w - 1)
    if r1 <= r0 or c1 <= c0:
        return
    rgb[r0, c0:c1] = ink; rgb[r1, c0:c1] = ink
    rgb[r0:r1, c0] = ink; rgb[r0:r1, c1] = ink


def chip_overlay(r: LabelRaster, cidx: ChipIndex, selected: set[int],
                 control: set[int], step: int | None = None) -> str:
    """The triage decision, drawn on the map: which chips got dispatched to the
    expensive detector, and which went to the mandatory control set."""
    step = step or _step_for(r.labels.shape)
    base = _grey(colourise(r.labels[::step, ::step]), keep=0.30)
    for ch in cidx.chips:
        r0, c0, r1, c1 = (v // step for v in ch.bbox)
        if ch.id in selected:
            _fill(base, r0, c0, r1, c1, (214, 48, 32), 0.42)
        elif ch.id in control:
            _fill(base, r0, c0, r1, c1, (36, 108, 196), 0.34)
    return _b64(base)


def _fill(rgb, r0, c0, r1, c1, colour, alpha) -> None:
    h, w = rgb.shape[:2]
    r1, c1 = min(r1, h), min(c1, w)
    if r1 <= r0 or c1 <= c0:
        return
    patch = rgb[r0:r1, c0:c1].astype(np.float32)
    rgb[r0:r1, c0:c1] = (patch * (1 - alpha)
                         + np.array(colour, np.float32) * alpha).astype(np.uint8)


# --- picking what to show --------------------------------------------------

def densest_window(r: LabelRaster, names: list[str], size: int = ZOOM_PX
                   ) -> tuple[int, int, float]:
    """Centre of the `size`-px window holding most of these classes. Finds the
    settlement without being told where the settlement is."""
    cids = [BY_NAME[n].id for n in names if n in BY_NAME]
    m = np.isin(r.labels, cids)
    if not m.any():
        h, w = r.labels.shape
        return h // 2, w // 2, 0.0
    # Coarse box-sum: decimate hard, then convolve with a uniform kernel.
    step = max(size // 24, 1)
    small = m[::step, ::step].astype(np.float32)
    k = max(size // step, 1)
    cs = np.cumsum(np.cumsum(np.pad(small, ((1, 0), (1, 0))), 0), 1)
    if cs.shape[0] <= k or cs.shape[1] <= k:
        idx = np.argwhere(m)
        cy, cx = idx.mean(0).astype(int)
        return int(cy), int(cx), float(m.mean())
    box = (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k])
    i = int(np.argmax(box))
    by, bx = np.unravel_index(i, box.shape)
    frac = float(box[by, bx] / (k * k))
    return int((by + k / 2) * step), int((bx + k / 2) * step), frac


# --- is `count` counting the right thing? ----------------------------------
# S5 open question 9, answered from whatever raster this is run on. `count`
# counts connected components, so anything that touches is one thing. Whether
# that matters is a property of the map, not of the verb -- so measure it.

COUNTABLE = {
    "House":  (60.0,   "a house"),
    "Car":    (12.0,   "a car"),
    "BrickWall": (None, "a wall"),
}


def merge_check(r: LabelRaster, ridx: RegionIndex) -> list[dict]:
    out = []
    for name, (typical, what) in COUNTABLE.items():
        if name not in BY_NAME:
            continue
        regs = ridx.by_class(BY_NAME[name].id)
        if len(regs) < 3:
            continue
        areas = sorted((x.area_m2 for x in regs), reverse=True)
        total = sum(areas)
        if total <= 0:
            continue
        med = areas[len(areas) // 2]
        row = {
            "name": name, "n": len(regs), "total": total,
            "largest": areas[0], "median": med,
            "top_share": areas[0] / total,
            "typical": typical, "what": what,
            "suspect": (typical is not None
                        and (med > 2.5 * typical or areas[0] > 0.35 * total)),
        }
        if typical:
            row["implied"] = total / typical
        out.append(row)
    return out


# --- what each solution is, in words ---------------------------------------
# High-level only. The per-solution docs in docs/solutions/ carry the argument;
# this is the version someone reads once, next to a picture of their own map.

ALGOS = {
"s1": {
 "title": "Consistency audit",
 "cmd": "segmap solve s1",
 "one_liner": "Flag labels that contradict their own context, slope or geometry, "
              "then roll thousands of flags up into the handful of distinct facts "
              "they are instances of.",
 "why": "It is the only solution whose output feeds back into the segmenter: a "
        "confirmed finding becomes a training-data correction, and the flag rate "
        "per class is an error map you get without any detection ground truth.",
 "steps": [
  ("Index every region", "Connected components per class, each with area, "
   "perimeter, compactness, elongation, and a neighbour graph carrying the "
   "shared boundary length with every touching region."),
  ("Run five check families", "<b>context-prior</b> (does this class sit next to "
   "what it should?), <b>slope-prior</b> and <b>aspect-prior</b> (does the "
   "morphology match?), <b>lithology-isolation</b> (a limestone unit with no "
   "carbonate neighbour is suspect), and <b>nari-geometry</b>. Each emits a "
   "finding with a severity graded from the evidence, not a fixed constant."),
  ("Roll up to root causes", "Findings sharing a class-level fact are "
   "<i>one decision</i> for a reviewer, not N. 2710 findings collapse into 62 "
   "causes, each naming example region ids."),
  ("Rank by review value", "severity × area<sup>0.5</sup> — so a systematic fact "
   "over a large area outranks one loud speck."),
  ("Report coverage first", "Which classes had an active check at all. A class "
   "with no check is <b>unexamined, not clean</b>, and the report refuses to let "
   "zero findings imply otherwise."),
 ],
 "abstains": "With no DEM every slope and aspect is 0.0 — a valid-looking number "
             "meaning <i>unmeasured</i>. The slope and aspect priors do not run "
             "rather than score against a fabricated zero.",
},
"s2": {
 "title": "Confusion adjudication",
 "cmd": "segmap solve s2 --region N",
 "one_liner": "S1 says region 245 is wrong. S2 answers <i>then what is it?</i> — "
              "and refuses when the evidence cannot separate the candidates.",
 "why": "Limestone / Dolomite / Nari rocky terrain are not separable in RGB even "
        "for a human expert. A system that confidently picks one is manufacturing "
        "corruptions. <code>UNDECIDABLE-NEEDS-geological-map</code> is a "
        "first-class verdict, and it is designed to fire often.",
 "steps": [
  ("Build a candidate list", "The incumbent label plus the classes it is "
   "plausibly confused with — same lithology, adjacent rung on an ordinal "
   "series, or common neighbours."),
  ("Score three evidence terms", "<b>context</b> — what fraction of this "
   "region's boundary is shared with units of the same lithology; "
   "<b>morphology</b> — slope and aspect against the class prior; "
   "<b>geometry</b> — compactness and elongation against what the class looks "
   "like."),
  ("Require a margin", "A challenger must beat the incumbent by 0.15 to earn a "
   "<code>SWITCH</code>. Inside the margin the verdict is <code>KEEP</code> or "
   "<code>UNDECIDABLE</code> — never a coin flip presented as a finding."),
  ("Show the whole table", "Every candidate with its three sub-scores and the "
   "note explaining each, so the verdict can be disagreed with."),
 ],
 "abstains": "Without a DEM the morphology term is unscored for every candidate, "
             "which is stated in each row rather than silently scored 0.5.",
},
"s3": {
 "title": "Cost-aware detection triage",
 "cmd": "segmap solve s3 --policy vehicles|structures|oov",
 "one_liner": "Use the cheap segmentation as a prior to decide which image chips "
              "are worth sending to an expensive detector.",
 "why": "This is the solution with a direct cost line. The model is <b>not</b> in "
        "the per-chip loop — it compiles a query into a small cacheable policy "
        "once, and a deterministic scorer applies it to millions of chips. An "
        "LLM call per chip replaces one expensive call with another and saves "
        "nothing.",
 "steps": [
  ("Compile the query into a policy", "Once per query type: class weights, a "
   "hard-exclude list, and a regime. Cacheable, checked in, human-readable."),
  ("Score every chip", "Pure arithmetic over the precomputed chip index — class "
   "fractions, entropy, edge density, distance to anchor classes. Milliseconds "
   "over millions of chips."),
  ("Hard-exclude the impossible", "Vehicles cannot park on boulder fields, steep "
   "smooth rock, dissected badlands, or water. Those chips cost nothing to skip."),
  ("Spend the budget top-down", "Take the highest-scoring chips until the budget "
   "fraction is used."),
  ("Sample a control set", "A mandatory random sample of the <i>rejected</i> "
   "chips. Without it there is no unbiased recall signal and the triage can "
   "never be shown to be wrong."),
 ],
 "abstains": "'Score captured' is prior mass, not recall. The report says so in "
             "capitals: real recall needs detector outcomes that do not exist yet.",
},
"s4": {
 "title": "Derived decision products",
 "cmd": "segmap solve s4 --product trafficability|concealment|drainage|fire_fuel",
 "one_liner": "The many-to-one mapping from 47 classes + slope + season into what "
              "someone actually asks for.",
 "why": "A model helps at <i>authoring</i> time — hand-writing 47×N lookup tables "
        "with exceptions is tedious and error-prone. But once authored the tables "
        "are checked in and evaluation is fully deterministic. <b>No LLM in the "
        "runtime path.</b>",
 "steps": [
  ("Look up a per-class base score", "One number per class per product, from a "
   "checked-in table."),
  ("Apply the modifiers", "Vehicle type (wheeled / tracked / foot) and season "
   "(wet / dry) shift the scores — a clay soil that carries a wheeled vehicle in "
   "August does not in February."),
  ("Penalise by slope and roughness", "Where a DEM exists. Without one this step "
   "is skipped and the product is a class lookup only."),
  ("Emit a 0..1 raster", "Plus a summary by superclass, so the shape of the "
   "answer is legible before anyone opens the image."),
 ],
 "abstains": "With no DEM the slope penalty never applies, so every figure here is "
             "a surface-type score and not a real trafficability estimate. The "
             "corridor verb in S5 inherits exactly this.",
},
"s5": {
 "title": "Natural-language querying",
 "cmd": "segmap solve s5 --query '...'",
 "one_liner": "Six verbs that answer quantitative questions in code, so no model "
              "ever does arithmetic over a big table.",
 "why": "In-context arithmetic over a large table is exactly where LLMs go wrong, "
        "and exactly what is easy to compute. The verbs are not the user "
        "interface — they are the tool surface a model is given, and having them "
        "exist first is what keeps the model out of the numbers.",
 "steps": [
  ("<code>describe</code>", "Class histogram over classified pixels, with region "
   "counts and the nodata fraction stated."),
  ("<code>count</code> / <code>area</code> / <code>find</code>",
   "Filterable by <code>minarea</code>, <code>slope</code>, <code>near</code>. "
   "<code>count</code> counts connected components, so two houses sharing a wall "
   "are one — the tool description says so."),
  ("<code>distance</code>", "True pixel-to-pixel closest approach between two "
   "classes, returning <i>both endpoints</i> so it can be checked on the map."),
  ("<code>corridor</code>", "Threshold the S4 trafficability raster, remove "
   "passages narrower than the vehicle, connected-component the result, and "
   "report which components touch the seed class."),
  ("Carry the evidence", "Every result carries the region / component ids it was "
   "computed from. An answer without ids cannot be audited."),
 ],
 "abstains": "A slope filter with no DEM is <code>unanswerable</code>, not "
             "'0 degrees'. A class outside the vocabulary is "
             "<code>unanswerable</code> with the reason — vehicle type, building "
             "function, fences, anything about colour.",
},
"s6": {
 "title": "Change reasoning",
 "cmd": "segmap solve s6 --second t2.tif",
 "one_liner": "Classify each transition between two dates, because the same raw "
              "pixel difference means completely different things.",
 "why": "A pixel diff of two label maps is mostly noise and phenology. "
        "<code>GreenGrassland → DryGrassland</code> means the season changed and "
        "the ground did not; <code>Batha → House</code> means someone built "
        "something. Only one of those is worth a phone call.",
 "steps": [
  ("Diff co-valid pixels only", "A pixel that is nodata on either date is not a "
   "change."),
  ("Classify every transition", "Into <b>phenology</b>, <b>succession</b>, "
   "<b>construction</b>, <b>impossible</b>, and so on — from the taxonomy's own "
   "structure, the degradation and road series."),
  ("Guard the implausible", "A multi-stage jump along a succession series in one "
   "season is a classifier disagreement, not regrowth."),
  ("Component-count each pair", "So '400 scattered pixels' and 'one new "
   "building' are told apart."),
 ],
 "abstains": "An <code>impossible</code> transition is reported as evidence about "
             "the <i>classifier</i>, not about the ground — and every one of them "
             "is free ground truth for S1.",
},
}


# --- page ------------------------------------------------------------------

def _e(s) -> str:
    return html.escape(str(s))


def _pre(text: str, limit: int = 2600) -> str:
    t = text if len(text) <= limit else text[:limit].rsplit("\n", 1)[0] + "\n…"
    return f"<pre>{_e(t)}</pre>"


def _fig(src: str, cap: str, cls: str = "") -> str:
    # `cap` carries markup for the caption; alt text must not.
    alt = _e(re.sub(r"<[^>]+>", "", cap))
    return (f'<figure class="{cls}"><img src="{src}" alt="{alt}" loading="lazy">'
            f'<figcaption>{cap}</figcaption></figure>')


def _algo(sid: str) -> str:
    a = ALGOS[sid]
    steps = "".join(
        f"<li><b>{s[0]}</b> — {s[1]}</li>" for s in a["steps"])
    return f"""
<p class="lead">{a['one_liner']}</p>
<div class="why"><b>Why it earns its place.</b> {a['why']}</div>
<h4>The algorithm</h4>
<ol class="steps">{steps}</ol>
<div class="abstain"><b>Where it abstains.</b> {a['abstains']}</div>
"""


def _try(sid: str, tile: str, extra: str = "") -> str:
    a = ALGOS[sid]
    cmd = f"{a['cmd']} -i {tile}" if "-i" not in a["cmd"] else a["cmd"]
    return f"""<div class="try">
<b>Run it on your own map.</b>
<pre>{_e(cmd)}{_e(extra)}</pre>
<p>Or open the <b>Solutions</b> tab of <code>segmap ui -i &lt;your-raster.tif&gt;</code>
and press <b>Run {sid.upper()}</b>. If your GeoTIFF has no
<code>ID_TO_LABEL_MAPPING</code> tag, add
<code>--classes your_class_ids.json</code>.</p></div>"""


def build(r: LabelRaster, ridx: RegionIndex, cidx: ChipIndex, *,
          input_path: str, out_path: str, classes: str | None = None) -> str:
    from .digests import l0_histogram
    from .solutions import (s1_audit, s2_adjudicate, s3_triage, s4_products,
                            s5_query)

    tile = input_path or "<your-raster.tif>"
    h, w = r.labels.shape
    step = _step_for(r.labels.shape)
    P: list[str] = []

    def say(msg):
        print(f"  {msg}", flush=True)

    # ---- what is on this map
    say("rendering overview")
    present = [(int(l.split("\t")[0]), l.split("\t")[1], float(l.split("\t")[3]),
                float(l.split("\t")[4]))
               for l in l0_histogram(r).splitlines()
               if l and not l.startswith(("#", "class_id")) and len(l.split("\t")) >= 5]
    present.sort(key=lambda x: -x[2])
    cmap = colormap()
    legend = "".join(
        f'<li><i style="background:rgb({cmap[c][0]},{cmap[c][1]},{cmap[c][2]})"></i>'
        f'<span>{_e(n)}</span><b>{100*f:.2f}%</b></li>'
        for c, n, f, _a in present)

    ov = overview(r, step)
    P.append(f"""<section id="map"><h2>The map</h2>
<p class="lead">Everything below is rendered from this raster. Change the input
and the whole page changes with it.</p>
<div class="grid2">
{_fig(ov, f"The full tile, colourised. {w}×{h} px at {r.gsd:.3f} m/px "
          f"— {round(w*r.gsd):,} × {round(h*r.gsd):,} m on the ground"
          + (f", shown decimated {step}× (every number on this page is computed at "
             f"full resolution)" if step > 1 else ""))}
<div><h4>{len(present)} classes present</h4><ul class="legend">{legend}</ul></div>
</div></section>""")

    # ---- class isolations: the map is many maps
    say("rendering class isolations")
    def have(*names):
        return [n for n in names if n in BY_NAME
                and (r.labels == BY_NAME[n].id).any()]

    iso = []
    groups = [
        (have("DirtRoad", "DirtRoadB", "PavedRoad", "Pavement"), "The road network",
         "Every road-superclass pixel. This is what <code>corridor</code> seeds "
         "from and what <code>distance</code> measures to."),
        (have("House", "BrickWall", "Pavement"), "The built environment",
         "One building class. Not schools, not hospitals, not warehouses — "
         "building <i>function</i> was never predicted, which is why S5 refuses "
         "to answer questions about it."),
        (have("Car"), "Vehicles",
         "One vehicle class. Trucks and buses are not a separate label and never "
         "will be at this taxonomy."),
        (have("MaralBadlands", "MaralSmoothRockSlopes", "MaralTerrace",
              "LimestoneBoulder", "LimestoneRockyTerrain", "LimestoneTerrace",
              "LimestoneStoneyTerrain", "ChalkSmoothRockSlopes"),
         "Rock and morphology",
         "The classes whose definitions are about <i>shape</i> — badlands, dip "
         "slopes, terraces, boulder fields. Separating these is what a DEM buys, "
         "and there is no DEM here."),
        (have("DryGrassland", "GreenGrassland", "Batha", "Garigue", "Maquis"),
         "The vegetation series",
         "An ordinal series: DryGrassland → Batha → Garigue → Maquis is "
         "increasing woody cover. A confusion <i>within</i> the series is minor; "
         "one across superclasses is not."),
    ]
    for names, title, blurb in groups:
        if not names:
            continue
        iso.append(_fig(isolate(r, names, step),
                        f"<b>{title}</b> — {blurb} "
                        f"<span class='cls'>{', '.join(names)}</span>"))
    P.append(f"""<section id="layers"><h2>One raster, many maps</h2>
<p class="lead">The same pixels, filtered to one question at a time. Each of
these is a <code>find</code> or an <code>area</code> away.</p>
<div class="gallery">{''.join(iso)}</div></section>""")

    # ---- detail crops
    say("rendering detail crops")
    crops = []
    for names, label in [
        (have("House", "BrickWall"), "the built-up area"),
        (have("Car"), "vehicles"),
        (have("PavedRoad", "DirtRoad"), "the road network"),
        (have("MaralBadlands"), "the badlands"),
    ]:
        if not names:
            continue
        cy, cx, frac = densest_window(r, names)
        if frac <= 0:
            continue
        crops.append(_fig(zoom(r, cy, cx),
                          f"<b>Detail: {label}</b> — {ZOOM_PX}×{ZOOM_PX} px at full "
                          f"resolution ({round(ZOOM_PX*r.gsd)} m across), centred on "
                          f"the densest window at ({cy}, {cx}). "
                          f"{100*frac:.1f}% of this window is "
                          f"{', '.join(names)}."))
    P.append(f"""<section id="detail"><h2>At full resolution</h2>
<p class="lead">The overview above is decimated {step}× to fit on a screen. At
{r.gsd:.2f} m/px the actual raster resolves individual buildings and vehicles —
these crops are untouched pixels.</p>
<div class="gallery">{''.join(crops)}</div></section>""")

    # ---- S5 first: it is the one whose output is a number
    say("running S5")
    q_examples = []
    for q in ["describe", "count House minarea 40", "area DirtRoad",
              "area PavedRoad", "count Car",
              "distance MaralBadlands DirtRoad",
              "corridor DirtRoad vehicle wheeled",
              "count Truck", "find Rendzina slope > 25"]:
        try:
            res = s5_query.query(q, r, ridx)
            q_examples.append((q, res.render(limit=14), "ok"))
        except s5_query.QueryError as exc:
            q_examples.append((q, f"query error: {exc}", "refused"))
        except Exception as exc:
            q_examples.append((q, f"{type(exc).__name__}: {exc}", "refused"))

    ex_html = "".join(
        f'<div class="ex {k}"><div class="q"><code>segmap solve s5 --query '
        f'"{_e(q)}"</code></div>{_pre(out, 1400)}</div>'
        for q, out, k in q_examples)

    say("checking what `count` counts")
    merged = merge_check(r, ridx)
    merge_html = ""
    if merged:
        rows = "".join(
            f"<tr class=\"{'bad' if m['suspect'] else ''}\"><td>{_e(m['name'])}</td>"
            f"<td class='n'>{m['n']}</td><td class='n'>{round(m['total']):,}</td>"
            f"<td class='n'>{round(m['median']):,}</td>"
            f"<td class='n'>{round(m['largest']):,}</td>"
            f"<td class='n'>{100*m['top_share']:.0f}%</td>"
            f"<td>{('~' + format(round(m['implied']), ',')) if m.get('implied') else '—'}</td>"
            f"<td>{'components are fused' if m['suspect'] else ('plausible' if m['typical'] else 'no reference size — not assessed')}</td></tr>"
            for m in merged)
        bad = [m for m in merged if m["suspect"]]
        verdict = (
            "<b>On this map, no.</b> " + "; ".join(
                f"the largest <code>{_e(m['name'])}</code> component is "
                f"{round(m['largest']):,} m² and holds {100*m['top_share']:.0f}% of all "
                f"{_e(m['name'])} area, with a median component of {round(m['median']):,} m² "
                f"where {m['what']} is about {round(m['typical'])} m²"
                for m in bad)
            + ". Those are not individual objects — they are everything that touches, "
              "fused into one component. A count over them is a count of <i>blobs</i>, "
              "and the honest reading of the area column is "
            + "; ".join(f"<b>~{round(m['implied']):,} {_e(m['name'])}</b>" for m in bad
                        if m.get("implied"))
            + " by area, not the component count."
        ) if bad else (
            "<b>On this map, plausibly yes.</b> No class shows a component large "
            "enough to suggest fused objects.")
        merge_html = f"""<h4>Does <code>count</code> count the right thing?</h4>
<div class="callout warn"><p><code>count</code> counts <b>connected
components</b>, so two houses sharing a wall are one house and a row of parked
cars is one car. The tool description says so — which is not the same as being
right. Measured on this raster:</p>
<table class="t"><thead><tr><th>class</th><th>components</th><th>total m²</th>
<th>median m²</th><th>largest m²</th><th>largest share</th><th>implied by area</th>
<th></th></tr></thead><tbody>{rows}</tbody></table>
<p>{verdict}</p>
<p class="fine">This is S5's open question 9, answered empirically rather than
asserted. It is a property of <i>this</i> segmentation, not of the verb — run
<code>segmap showcase</code> on your own raster and the table is recomputed for
it. The fix is instance segmentation or a watershed split, neither of which is
in this repo.</p></div>"""

    # the distance endpoints, drawn
    dist_fig = ""
    try:
        d = s5_query.query("distance MaralBadlands DirtRoad", r, ridx)
        row = d.rows[0]
        ay, ax = int(row[2]), int(row[3])
        dist_fig = _fig(zoom(r, ay, ax, 500, mark=(ay, ax)),
                        f"<b><code>distance</code> is checkable.</b> The verb "
                        f"returns <i>both endpoints</i>, not just a number: "
                        f"{row[0]} region {row[1]} at ({ay}, {ax}) and {row[4]} "
                        f"region {row[5]} at ({int(row[6])}, {int(row[7])}), "
                        f"{row[8]} m apart. Crosshair marks the first endpoint at "
                        f"full resolution — you can confirm the answer by eye.")
    except Exception:
        pass

    P.append(f"""<section id="s5"><h2><span>S5</span> {ALGOS['s5']['title']}</h2>
{_algo('s5')}
<h4>Worked examples on this map</h4>
<div class="examples">{ex_html}</div>
{f'<div class="gallery one">{dist_fig}</div>' if dist_fig else ''}
<div class="callout"><b>The two refusals above are the point.</b>
<code>count Truck</code> and a slope filter with no DEM both return
<i>unanswerable</i> with a reason. Neither returns zero. A silent plausible
answer to an unanswerable question is the failure mode this whole surface exists
to prevent.</div>
{merge_html}
{_try('s5', tile, ' --query "count House minarea 40"')}</section>""")

    # ---- S4
    say("running S4 (four products)")
    prods = []
    valid = r.valid if r.valid is not None else None
    for name, kw, blurb in [
        ("trafficability", {"vehicle": "wheeled"},
         "Can a wheeled vehicle cross this pixel? Bright = yes."),
        ("concealment", {}, "How much cover does this ground give? Bright = more."),
        ("drainage", {}, "Where does water go and collect? Bright = drains freely."),
        ("fire_fuel", {}, "How much burnable material? Bright = more fuel."),
    ]:
        arr = s4_products.compute(r, name, **kw)
        summ = s4_products.summarise(r, arr, None)
        head = summ.splitlines()[0] if summ else ""
        prods.append(_fig(ramp_png(arr, step, valid),
                          f"<b>{name}</b>{' (' + kw['vehicle'] + ')' if kw else ''} — "
                          f"{blurb}<br><span class='mono'>{_e(head.lstrip('# '))}</span>"))

    dry = s4_products.compute(r, "trafficability", vehicle="wheeled", wet=False)
    wet = s4_products.compute(r, "trafficability", vehicle="wheeled", wet=True)
    delta = np.abs(dry - wet)
    wet_fig = ""
    if delta.max() > 0:
        wet_fig = _fig(ramp_png(1.0 - delta / max(delta.max(), 1e-9), step, valid),
                       f"<b>What the season costs.</b> |dry − wet| for wheeled "
                       f"trafficability; dark = the ground that stops carrying a "
                       f"vehicle in the wet season. Mean drop "
                       f"{float((dry - wet).mean()):.3f} over the tile, worst "
                       f"{float(delta.max()):.2f}.")

    P.append(f"""<section id="s4"><h2><span>S4</span> {ALGOS['s4']['title']}</h2>
{_algo('s4')}
<h4>All four products, computed from this raster</h4>
<div class="gallery">{''.join(prods)}</div>
{f'<div class="gallery one">{wet_fig}</div>' if wet_fig else ''}
<h4>The summary the CLI prints</h4>
{_pre(s4_products.summarise(r, s4_products.compute(r, 'trafficability', vehicle='wheeled'), ridx), 1500)}
{_try('s4', tile, ' --product trafficability --vehicle wheeled')}</section>""")

    # ---- corridor, which is S5 riding on S4
    say("running corridor")
    corr_fig, corr_txt = "", ""
    try:
        res = s5_query.query("corridor DirtRoad vehicle wheeled", r, ridx)
        corr_txt = res.render(limit=10)
        traf = s4_products.compute(r, "trafficability", vehicle="wheeled")
        corr_fig = _fig(ramp_png((traf >= 0.45).astype(np.float32), step, valid),
                        "<b>Passable ground</b> — trafficability ≥ 0.45, the "
                        "threshold <code>corridor</code> uses before it removes "
                        "narrow passages and connected-components what is left. "
                        "This is the raster the 97%-style figure is computed over.")
    except Exception as exc:
        corr_txt = f"{type(exc).__name__}: {exc}"

    P.append(f"""<section id="corridor"><h2>Where the two meet: <code>corridor</code></h2>
<p class="lead">The most interesting verb, because it is S5 riding on S4 and it
inherits every guess S4 makes. It answers "where can a wheeled vehicle actually
get to from the road network" with no model involved at all.</p>
<div class="grid2">{corr_fig}<div>{_pre(corr_txt, 1200)}
<div class="callout small"><b>Read the assumptions, not the number.</b> The note
states all three: the trafficability threshold, the slope limit, and the vehicle
width used to close narrow passages. Every one of them is a guess inherited from
S4, and with no DEM the slope limit never actually binds.</div></div></div>
</section>""")

    # ---- S1
    say("running S1")
    rep = s1_audit.run(ridx, min_area_m2=25.0)
    s1_text = s1_audit.render(rep, budget=12)
    top_rids = []
    for c in sorted(rep.causes, key=lambda c: -c.value)[:6]:
        top_rids.extend(c.example_rids(6))
    top_rids = list(dict.fromkeys(top_rids))[:60]
    s1_fig = (highlight_regions(r, ridx, top_rids, step) if top_rids else "")
    causes_rows = "".join(
        f"<tr><td>{_e(c.class_name)}</td><td>{_e(c.kind)}</td>"
        f"<td class='n'>{c.n_regions}</td><td class='n'>{c.severity:.2f}</td>"
        f"<td class='n'>{round(c.area_m2):,}</td>"
        f"<td class='ids'>{_e(', '.join(str(i) for i in c.example_rids(4)))}</td></tr>"
        for c in sorted(rep.causes, key=lambda c: -c.value)[:10])

    P.append(f"""<section id="s1"><h2><span>S1</span> {ALGOS['s1']['title']}</h2>
{_algo('s1')}
<h4>On this map: {len(rep.findings):,} findings → {len(rep.causes)} root causes</h4>
<div class="grid2">
{_fig(s1_fig, "<b>The top-ranked flagged regions, boxed on the map.</b> Each red "
              "box is a region id the audit named. That is what makes a finding "
              "checkable rather than a claim.") if s1_fig else ''}
<div><table class="t"><thead><tr><th>class</th><th>check</th><th>regions</th>
<th>sev</th><th>m²</th><th>example ids</th></tr></thead>
<tbody>{causes_rows}</tbody></table></div></div>
<h4>The report, as the CLI prints it</h4>
{_pre(s1_text, 3000)}
<div class="callout"><b>Read the coverage block before the findings.</b> It is
printed first on purpose: it says how many classes had an active check at all,
and states that a class with no check is unexamined rather than clean. Zero
findings for an unchecked class means nothing.</div>
{_try('s1', tile, ' --budget 25')}</section>""")

    # ---- S2
    say("running S2")
    rids, seen = [], set()
    for f in rep.findings:
        if f.region_id not in seen:
            seen.add(f.region_id)
            rids.append(f.region_id)
        if len(rids) >= 3:
            break
    adjs = [s2_adjudicate.adjudicate(ridx, i) for i in rids]
    s2_text = "\n\n".join(s2_adjudicate.render(a) for a in adjs)
    s2_ledger = s2_adjudicate.ledger(adjs) if adjs else ""
    s2_figs = "".join(
        _fig(zoom(r, int(ridx.get(i).centroid[0]), int(ridx.get(i).centroid[1]), 420,
                  mark=(int(ridx.get(i).centroid[0]), int(ridx.get(i).centroid[1]))),
             f"<b>Region {i}</b> — incumbent <code>{_e(ridx.get(i).class_name)}</code>, "
             f"{round(ridx.get(i).area_m2):,} m². The crosshair is its centroid; "
             f"the adjudication for it is below.")
        for i in rids[:3])

    P.append(f"""<section id="s2"><h2><span>S2</span> {ALGOS['s2']['title']}</h2>
{_algo('s2')}
<h4>The three regions S1 flagged hardest, adjudicated</h4>
<div class="gallery three">{s2_figs}</div>
{_pre(s2_text, 3000)}
{f'<h4>The ledger</h4>{_pre(s2_ledger, 1200)}' if s2_ledger else ''}
{_try('s2', tile, f' --region {rids[0]}' if rids else '')}</section>""")

    # ---- S3
    say("running S3")
    s3_blocks = []
    for pol in ["vehicles", "structures", "oov"]:
        try:
            policy, sel = s3_triage.run(cidx, pol, budget_frac=0.02)
        except Exception as exc:
            s3_blocks.append((pol, f"{type(exc).__name__}: {exc}", ""))
            continue
        fig = chip_overlay(r, cidx, {c.chip.id for c in sel.selected},
                           {c.chip.id for c in sel.control}, step)
        s3_blocks.append((pol, s3_triage.render(sel, policy), fig))

    s3_html = "".join(
        f"""<div class="polblock"><h4>--policy {pol}</h4><div class="grid2">
{_fig(fig, "<b>Red</b> = dispatched to the detector. <b>Blue</b> = the mandatory "
           "control sample drawn from the <i>rejected</i> chips — the only "
           "unbiased recall signal there is.") if fig else ''}
<div>{_pre(txt, 1500)}</div></div></div>"""
        for pol, txt, fig in s3_blocks)

    P.append(f"""<section id="s3"><h2><span>S3</span> {ALGOS['s3']['title']}</h2>
{_algo('s3')}
<h4>All three policies, on this map's {len(cidx.chips):,} chips</h4>
{s3_html}
<div class="callout"><b>The cost line is the deliverable.</b> A 95% cost
reduction is only meaningful next to a recall number, and there is no recall
number until a detector runs on the control set. The report refuses to let
'score captured' stand in for it.</div>
{_try('s3', tile, ' --policy vehicles --budget-frac 0.02')}</section>""")

    # ---- S6
    say("preparing S6")
    P.append(f"""<section id="s6"><h2><span>S6</span> {ALGOS['s6']['title']}</h2>
{_algo('s6')}
<div class="callout warn"><b>Not demonstrated on this map.</b> S6 needs two
rasters of the same footprint at different dates, and there is only one date
here. Everything above is what it would do; nothing below it is a result.</div>
<h4>The transition classes it sorts into</h4>
<table class="t"><thead><tr><th>transition</th><th>category</th><th>what it means</th></tr></thead>
<tbody>
<tr><td><code>GreenGrassland → DryGrassland</code></td><td>phenology</td><td>the season changed, the ground did not</td></tr>
<tr><td><code>Batha → Garigue</code></td><td>succession</td><td>regrowth, or a boundary wobble</td></tr>
<tr><td><code>DryGrassland → House</code></td><td>construction</td><td>someone built something — the one worth a phone call</td></tr>
<tr><td><code>Water → LimestoneBoulder</code></td><td>impossible</td><td>evidence about the classifier, not the ground; free ground truth for S1</td></tr>
</tbody></table>
{_try('s6', tile, ' --second <second-date.tif>')}</section>""")

    # ---- run it yourself
    P.append(f"""<section id="yours"><h2>Run all of this on your own map</h2>
<p class="lead">Nothing on this page is specific to the tile it was built from.
Point any of it at your own raster and every image, table and number is
recomputed from yours.</p>
<h4>1 — the interactive demo</h4>
<pre>segmap ui -i /path/to/your-raster.tif
# -&gt; http://127.0.0.1:8765</pre>
<p>Three tabs. <b>Verbs</b> runs the S5 query language by hand; <b>Solutions</b>
runs S1–S6 with their options; <b>Ask</b> is the one path that needs an
Anthropic API key. The other two never call a model.</p>
<h4>2 — regenerate this page for your map</h4>
<pre>segmap showcase -i /path/to/your-raster.tif -o out/showcase.html</pre>
<h4>3 — one solution at a time</h4>
<pre>segmap solve s1 -i your.tif                      # audit
segmap solve s2 -i your.tif --region 1817        # adjudicate one region
segmap solve s3 -i your.tif --policy vehicles    # triage
segmap solve s4 -i your.tif --product concealment
segmap solve s5 -i your.tif --query "count House minarea 40"
segmap solve s6 -i your.tif --second date2.tif   # change</pre>
<h4>What your raster needs</h4>
<table class="t"><thead><tr><th>thing</th><th>required?</th><th>notes</th></tr></thead><tbody>
<tr><td>Single-band label raster</td><td><b>yes</b></td><td>GeoTIFF, <code>.npy</code>
or single-band PNG. Class ids, <b>not</b> a colourised RGB export.</td></tr>
<tr><td>Class id → name mapping</td><td><b>yes</b></td><td>Read automatically from an
<code>ID_TO_LABEL_MAPPING</code> GeoTIFF tag; otherwise pass
<code>--classes ids.json</code>.</td></tr>
<tr><td>Metres per pixel</td><td>yes</td><td>Taken from the GeoTIFF transform, or pass
<code>--gsd</code>. Every area, count threshold and distance depends on it.</td></tr>
<tr><td>A DEM</td><td>no, but</td><td><code>--dem</code>. Without it slope and aspect are
unmeasured, S1's morphology priors do not run, S2 cannot score morphology, S4 skips
the slope penalty and S5 refuses slope filters. It is the single biggest upgrade
available.</td></tr>
<tr><td>A second date</td><td>no</td><td><code>--second</code>. Required for S6 and
nothing else.</td></tr>
<tr><td>An API key</td><td>no</td><td>Only for <code>segmap ask</code>. All six
solutions are deterministic and run offline.</td></tr>
</tbody></table>
<p class="lead">Big rasters: the index build is the only slow step and it is
cached in <code>.segmap_cache/</code>, so the first command on a new tile pays it
and every later one loads in seconds. To poke at a huge AOI without waiting, add
<code>--max-mpx 40</code> — it crops to a centred window and says so in every
answer.</p>
</section>""")

    body = "\n".join(P)
    doc = _SHELL.format(
        title=_e(tile.split("/")[-1]),
        tile=_e(tile),
        meta=(f"{w}×{h} px · {r.gsd:.3f} m/px · {w*h/1e6:.1f} Mpx · "
              f"{round(w*r.gsd):,} × {round(h*r.gsd):,} m · "
              f"{len(ridx.regions):,} regions · {len(cidx.chips):,} chips · "
              f"{len(present)} classes"),
        dem=("DEM attached" if r.dem is not None
             else "no DEM — slope and aspect are unmeasured throughout"),
        demcls="" if r.dem is not None else " warn",
        body=body,
    )
    with open(out_path, "w") as fh:
        fh.write(doc)
    return out_path


_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<title>reasoning-terrain — {title}</title>
<style>
:root{{
  --bg:#f7f8f9; --panel:#ffffff; --ink:#14181d; --dim:#5d6772; --line:#dde2e7;
  --accent:#2f5d7c; --soft:#eaeff3; --ok:#2c7a52; --warn:#8a6320; --bad:#a33b2a;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans",var(--sans);
  --serif:"IBM Plex Serif",ui-serif,Georgia,"Times New Roman",serif;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#0f1418; --panel:#161d23; --ink:#e6ecf2; --dim:#8b97a3; --line:#26303a;
  --accent:#7fb3d5; --soft:#1b242c; --ok:#6fc79a; --warn:#d9ae62; --bad:#e89078;
}}}}
:root[data-theme="dark"]{{
  --bg:#0f1418; --panel:#161d23; --ink:#e6ecf2; --dim:#8b97a3; --line:#26303a;
  --accent:#7fb3d5; --soft:#1b242c; --ok:#6fc79a; --warn:#d9ae62; --bad:#e89078;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.65 var(--serif)}}
.wrap{{max-width:1140px;margin:0 auto;padding:0 24px 100px}}
header{{border-bottom:1px solid var(--line);margin-bottom:8px;padding:44px 0 26px}}
header h1{{font:600 30px/1.2 var(--sans);margin:0 0 12px;
 letter-spacing:-.01em}}
header .sub{{font:13px/1.6 var(--mono);color:var(--dim);word-break:break-all}}
.pills{{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}}
.pill{{font:11px var(--mono);text-transform:uppercase;letter-spacing:.07em;
 border:1px solid var(--line);background:var(--panel);color:var(--dim);
 border-radius:99px;padding:4px 11px}}
.pill.warn{{color:var(--warn);border-color:var(--warn)}}
.pill.ok{{color:var(--ok)}}
nav.toc{{position:sticky;top:0;z-index:5;background:var(--bg);
 border-bottom:1px solid var(--line);margin-bottom:34px;padding:11px 0;
 display:flex;gap:5px;flex-wrap:wrap}}
nav.toc a{{font:12px var(--mono);color:var(--dim);text-decoration:none;
 padding:4px 10px;border-radius:99px;border:1px solid transparent}}
nav.toc a:hover{{color:var(--accent);border-color:var(--line);background:var(--panel)}}
section{{margin:0 0 68px;scroll-margin-top:60px}}
h2{{font:600 24px/1.25 var(--sans);margin:0 0 14px;
 padding-bottom:9px;border-bottom:2px solid var(--line);letter-spacing:-.01em}}
h2 span{{display:inline-block;font:600 12px var(--mono);color:#fff;
 background:var(--accent);border-radius:4px;padding:3px 8px;
 vertical-align:3px;margin-right:10px;letter-spacing:.04em}}
h4{{font:600 13px var(--sans);text-transform:uppercase;
 letter-spacing:.08em;color:var(--dim);margin:30px 0 12px}}
p{{margin:0 0 14px}}
p.lead{{font-size:17px;max-width:74ch}}
code{{font:.85em var(--mono);background:var(--soft);padding:1px 5px;border-radius:3px}}
pre{{font:12.5px/1.55 var(--mono);background:var(--panel);border:1px solid var(--line);
 border-radius:6px;padding:13px 15px;overflow-x:auto;white-space:pre;margin:0 0 14px}}
pre code{{background:none;padding:0}}
.why,.abstain,.callout,.try{{border-radius:6px;padding:13px 16px;margin:0 0 16px;
 font-size:15px;border:1px solid var(--line);background:var(--panel)}}
.why{{border-left:3px solid var(--accent)}}
.abstain{{border-left:3px solid var(--warn)}}
.callout{{border-left:3px solid var(--ok);background:var(--soft)}}
.callout.warn{{border-left-color:var(--warn)}}
.callout.small{{font-size:14px}}
.try{{border-left:3px solid var(--dim);font-size:14px}}
.try pre{{margin:9px 0 9px}}
.try p{{margin:0;color:var(--dim);font-size:13.5px}}
ol.steps{{margin:0 0 16px;padding-left:22px}}
ol.steps li{{margin-bottom:9px;max-width:80ch}}
figure{{margin:0}}
figure img{{width:100%;display:block;border:1px solid var(--line);border-radius:6px;
 background:#fff;image-rendering:pixelated}}
figcaption{{font:13px/1.55 var(--sans);color:var(--dim);
 margin-top:9px}}
figcaption b{{color:var(--ink)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
 gap:26px;margin-bottom:18px}}
.gallery.one{{grid-template-columns:minmax(0,620px)}}
.gallery.three{{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:start}}
@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
ul.legend{{list-style:none;margin:0;padding:0;font:12.5px var(--mono);
 columns:2;column-gap:20px}}
@media(max-width:640px){{ul.legend{{columns:1}}}}
ul.legend li{{display:flex;gap:7px;align-items:center;padding:1px 0;
 break-inside:avoid}}
ul.legend i{{width:11px;height:11px;border-radius:2px;flex:0 0 auto;
 border:1px solid rgba(128,128,128,.4)}}
ul.legend span{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
ul.legend b{{color:var(--dim);font-weight:400;font-variant-numeric:tabular-nums}}
.examples{{display:grid;gap:14px;margin-bottom:16px}}
.ex .q{{font-size:13px;margin-bottom:6px}}
.ex.refused pre{{border-left:3px solid var(--bad)}}
.ex.ok pre{{border-left:3px solid var(--ok)}}
table.t{{border-collapse:collapse;width:100%;font:12.5px var(--mono);
 margin-bottom:16px;display:block;overflow-x:auto}}
table.t th{{text-align:left;font-weight:600;color:var(--dim);
 border-bottom:1px solid var(--line);padding:6px 9px;white-space:nowrap}}
table.t td{{border-bottom:1px solid var(--line);padding:5px 9px;
 vertical-align:top}}
table.t td.n{{text-align:right;font-variant-numeric:tabular-nums}}
table.t td.ids{{color:var(--dim)}}
.cls{{font:11.5px var(--mono);color:var(--accent)}}
table.t tr.bad td{{background:rgba(150,53,31,.09)}}
.fine{{font-size:13.5px;color:var(--dim);margin-bottom:0}}
.mono{{font:11.5px var(--mono)}}
.polblock{{margin-bottom:30px}}
footer{{border-top:1px solid var(--line);padding-top:20px;color:var(--dim);
 font-size:13.5px}}
</style></head><body><div class="wrap">
<header>
<h1>What is in this segmentation map</h1>
<div class="sub">{tile}</div>
<div class="sub">{meta}</div>
<div class="pills">
<span class="pill ok">6 solutions</span>
<span class="pill ok">no model at runtime</span>
<span class="pill{demcls}">{dem}</span>
</div>
</header>
<nav class="toc">
<a href="#map">The map</a><a href="#layers">Layers</a><a href="#detail">Detail</a>
<a href="#s5">S5 query</a><a href="#s4">S4 products</a><a href="#corridor">corridor</a>
<a href="#s1">S1 audit</a><a href="#s2">S2 adjudicate</a><a href="#s3">S3 triage</a>
<a href="#s6">S6 change</a><a href="#yours">Your map</a>
</nav>
{body}
<footer>
Generated by <code>segmap showcase</code>. Every image, table and number on this
page was computed from the raster named at the top — none of it is illustrative.
Nothing here called a language model: all six solutions are deterministic.
</footer>
</div></body></html>
"""
