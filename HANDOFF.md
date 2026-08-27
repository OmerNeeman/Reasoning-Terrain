# HANDOFF — reasoning-terrain (RT)

**Read this first. It is written for someone with zero context — a new engineer,
or a fresh AI session — and it stands alone. The other docs are depth; this is
the map.**

The project is **reasoning-terrain**, **RT** for short; both names are used
throughout these docs. The import package is still `segmap_digest` and the
command is still `segmap` — those were deliberately not renamed. See §9.16.

Everything numeric below was produced by running a command in this repo against
the real data in `data/incoming/` on 2026-08-25, or by reading a file in it.
Anything I could not verify is marked `[unverified]`. Where an earlier session's
notes disagreed with the repo, the repo won and the correction is called out.

---

## 1. What this is, and why it exists

The question RT answers is: **what can a reasoning LLM do with a Smart Terrain
segmentation map?**

The premise is that a segmentation raster is unusually good LLM input because it
is already symbolic. The segmenter learned `TerraRosa` as an integer and knows
nothing about clay mineralogy. The LLM knows the geology and cannot see pixels.
The **class names are the interface between them** — which is why the prose
definitions in `taxonomy.py` are the highest-leverage artifact in the repo.

Two hard scope decisions, both from the stakeholder, both load-bearing:

- **No RGB imagery, anywhere.** The input is the label raster and nothing else.
  `loader.load()` will refuse a 3-band PNG rather than guess class ids back out
  of colour.
- **There is no downstream detector in this pipeline.** It is
  segmentation-in, user-facing-interpretation-out. This matters because S3 was
  designed around a detector that does not exist — see §8.

The repo is standalone at `~/PycharmProjects/ST_repos/reasoning-terrain`, a
sibling of the other ST repos. It is **not wired into `smart-terrain-v2-devenv`**,
and the data is gitignored. It **does** have a git remote now —
`git@github.com:OmerNeeman/Reasoning-Terrain.git`, with `master` pushed and
`origin/master` at the same commit — so this is no longer a local-only repo.
30 commits on `master` (`git rev-list --count master`).

---

## 2. Repo layout

| Path | What it is |
|---|---|
| `src/segmap_digest/taxonomy.py` | **The 47 classes.** Prose definitions, 47→9 superclass collapse, ordinal series, lithology×morphology grid, active priors, and `RETIRED_PRIORS`. The file an expert should review. |
| `src/segmap_digest/loader.py` | Read one label raster. Wire-id→dense-id remap **by name**, nodata mask, degrees→metres GSD conversion, centre-crop (`--max-mpx`), colormap. |
| `src/segmap_digest/mosaic.py` | Stitch a directory of tiles by their affine transforms. Raises on mixed CRS or pixel size rather than resampling. |
| `src/segmap_digest/index.py` | **Stage 0.** Region index (connected components + geometry + neighbour graph) and chip index (fixed grid + histograms + distance-to-anchor). Query-independent by design. |
| `src/segmap_digest/cache.py` | Memoises mosaic / regions / chips to `.segmap_cache/`, keyed on file paths+mtimes+sizes, GSD, crop, `--min-px`, chip size, anchors, schema version. Big arrays are memory-mapped. |
| `src/segmap_digest/digests.py` | The six representation levels (`l0 l1 l1q l2 l3 chips`) plus an offline token estimator. |
| `src/segmap_digest/audit.py` | Deterministic prior checks → graded-severity `Finding` candidates. The cheap half of the reasoning layer. |
| `src/segmap_digest/report.py` | Writes the end-to-end HTML page. |
| `src/segmap_digest/tools.py` | The six S5 verbs as Anthropic tool definitions. Runs offline (`segmap tools`). |
| `src/segmap_digest/ask.py` | Drives the SDK tool runner, or the `--digest` interpretive path. **Never exercised live** — see §9. |
| `src/segmap_digest/synth.py` | Synthetic fixture tile and synthetic second date, so everything runs with zero data. |
| `src/segmap_digest/cli.py` | The `segmap` command. |
| `src/segmap_digest/solutions/s1..s6` | The six map-consuming solutions, each a runnable naive implementation. |
| `docs/solutions/S*.md` | One doc per solution: heuristics, open questions, options. |
| `docs/pipeline.html` | Deep dive on the pipeline. **Written concurrently by another agent — do not edit from here.** |
| `examples/smart_terrain_class_ids.json` | The wire-id → class-name mapping as read off the sinai drop. 47 entries. |
| `data/incoming/{sinai,aza,leb}/` | The real exports. Gitignored. |

---

## 3. The pipeline, briefly

```
GeoTIFF(s)                    ── loader / mosaic ──►  LabelRaster
                                                      (labels, valid mask, gsd, transform, crs)
LabelRaster  ── index (cached) ──►  RegionIndex  +  ChipIndex
RegionIndex  ── digests ──►  l0 l1 l1q l2 l3 chips   (text an LLM can read)
RegionIndex  ── audit ──►  Findings                  (candidates, not errors)
both         ── solutions S1..S6 ──►  worklists, verdicts, triage, products, answers, diffs
anything     ── report / ask ──►  HTML page, or an LLM answer built from tool calls
```

Stage 0 (the index) is the only expensive step, it is query-independent, and it
is cached. Everything after it is seconds. **`docs/pipeline.html` covers this
properly** — go there for depth.

---

## 4. Real data inventory

Verified with `rasterio` over every file, and with `segmap digest l0`.

### sinai — `data/incoming/sinai/`

| | |
|---|---|
| Tiles | 20 GeoTIFFs, each 8192×8192, 1 band, uint8, nodata=0 |
| CRS | EPSG:4326 (geographic), transform `a=5.2e-06`, `e=-4.5e-06` |
| Mosaic | **31822×37535 px = 1194.4 Mpx @ 0.496 m/px** |
| Classified | **59.3%** of the extent; **40.7% nodata**; **174.504 km²** classified |
| Regions | **133,976** at `min_area_px=12` |
| Classes present | **35** in the pixel histogram; **33** survive into the region index |
| `ID_TO_LABEL_MAPPING` tag | present on **all 20** tiles |
| DEM | none |

Composition is extremely lopsided: **Rendzina 64.9%**, DryGrassland 17.7%,
Batha 3.9%, TerraRosa 3.0%, Water 2.7%, MaralBadlands 1.9%. Everything else is
under 1.5%.

### aza — `data/incoming/aza/`

| | |
|---|---|
| Tiles | 4 GeoTIFFs, each 8192×8192, 1 band, uint8, nodata=0 |
| CRS | EPSG:4326, transform `a=1.3e-06`, `e=-1.1e-06` |
| Mosaic | **8509×8405 px = 71.5 Mpx @ 0.122 m/px** |
| Classified | **100.0%** — no nodata at all |
| Area | **1.073 km²** |
| Regions | **95,170** |
| Classes present | **29** |
| `ID_TO_LABEL_MAPPING` tag | **absent on all 4 tiles** |
| DEM | none |

Composition: **Shadow 28.9%**, MaralBadlands 28.1%, House 8.9%, Batha 6.4%,
DryGrassland 5.3%, DirtRoad 4.8%, Clutter 4.7%. Nearly a third of this AOI is
labelled `Shadow`, which is an illumination artifact, not a surface — treat any
aza area figure with that in mind.

### leb — `data/incoming/leb/`

**Empty.** Zero files. Pointing `-i` at it raises a clear error by design
("An AOI whose data has not landed yet is expected to be skipped, not to produce
an empty report").

### There is no DEM anywhere

Verified: every cached region index has `has_terrain: false`. This is the single
biggest constraint on the whole project — see §9.

---

## 5. The taxonomy, and the wire-id trap

**The taxonomy is 47 classes, not the 45 it was originally built against.**
`taxonomy.py` asserts this at import.

The classes the segmenter writes into a GeoTIFF are **sparse "wire" ids in
0..241** (`Unclassified=0`, `Clutter=2`, … `ChalkTerrace=241`), not the dense
0..46 ids in `taxonomy.py`. The exporter ships the translation in an
`ID_TO_LABEL_MAPPING` GeoTIFF tag; `loader.class_map_from_tags` reads it and
`loader.wire_lut` translates **by name**, refusing any name the taxonomy does not
define rather than folding it into `Unclassified`.

**What went wrong originally, and why it is a trap.** The first cut assumed the
wire ids in ascending order corresponded positionally to the class list. That
was right for the first 30 classes and wrong for the rest: two classes were
missing from the assumed list — `LimestoneHardRockLineament` (wire 170, dense
30) and `ChalkTerrace` (wire 241, dense 46) — and inserting `LimestoneHardRockLineament`
at position 30 shifted every class after it by one. Silent, total, and only
visible as "this map has a suspicious amount of dolomite".

**The trap that remains:** I verified that today, with the corrected 47-class
list, **ascending wire order matches dense id order for all 47 classes, with
zero mismatches.** In other words a positional assumption would *accidentally
work right now*, and would break silently the next time a class is added or
removed. Do not reintroduce one. The name-based mapping in `loader.py` is the
correct mechanism and must stay.

### Structure inside the flat list

- **Lithology × morphology grid**, sparse and asymmetric on purpose. Verified
  counts: Limestone **7** morphologies, Dolomite 4, Nari 4, Marl 3, Basalt 3,
  Chalk **2** — **23 rock classes** in total. (README says "limestone has six
  morphologies, chalk exactly one"; that is stale and wrong. The code comment in
  `taxonomy.py:250` says seven and two, and it is right.)
- **Ordinal series.** `DryGrassland → Batha → Garigue → Maquis` is the
  Mediterranean degradation gradient; `DirtRoadB → DirtRoad → PavedRoad` the road
  grade. `class_distance()` gives near-misses credit; plain cross-entropy does not.
- **47 → 9 superclasses** for coarse reasoning and legible colourmaps.

---

## 6. What is trustworthy, and what is not

**Trust these.** They are computed from the raster by code, with the honest
denominator (classified pixels, never the whole extent):

- class histograms and areas (`l0`, `describe`)
- region geometry: area, perimeter, compactness, elongation, bbox, centroid
- the neighbour graph and shared boundary lengths
- pixel-to-pixel `distance` between two classes
- corridor connectivity, *given* that its trafficability constants are guesses
- token counts and cache timings

**Do not trust these.**

- **Every slope, aspect and elevation number is `0.0`, and means "unmeasured".**
  There is no DEM. The code mostly abstains now (S1 skips the slope/aspect
  priors, S2 returns a neutral 0.5 and says so, S5's `slope <` filter raises
  `Unanswerable`) — but the `slope`, `slope_sd`, `elev` and `aspect_cv` columns
  are still printed as `0.0` in every `l2` row. Anything reading that table
  without knowing this will read "flat" where it should read "unknown".
- **Every heuristic constant in `solutions/` and `audit.py` is a guess.** They
  are grouped at the top of each module with a "where this should actually come
  from" comment. Severity is an ordinal ranking signal *within* a kind; it is not
  a calibrated error probability, and nothing has checked whether the bands are
  comparable across kinds.
- **S6's "% plausibly real change" is measured against a synthetic second date.**
  On the sinai mosaic it reports 0.2% of pixels differing and "73% of the
  differing area is plausibly REAL change" — but the second date is
  `synth.second_date()`, a synthetically aged copy of the first. The number
  describes the synthesiser, not the world.
- **S3's "84% of score captured" is prior mass, not recall.** The module says so
  in a WARNING line. There are no detector outcomes to calibrate against, and no
  detector.
- **aza's class mapping is unconfirmed** — see §9.

---

## 7. Decisions already made, and why. Do not relitigate these.

1. **S5 uses LLM tool calls, not code execution.** The model chooses a verb and
   its arguments; `s5_query` does the arithmetic. The model cannot return a
   count, an area or a distance the code did not compute, because it has no
   other way to see the raster. Auditability beats flexibility here. Six verbs
   exist: `find count area distance corridor describe` (verified via
   `segmap tools`).
2. **Mosaics, not per-tile analysis.** Regions are connected components, so
   per-tile analysis cuts every region at the seam — one wadi across four tiles
   becomes four regions with four wrong areas and four wrong neighbour lists.
3. **Mixed CRS or resolution raises rather than resampling.** Resampling a
   *label* raster invents classes at every boundary.
4. **`--max-mpx` crops, it does not stride.** Striding a label raster deletes
   every feature narrower than the stride — roads and wadis here are 1–4 px. A
   crop changes exactly one thing (which ground you are looking at) and the
   returned raster carries a `subset_note` that every renderer repeats.
5. **The class definitions in `taxonomy.py` are the product.** They are what an
   LLM reasons over. They are prose in a flat file, not code, precisely so a
   domain expert can edit them.
6. **The LLM is never in a per-item loop.** For triage the model compiles a
   small cached policy; a deterministic scorer applies it to millions of chips.
7. **Never truncate silently.** Every `--limit` emits a `# NOTE: N omitted` line.
   Enforced by a test.
8. **S2 is allowed to abstain.** `UNDECIDABLE-NEEDS-geological-map` is a
   first-class verdict. Net gain is corrections minus corruptions, and it can be
   negative.
9. **The soil-genesis priors are retired and must not come back from a
   textbook.** See §8, item 6.

---

## 8. Bugs found on real data — cautionary tales

These are the most useful part of this document. Every one was a **silent
corruption**: valid-looking output, wrong content, no error. All are fixed; the
lesson in each is not.

**1. Degrees read as metres.** In EPSG:4326 a 0.5 m/px export has
`transform.a = 5.2e-06`. Taking `abs(a)` at face value made every area in the
report **ten orders of magnitude too small** — and the report still rendered
cleanly. Fixed in `loader.geotiff_gsd()`, which converts when `crs.is_geographic`.
*Lesson: a unit error in a georeferenced pipeline does not raise; it just quietly
lies.*

**2. `nodata=0` collides with `Unclassified`, whose id is also 0.** The two are
not distinguishable in the file. Counting nodata as a class made 40% of an arid
AOI read as "unclassified terrain", and made S5's corridor verb score the nodata
margin as trafficable (`Unclassified.traffic = 0.5`, above the 0.45 threshold)
and divide by the full extent. On tile `x800_y846` — which I verified is **67.4%
classified, 11.13 km² of real ground inside a 16.52 km² extent** — that reported
**~15.9 km² reachable** `[unverified: the pre-fix figure is from session notes;
the mechanism and the tile arithmetic are verified]`. Today the same query
correctly returns **10.52 km² = 94.4% of classified area**. Fixed by
`LabelRaster.valid`, `n_valid` as the denominator everywhere, and nodata scoring
0 in `s4_products.compute`. *Lesson: if the segmenter genuinely emits
`Unclassified`, that class is currently being thrown away, and the exporter needs
to give nodata a value of its own.*

**3. No DEM means slope reads `0.0` — a valid-looking number meaning
"unmeasured".** Before this was caught, S1 emitted **severity-0.80
slope-violations** against every badlands polygon on the map, and S2 switched a
real label to `MaralTerrace` citing *"aspect coherent (0.00)"* — a manufactured
corruption, confidently argued from an input that was never measured. Fixed by
`RegionIndex.has_terrain`, and by making the affected checks **abstain and say
so** rather than score. Verified today: S1's coverage block prints
`ABSTAINED -- no DEM: every mean_slope and aspect_circvar is 0.0, which is a
valid-looking number meaning 'unmeasured'`; S2 prints `morphology unscored: no
DEM`; S5 raises `Unanswerable`. *Lesson: a missing input that defaults to a
plausible value is worse than one that crashes.*

**4. `build_regions` renumbered region ids but left the label array on the old
numbering.** Every zonal statistic then read a *different region* than the one it
was labelled with. Nothing errored; the numbers were simply about the wrong
polygons. Fixed at `index.py:178-191`, where the LUT is applied to `glob` in the
same breath as the id remap, with the reason in a comment. *Lesson: two
representations of the same identity must be renumbered atomically or not at
all.*

**5. Four memory/complexity blowups, invisible at 1024 px and fatal at 8192 px.**
Each is now fixed with the reason recorded inline:
- `ndi.center_of_mass(np.ones_like(glob), ...)` made an int32 copy of the label
  array — 4 bytes/px of pure waste, several GB on a mosaic. Now `uint8`.
- `glob.max()` sat inside the per-pair neighbour loop, making it
  O(n_pairs × n_pixels) — a hang on a real tile. Hoisted.
- `build_chips` held one distance transform per anchor: 8 bytes/px/anchor,
  **112 GB for seven anchors over a 2 Gpx mosaic**. Now one anchor at a time,
  reduced to per-chip minima before the next.
- Every command rebuilt the index. On the sinai mosaic that is 24 min and 47 GB
  *per question*. Now cached and memory-mapped.

**6. Two of our own assumptions about the model were wrong, both corrected by
the stakeholder.**

- **The soil-genesis priors.** We encoded the textbook: terra rossa forms on hard
  carbonate, rendzina on chalk and marl. The class owner confirmed these classes
  are **soil types, not genetic units** — the segmenter assigns them from what the
  soil surface looks like, not from what rock it formed on. The cost was measured:
  the Rendzina `expects` prior alone produced **9,793 of 41,669 findings** on the
  sinai mosaic and filled the entire top-25 worklist with one sentence repeated
  twenty-five times `[per the note in taxonomy.py; the post-fix count of 26,592
  findings is verified]`. Four priors retired to `taxonomy.RETIRED_PRIORS` as
  *text, not checks*, so the next person to read a soil-genesis chapter finds the
  reason instead of re-deriving them. Definitions rewritten to say
  "The label is a SOIL TYPE ... carries no claim about the parent rock".
  *Lesson: a prior earns its place only if it is a statement about the mapped
  object — its geometry, its position, the physics it must obey. A statement
  about how the object came to exist is a statement about the world, not about
  the label, and the label is all this pipeline can see.*
- **S3's purpose.** It was designed as detector-dispatch triage. There is no
  detector. It is being re-pointed at analyst attention and LLM token budget.
  **The S3 module and `docs/solutions/S3-triage.md` still carry the old
  dispatch/cost wording** — verified, and a known inconsistency, not a fresh bug.

---

## 9. Open items, ranked by what they block

**1 — No DEM. Blocks the most by a wide margin.**
23 of 47 classes are lithology×geomorphology, and morphology (dip slope, terrace,
badlands, boulder field) is what slope / aspect / curvature resolve and colour
does not. Without it those classes are **unfalsifiable**: S1's slope and aspect
priors abstain, S2's morphology term returns a neutral 0.5, S5's slope filter
refuses, and `corridor` returns **93.9% of the sinai mosaic** because every slope
is 0.0 and nothing is ever too steep. For this taxonomy the DEM is worth more
than RGB.

**2 — S1's substantive coverage is 3.0% of the sinai mosaic.**
Verified today: `15 of 33 classes present have at least one active substantive
check; those classes hold 3.0% of classified region area`. The unexamined 97% is
Rendzina (65%), DryGrassland (17.7%), Batha (3.9%), TerraRosa (3.0%), Water
(2.7%). A short findings list on this map is **not a clean bill of health**, and
S1's coverage block says so explicitly. Roughly half of the gap is item 1; the
other half is that most non-rock classes simply have no prior to contradict them.
*(See §11 for the coverage-figure discrepancy and its resolution.)*

**3 — S1's blocking question: what is the false-positive rate?**
Nothing has measured it. It needs ~150 reviewed regions. Until then no precision
number exists for any S1 output.

**4 — `DryGrassland` islands inside `Rendzina`: 15,469 regions, the #1 root
cause on sinai.** Is this a real error or is it normal terrain? Nobody here can
answer it; it needs the class owner. It is 0.15% of map area but a quarter of all
findings by region count.

**5 — The isolated-speck "distant host" filter does not discriminate.**
`class_distance()` returns **exactly 1.0 for all 1,592 cross-superclass ordered
pairs**, and **90.9% of all ordered class pairs are ≥ the 0.70 speck threshold**
(verified by enumeration). So the filter admits nearly everything rather than
selecting, and the "strangeness" term it feeds into severity is a constant 1.0
for every finding it produces. Result: **97% of the 26,592 sinai findings are
`isolated-speck`**, ranked only by enclosure and smallness. *(Correction: an
earlier note said the filter "selects nothing". It is the opposite — it selects
almost everything, which is the same uselessness from the other end.)*

**6 — aza's class mapping is unconfirmed, and nothing flags it.**
The aza tiles carry no `ID_TO_LABEL_MAPPING` tag, so sinai's mapping is applied
via `--classes examples/smart_terrain_class_ids.json`. **No warning is printed
anywhere** — I checked stderr, the digest headers and the report. Worse, without
`--classes` the mosaic path does not validate at all: `mosaic.py:84` passes raw
wire ids straight through as if they were dense ids when a tile has no tag, and
the failure surfaces much later as a bare `KeyError: np.int64(94)` from
`digests.py`. The single-tile path (`loader._load_labels` → `_check`) gives a
proper actionable error; the mosaic path skips it. **Two fixes wanted: call
`_check` in the mosaic path, and print a provenance line when a mapping is
borrowed from another AOI.**

**7 — `l2` is unusable at AOI scale, and so is `chips`.**
Verified token estimates on the sinai mosaic: `l0` **398**, `l1` **1,541**,
`chips` **425,882**, `l3` **1,701,818**, `l1q` **1,735,351** (96 s to build),
`l2` **3,047,077**. On aza's single km²: `chips` 29,421, `l2` **2,259,149**.
*(Correction: an earlier note said `l0`/`l1`/`chips` all ship. Only `l0` and `l1`
are cheap on sinai — `chips` is 426k tokens there, and 38.8% of those 18,104
chips are entirely nodata and are emitted anyway.)*
`l2`/`l3` scale with region **count**, not tile size, so the lever is
fragmentation, not extent.

**8 — Region-level analysis breaks down on this data.**
The sinai mosaic's largest region is a single `Rendzina` component of **86.23 km²
— 49.4% of all region area — with 51,810 neighbours**. The top 10 regions hold
65.9% of the area; the median region is **26 m²**. Half the map is one percolating
blob whose compactness (0.00) and elongation (1.1) say nothing, and the other
half is salt-and-pepper: **20.4% of indexed regions are below the 32-cell speck
floor**. "Region" is not a meaningful unit here without a morphological opening
step upstream. This was not in the prior notes and I think it is important.

**9 — The live LLM path has never run.**
`anthropic 1.0.0` is installed; `ANTHROPIC_API_KEY` is **not set** in this
environment. `segmap ask` has never made a real request. `segmap tools` works
offline and prints all six definitions, so the surface is inspectable, but
nothing about the model's actual verb selection has been observed. S5's doc calls
for ten real analyst questions plus one live run.

**10 — Answers are not georeferenced.**
The transform and CRS are loaded and carried on `LabelRaster`, but **no digest or
query output emits lat/lon** — verified, everything is pixel `cy`/`cx` and region
ids. A region id is checkable only against `segmap digest l2`, not against a map.
For anyone who has to act on a finding, this is the gap between an answer and a
usable answer.

**11 — `near` uses centroid distance; `distance` does it properly.**
`s5_query._find` filters on centroid-to-nearest-class distance, which is wrong
for a long sinuous region whose centroid can be far from every part of it. It
emits a note saying so, which is honest but not a fix. `distance` is pixel-to-
pixel and correct.

**12 — S4's product choice is undecided.** 39 options are catalogued in
`docs/solutions/S4-options.md` — the scoring table has 39 rows and the doc states
the count on the line directly under it ("**39 products. Pick these three:**").
The old pointer here, "line 879, 37 rows", was wrong twice over: the table has
grown, and line 879 is now prose in the G3 biomass section.

The doc *recommends* three — GO/SLOW-GO/NO-GO, diggability, obstacle inventory —
but explicitly says "Nothing here is implemented. It exists so that someone can
choose." The stakeholder is still choosing.

**13 — `min_area_px=12` is in resolution cells, which means different things per
AOI.** 12 cells is **0.180 m² on aza** and **2.954 m² on sinai** — a 16× swing in
what gets indexed at all. The same class of mistake produced the aza worklist
that ranked 300 m² *buildings* as isolated specks, and it is why `audit.py` now
carries two separate windows (cells for specks, m² for OOV candidates) with the
reasoning written out at `audit.py:59-96`.

**14 — Stale documentation.**
- The `colormap()` docstring says `"""(45, 3) uint8 RGB palette."""` — the
  docstring is **wrong**, the array it builds is `(N_CLASSES, 3)` = `(47, 3)`.
  It is at `loader.py:406-407`, not `loader.py:321` as this line used to say.
  Fixing the docstring is a code change and has not been made.
  (`docs/solutions/S4-options.md` said **45 classes** throughout and was
  corrected on 2026-08-27; `S4-products.md` and `README.md` were corrected on
  2026-08-25, along with README's `l0` token count and its fixture region
  count.)
- `QUICKSTART.md:42` (not `:37`, which is about the editable install) describes
  the synthetic fixture as coherent because it puts "terra rossa on hard
  carbonate" — the retired hypothesis, presented as a virtue.
- `S4-options.md` uses rendzina/terra rossa parent-rock inference as product
  signal in **C3 — Dust potential** (`Rendzina` as "a shallow pale calcareous
  soil", ~line 494) and **C5 — Erosion susceptibility** (`Rendzina` as "shallow
  soil on soft carbonate", ~line 539). The old pointers, `:420` and `:546`, no
  longer land on either — search by section heading, the line numbers in that
  file move every time the catalogue grows.
- S3's module and doc still use detector-dispatch/cost framing (§8, item 6).

**15 — The rename invalidated the whole index cache. Budget 24 minutes before
you conclude something is broken.**
The cache key includes the input files' **absolute paths** (§2, `cache.py`), and
the directory moved from `segmap-digest-poc` to `reasoning-terrain`, so every
entry under `.segmap_cache/` now misses. That is the cache working as designed —
a stale entry misses rather than answering — but it means the **first mosaic
query after the rename is a cold rebuild: ~24 min and 47 GB on sinai**, not the
1.5 s the tables below quote. Nothing is wrong. Run `segmap index -i
data/incoming/sinai/` once and the warm numbers come back. `segmap index --list`
still lists the orphaned entries under their old paths; they are dead weight and
can be deleted.

The rename broke one other path: any **editable install made before it still
points at the old directory**, so a bare `segmap` fails with
`ModuleNotFoundError: No module named 'segmap_digest'` until you re-run
`pip install -e '.[geo,dev]'` from here. `PYTHONPATH=src python3 -m
segmap_digest.cli ...` works either way.

**16 — The Python package and the CLI were deliberately *not* renamed.**
The distribution is `reasoning-terrain`, but the import package is still
`segmap_digest` and the command is still `segmap`. Renaming those touches every
import, the console-script entry point, the cache schema version and 202 tests,
and would invalidate the cache a second time — worth doing in one deliberate
pass if the `segmap` name stops making sense, not as a side effect of a doc
change.

---

## 10. How to run it in 60 seconds

```bash
cd ~/PycharmProjects/ST_repos/reasoning-terrain
pip install -e '.[geo,dev]'          # numpy scipy pillow rasterio pytest
pytest tests -q                      # 202 passed, 1 skipped in ~24 s (2026-08-27)

segmap legend --full                 # the 47 class definitions
segmap compare                       # zero data: synthetic tile, all levels, token counts
```

On the real exports — **the cached sinai index no longer matches, because the
rename changed the input paths the cache keys on. The first of these commands
rebuilds it cold: ~24 min, 47 GB. See §9.15.** After that rebuild they are
seconds again, and the timings below apply.

```bash
segmap index --list                                     # what is cached
segmap digest l0   -i data/incoming/sinai/              # ~5 s
segmap solve  s1   -i data/incoming/sinai/              # ~2 s
segmap solve  s5   -i data/incoming/sinai/ --query "corridor PavedRoad"
segmap solve  s5   -i data/incoming/sinai/ --query "count House minarea 40"
```

aza **requires** the class mapping (its tiles have no tag):

```bash
segmap digest l0 -i data/incoming/aza/ --classes examples/smart_terrain_class_ids.json
segmap report    -i data/incoming/aza/ --classes examples/smart_terrain_class_ids.json -o out/aza
```

`segmap report` writes a self-contained HTML page (verified: aza, 1 m 40 s, most
of it the 128-px chip index).

**Cache economics, measured from `.segmap_cache/*/meta.json`:**

| sinai mosaic, 1194 Mpx | cold | warm |
|---|---:|---:|
| stitch mosaic | 18.8 s | — |
| region index (133,976 regions) | 370.0 s | — |
| chip index (18,104 chips @ 256 px) | 1071.1 s | — |
| **total** | **24 min 20 s** | **~1.5–5 s per question** |

The chip index is 73% of the cold cost. Every command prints on stderr which
cache it used. `--refresh-index` rebuilds, `--no-cache` bypasses, `$SEGMAP_CACHE`
relocates.

---

## 11. The coverage discrepancy — resolved

Two figures were in circulation for S1's coverage on sinai: *"15 of 33 classes,
3.0% of area"* and *"substantive check on 1.4% of area"*. **Both are correct.
They are different inputs, and neither is wrong.**

| Input | Regions | Root causes | Findings | Coverage |
|---|---:|---:|---:|---|
| `data/incoming/sinai/` (20-tile mosaic) | 133,976 | 178 | 26,592 | **15 of 33 classes, 3.0% of area** |
| `data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif` (one tile) | 11,058 | 34 | 2,484 | **8 of 20 classes, 1.4% of area** |
| `data/incoming/aza/` (4-tile mosaic) | 95,170 | 120 | 5,980 | **12 of 29 classes, 33.7% of area** |

All three verified by running `segmap solve s1` today. The 1.4% figure is the
one printed in `QUICKSTART.md:304`, whose worked examples all use that single
tile; the 3.0% figure is the mosaic one quoted in `docs/solutions/S1-audit.md:16`.

**The AOI-level number to quote is 3.0% for sinai and 33.7% for aza.** aza scores
far higher only because MaralBadlands — a rock class, and therefore covered by
the `lithology-isolation` check — is 28% of that AOI, whereas sinai is
two-thirds Rendzina, which has no active check at all now that the genesis priors
are retired.

The mechanism, for anyone changing it: `s1_audit.Coverage.n_checked` counts
classes present on the map with at least one check in `SUBSTANTIVE_CHECKS`
(`context-prior`, `slope-prior`, `aspect-prior`, `lithology-isolation`,
`nari-geometry`). `enclosed-speck` is deliberately excluded — it applies to all
47 classes and tests the shape of a component, not whether its class is right,
so counting it would make coverage look complete when it is not. With no DEM the
slope and aspect families never activate, so in practice coverage today is
"rock classes, plus Car / Maquis / HydromorpicSoil".

---

## 12. Where to look for depth

| Question | Go here |
|---|---|
| How does the pipeline fit together, stage by stage? | `docs/pipeline.html` |
| I want to run ten real questions and see real output | `QUICKSTART.md` |
| Representation ladder, token costs, design rules | `README.md` |
| What each class *means* | `src/segmap_digest/taxonomy.py`, or `segmap legend --full` |
| Why the soil priors were retired | `taxonomy.py`, the block above `RETIRED_PRIORS` |
| S1 audit heuristics, and the constants that are guesses | `docs/solutions/S1-audit.md` |
| S2 adjudication, and why abstaining is a verdict | `docs/solutions/S2-adjudicate.md` |
| S3 triage (**read with §8 item 6 in mind — the detector does not exist**) | `docs/solutions/S3-triage.md` |
| The 37 candidate derived products | `docs/solutions/S4-options.md` |
| S5 query verbs and the tool surface | `docs/solutions/S5-query.md`, `segmap tools` |
| S6 change detection and the co-registration threat | `docs/solutions/S6-change.md` |
| What to send us to unblock things | `data/incoming/README.md` |

### The next three things worth doing, in order

1. **Get a DEM.** It unblocks 23 classes, four checks, two solutions and the
   corridor verb. Nothing else on this list is worth as much.
2. **Get ~150 reviewed regions.** That is S1's blocking question, and it converts
   every severity number in the repo from ordinal to calibrated.
3. **Get the aza class mapping confirmed**, and add the provenance warning and
   the `_check` call described in §9.6 — so the next borrowed mapping announces
   itself instead of crashing three modules downstream.

Then: the geological map (the largest single accuracy lever — it is what S2
abstains on today), a second real date for S6, and ten real analyst questions
plus one live `segmap ask` run for S5.
