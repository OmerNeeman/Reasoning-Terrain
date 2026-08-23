# segmap-digest-poc

Turn a **45-class Smart Terrain segmentation raster** into representations a
reasoning LLM can actually work with — and compare those representations by
token cost, side by side.

The premise: a segmentation map is unusually good LLM input because it is
already symbolic. The segmenter learned `TerraRosa` as integer 15 and knows
nothing about decalcification clay. The LLM knows the geology but can't see the
pixels. The class names are the interface between them — which makes the class
definitions in [`taxonomy.py`](src/segmap_digest/taxonomy.py) the single
highest-leverage artifact here.

No RGB imagery is involved anywhere in this pipeline.

---

## Install and run

```bash
pip install -e .            # numpy, scipy, pillow
pip install -e '.[geo]'     # + rasterio, for GeoTIFF input
pip install -e '.[llm]'     # + anthropic, for the `ask` command
```

Everything runs with **zero data** — [`synth.py`](src/segmap_digest/synth.py)
generates a spatially coherent fixture tile (terra rossa on hard carbonate,
rendzina on chalk, badlands only where it's steep, maquis on shaded slopes, a
village with roads and parked cars). Swap in a real tile with `-i tile.tif` the
moment you have one.

```bash
segmap compare                       # all levels, with token counts
segmap digest l2 --limit 40          # region table
segmap audit                         # consistency findings
segmap preview -o tile.png           # colourised PNG, for eyeballing
segmap legend --full                 # the 45 class definitions
```

---

## The representation ladder

Measured by `segmap compare` on the synthetic 1024×1024 tile at 0.3 m/px
(1,583 regions):

| level  | tokens (est.) | what it is | good for |
|--------|--------------:|------------|----------|
| `raw`  | 786,432 | label raster as text | **never do this** |
| `l0`   | 368 | class histogram | composition, sanity checks |
| `chips`| 538 | fixed-grid chip index (256 px) | detection triage — the unit of spend |
| `l1`   | 2,007 | 16×16 grid, top-3 classes per cell | "where is what" |
| `l1q`  | 10,916 | quadtree — uniform areas collapse to one token | spatial structure, compressed |
| `l3`   | 36,534 | region adjacency graph | connectivity, corridors |
| `l2`   | 47,908 | region table with terrain attributes | **the workhorse**: audit, adjudication |

Two things that table makes obvious:

- **`l0`, `l1`, and `chips` are effectively free.** Start there and only descend
  when the question needs it.
- **`l2`/`l3` scale with region *count*, not tile size**, so a fragmented map is
  what makes them expensive — this fixture has 1,583 regions at 12 px minimum.
  `--min-area 50` (m²) and `--limit N` are the levers, and both report what they
  dropped. The quadtree's cost moves the same way: a real map with cleaner,
  larger regions compresses much harder than this noisy fixture does.

Everything is TSV, not JSON: roughly 2–3× fewer tokens for the same content,
with one legend emitted separately instead of repeating key names per row.

`segmap compare --dump out/` writes each level to a file so you can read them.
`--count-tokens` swaps the offline estimate for exact counts from the API.

---

## What's in the box

**[`taxonomy.py`](src/segmap_digest/taxonomy.py)** — the 45 classes plus the
structure hiding inside the flat list:

- five overlapping sub-ontologies (artifact / anthropogenic / hydrology /
  pedology / lithology×geomorphology / vegetation / land use)
- the **lithology × morphology grid** — sparse and asymmetric on purpose:
  limestone has six morphologies, chalk exactly one. Anything off-grid is a
  definitional error, not a judgement call.
- **ordinal series** — `DryGrassland → Batha → Garigue → Maquis` is the
  Mediterranean degradation gradient. A Garigue/Batha confusion is a near-miss;
  a Garigue/House confusion is not. `class_distance()` encodes that; plain
  cross-entropy does not.
- **co-occurrence priors** — terra rossa forms on hard carbonate, rendzina on
  chalk and marl, nari *caps* units, a dip slope needs low aspect variance,
  badlands need relief. World knowledge the CNN never had access to.
- a 45 → 9 **superclass** collapse for coarse reasoning and legible colourmaps.

**[`index.py`](src/segmap_digest/index.py)** — Stage 0, raster → symbolic index.
Two indices because they answer different questions:

- **Region index**: connected components with area, perimeter, compactness,
  elongation, mean/σ slope, elevation, **aspect circular variance**, and a
  neighbour graph with shared boundary lengths. Tells you *why*.
- **Chip index**: fixed grid at the detector's input size, with per-chip class
  histogram, entropy, class-pair interface lengths, and distance-to-anchor-class
  fields. The *unit of spend* for detection triage.

Both are query-independent: computed once per tile, reused by every question.
The distance transforms are the expensive part and are deliberately outside any
per-query loop.

**[`audit.py`](src/segmap_digest/audit.py)** — the cheap half of the reasoning
layer. Applies the priors deterministically and emits *candidates*:

```
rid   class            kind                       sev  area_m2  message
434   HydromorpicSoil  slope-violation           0.80      247  mean slope 6.7 deg outside required 0.0-5.0
1490  BasaltRockyTerr  isolated-speck            0.65       60  island fully enclosed by Garigue (dist 1.00)
4     Clutter          oov-candidate             0.50       32  outside the 45-class vocabulary; detector target
245   TerraRosa        missing-expected-context  0.45      351  no boundary with any hard-carbonate unit
```

Deciding which candidates are real is the LLM's job. *Enumerating* them is not,
and paying an LLM to enumerate would defeat the point.

**[`ask.py`](src/segmap_digest/ask.py)** — optional. Sends a digest plus a
question to Claude (`claude-opus-5`, adaptive thinking, class legend behind a
prompt-cache breakpoint so iterating on questions against one tile is cheap).

```bash
export ANTHROPIC_API_KEY=...
segmap ask --level l2 --with-audit \
  "Which regions are most likely misclassified, and what data would settle each?"
```

---

## Design rules this POC follows

These are the things that were easy to get wrong:

- **The LLM is not in a per-item loop.** For triage, the model compiles a query
  into a small cached policy; a deterministic scorer applies it to millions of
  chips. Amortized reasoning cost per chip ≈ 0.
- **Never truncate silently.** Every `--limit` emits a `# NOTE: N omitted` line.
  A cap that reads as "this is everything" is how you ship a wrong answer. This
  is enforced by a test.
- **Numbers come from code, not from the model.** Areas, counts, distances, and
  adjacency are computed here and handed over as facts.
- **Colourised label maps are for humans.** Asking a VLM to read exact colours
  off a 45-entry legend is a task VLMs are bad at; `loader.load()` refuses a
  3-band PNG rather than guessing class ids back out of RGB.

---

## Next steps

1. **Write the real class definitions.** The ones in `taxonomy.py` are a
   starting point and need expert review — they are the text the model reasons
   over, and they're a flat file, not code.
2. **Point it at a real tile** (`-i tile.tif` keeps the georeference) and a real
   DEM. For this taxonomy the DEM is worth more than RGB: dip slope, terrace,
   badlands and boulder field are all *morphology*, which slope/aspect/curvature
   resolves and colour does not.
3. **Add the lithology map join.** Limestone / dolomite / nari are not separable
   in RGB even for a human expert — that separation is a map lookup, and it's
   the largest single accuracy lever available.
4. **Then** build the detection-triage policy layer on top of the chip index
   (policy JSON → deterministic scorer → budgeted selection → outcome log with a
   rejected-chip control set).

## Tests

```bash
pytest tests -q
```
