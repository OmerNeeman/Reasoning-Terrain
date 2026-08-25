# segmap-digest-poc

Turn a **47-class Smart Terrain segmentation raster** into representations a
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
segmap legend --full                 # the 47 class definitions
```

New here? [**QUICKSTART.md**](QUICKSTART.md) is ten minutes, no API key, with
ten real questions and their real output.

### On real exports

```bash
segmap index  -i data/aoi/                           # build the index once
segmap report -i tile.tif        -o out/aoi          # one tile
segmap report -i data/aoi/       -o out/aoi_mosaic   # a directory = mosaic
segmap report -i data/aoi/ --classes ids.json -o out/aoi_mosaic
```

**The index is cached.** `build_regions` / `build_chips` are query-independent
by design, and used to be rebuilt by every single command — on the sinai mosaic
that is **24 min and 47 GB per question**. [`cache.py`](src/segmap_digest/cache.py)
writes them to `.segmap_cache/` as `.npy` plus a `meta.json`, keyed on the input
files' paths, mtimes and sizes, the GSD override, the crop window, `--min-px`,
chip size, anchor classes and a schema version constant, so a stale cache misses
rather than answering. The two big arrays — the region-id raster and, for a
directory input, the stitched mosaic — are memory-mapped rather than read, so
the warm path is 0.8 s and a gigabyte instead of 24 min and 47 GB:

| sinai mosaic, 1194 Mpx | cold | warm |
|---|---:|---:|
| `solve s5 --query "count House minarea 40"` | 24 min 24 s, 47.0 GB | **1.5 s, 1.5 GB** |

Every command says on stderr which cache it used. `segmap index -i <path>`
builds explicitly, `--list` shows what is cached, `--refresh-index` rebuilds,
`--no-cache` bypasses, `$SEGMAP_CACHE` relocates.

`--max-mpx N` crops an AOI to a centred window of N megapixels **at full
resolution** when the whole thing is too expensive to wait for. It crops rather
than strides because striding a label raster deletes every feature narrower than
the stride; and it states the window, its share of the extent and its share of
the AOI's classified pixels in every output that carries it, including the
question handed to the model.

Three things a real Smart Terrain GeoTIFF does that the fixture does not, all
handled in [`loader.py`](src/segmap_digest/loader.py):

- **Class ids on the wire are sparse, 0..241**, not the dense 0..46 in
  `taxonomy.py`. The exporter ships the translation as an
  `ID_TO_LABEL_MAPPING` GeoTIFF tag and we read it automatically, mapping **by
  name**. A name the taxonomy does not define is an error, not something to
  fold into `Unclassified`. Tiles without the tag need `--classes` —
  [`examples/smart_terrain_class_ids.json`](examples/smart_terrain_class_ids.json)
  is the mapping as read off the sinai drop.
- **Nodata is 0, which is also `Unclassified`'s id.** The two are not
  distinguishable in the file. Every 0 is treated as no-data, every fraction is
  over classified pixels only, and the report says how much was excluded. If the
  segmenter genuinely emits `Unclassified`, that class is being thrown away and
  the exporter needs to give nodata a value of its own.
- **The CRS is geographic.** A 0.5 m/px export in EPSG:4326 has a transform of
  `5e-06`; reading that as metres makes every area in the report ten orders of
  magnitude too small.

Point `-i` at a **directory** to mosaic. Regions are connected components, so
per-tile analysis cuts every region at the seam — one wadi across four tiles
becomes four regions with four wrong areas and four wrong neighbour lists.
Placement comes from the affine transforms; mixed CRS or resolution raises
rather than resampling, because resampling a *label* raster invents classes at
every boundary. The mosaic is cropped to the bounding box of actual data, which
drops no labelled pixel.

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

**[`taxonomy.py`](src/segmap_digest/taxonomy.py)** — the 47 classes plus the
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
- a 47 → 9 **superclass** collapse for coarse reasoning and legible colourmaps.

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
4     Clutter          oov-candidate             0.50       32  outside the 47-class vocabulary; detector target
245   TerraRosa        missing-expected-context  0.45      351  no boundary with any hard-carbonate unit
```

Deciding which candidates are real is the LLM's job. *Enumerating* them is not,
and paying an LLM to enumerate would defeat the point.

**[`tools.py`](src/segmap_digest/tools.py) + [`ask.py`](src/segmap_digest/ask.py)**
— optional. `segmap ask` gives Claude (`claude-opus-5`, adaptive thinking, class
legend behind a prompt-cache breakpoint) the **S5 query verbs as tools** and runs
the SDK's tool runner over them. The model chooses the verb and the arguments;
`s5_query` does the arithmetic. It cannot return a count, an area or a distance
that the code did not compute, because it has no other way to see the raster.

```bash
export ANTHROPIC_API_KEY=...
segmap ask "How much of this tile can a wheeled vehicle reach from the road network?"
segmap tools                    # the tool definitions, no SDK or key needed
```

Every tool result carries the region ids it was computed from, and the answer is
followed by the call log, so any number in it can be reproduced with
`segmap solve s5 --query '...'`. Results come back as one of four statuses:
`ok`, `empty` (a measured zero — a real finding), `unanswerable` (vehicle type,
building function, fence presence, anything about colour: the label raster
cannot answer it, and an empty table saying "none found" would be a wrong
answer), or `error`.

For interpretive questions no verb covers, `--digest` keeps the old path — send
a digest and let the model read it:

```bash
segmap ask --digest --level l2 --with-audit \
  "Which regions are most likely misclassified, and what data would settle each?"
```

---

## The six solutions

Shared infrastructure (taxonomy · index · digests) feeds six map-consuming
solutions. Each has a **runnable naive implementation** and a doc with its
heuristics, open questions, and options.

```bash
segmap solve s1                                     # audit
segmap solve s2 --region 245                        # adjudicate
segmap solve s3 --policy vehicles --chip 128        # triage
segmap solve s4 --product trafficability --wet      # products
segmap solve s5 --query "corridor PavedRoad"        # query
segmap solve s6                                     # change
```

| | Solution | What it answers | Docs |
|---|---|---|---|
| **S1** | audit | which labels contradict their context, slope, or geometry | [S1](docs/solutions/S1-audit.md) |
| **S2** | adjudicate | given a suspect region, what *is* it — or is it undecidable | [S2](docs/solutions/S2-adjudicate.md) |
| **S3** | triage | which chips are worth sending to an expensive detector | [S3](docs/solutions/S3-triage.md) |
| **S4** | products | trafficability, concealment, drainage, fire fuel | [S4](docs/solutions/S4-products.md) |
| **S5** | query | counts, areas, distances, movement corridors | [S5](docs/solutions/S5-query.md) |
| **S6** | change | semantic diff: phenology vs succession vs real change | [S6](docs/solutions/S6-change.md) |

Three things the naive versions already get right, because they're the parts
that are easy to get wrong later:

- **S2 abstains.** `UNDECIDABLE-NEEDS-geological-map` is a first-class verdict.
  Limestone/Dolomite/Nari aren't separable without that layer, and a system that
  picks one confidently manufactures corruptions. Net gain is corrections minus
  corruptions, and it can be negative.
- **S3 keeps a control set.** 3% of *rejected* chips get dispatched anyway. It's
  the only unbiased recall signal you'll ever have; without it there is no
  evidence the other 90% of savings is safe, and no alarm when you over-prune.
- **S6 separates phenology from change.** On the fixture, ~76% of differing area
  is plausibly real — a raw pixel diff would have called 100% of it change. A
  `Limestone→Dolomite` transition is reported as `impossible`, i.e. a label
  error, not a landslide.

**Every heuristic constant in `solutions/` is a guess.** They're grouped at the
top of each module, named, and tabulated in the docs with a "where this should
actually come from" column. Don't treat any number they produce as a finding.

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
  off a 47-entry legend is a task VLMs are bad at; `loader.load()` refuses a
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
4. **Answer the blocking question in each solution doc.** They're the first
   numbered item in each:
   - S1 — what's the false-positive rate? (needs ~150 reviewed regions)
   - S2 — is the true class even in the shortlist?
   - S3 — what's the detector's cost model: per call, per megapixel, or per second?
   - S4 — is a 0–1 score the right output, or GO/SLOW-GO/NO-GO?
   - S5 — ~~tool calls or code against the index?~~ **decided: tool calls.**
     Next: ten real analyst questions, to find out whether six verbs is the
     right surface — and one live run, which has not happened yet.
   - S6 — run the null test and the 1-px-shift test; co-registration is the threat.
5. **Then** replace the guessed constants with measured ones, starting with S3's
   `class_weights` (from a detector outcome log) and S4's per-class `traffic`
   values (from whoever owns vehicle doctrine).

## Tests

```bash
pytest tests -q
```
