# Quickstart — ten minutes, no API key

**reasoning-terrain (RT)** turns a 47-class Smart Terrain segmentation raster
into things a reasoning model can work with, and answers questions about it.
**Most of it needs no API key and no data.** Start there.

Every command and every block of output below was run on this checkout. Timings
are wall clock on a 28-core box; the memory figures are peak RSS from
`/usr/bin/time -v`.

**If you have exactly ten minutes:**

```bash
pip install -e '.[geo]'
T=data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif
segmap report -o out/demo          # 17 s, needs no data at all
segmap index -i $T                 # 29 s, once
segmap solve s5 -i $T --query describe
```

Then work down §4. The whole 1194 Mpx AOI is §6; it needs a 24-minute build
first, so kick that off now if you want it.

---

## 1. Install (30 seconds)

```bash
pip install -e '.[geo]'      # numpy, scipy, pillow, rasterio
```

`[geo]` is what lets it read GeoTIFFs. Skip it and you get the synthetic
fixture only.

Re-run this even if you installed before — the repo was renamed to
`reasoning-terrain`, and an editable install made under the old directory name
still points at a path that no longer exists (`segmap` then fails with
`ModuleNotFoundError: No module named 'segmap_digest'`).

## 2. The zero-data path

`synth.py` generates a spatially coherent fixture tile — terra rossa on hard
carbonate, badlands only where it's steep, a village with roads and parked cars
— so every command works with no `-i` at all.

```bash
segmap compare                        # the representation ladder, with token counts
segmap solve s5 --query "corridor PavedRoad"
segmap report -o out/demo             # the whole pipeline as one HTML page
```

```
$ segmap compare
level      chars    tokens  description
--------------------------------------------------------------
raw      3145728    786432  label raster as text -- never do this
l0          1371       371  class histogram
l1          7427      2007  NxN grid digest
l1q        40388     10916  quadtree
l2        177259     47908  region table
l3        135176     36534  adjacency graph
chips       1992       538  chip index
```

`segmap report -o out/demo` takes 17 s on the fixture and writes
`out/demo/index.html`. Open it. It is the fastest way to see what the whole
thing does, and every section says what it assumed.

## 3. Real data: index once, then ask

The sinai drop is 20 GeoTIFFs of 8192×8192 px at 0.5 m/px. Building the region
and chip indices over the whole mosaic is expensive and used to happen *on every
single query*. Now it happens once:

```bash
segmap index -i data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif
```

```
# input: data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif -> 8192x8192 px @ 0.496 m/px, 67.1 Mpx, 100.0% classified
# building regions index, this will take a while (cached afterwards in .segmap_cache/regions-2249af801549f08c)
# built regions index in 14.7s and cached it
# building chips index, this will take a while (cached afterwards in .segmap_cache/chips-c9a70cd7c8228ac4)
# built chips index in 12.9s and cached it
indexed data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif: 11058 regions, 1024 chips of 256 px -> .segmap_cache
```

Every later command finds that cache and says so on stderr:

```
# using cached regions index (built 5min ago, 15s to build, .segmap_cache/regions-2249af801549f08c)
```

| | cold | warm | cache on disk |
|---|---:|---:|---:|
| one sinai tile (67 Mpx, 11,058 regions) | 16.5 s, 1.6 GB | **1.4 s, 0.7 GB** | 269 MB |
| sinai mosaic (1194 Mpx, 133,976 regions) | 24 min 24 s, 47.0 GB | **1.5 s, 1.5 GB** | 7.2 GB |

Three things get cached: the region index, the chip index, and — for a directory
input only — the stitched mosaic itself. The cache invalidates on the input
files' paths, mtimes and sizes, the GSD override, the crop window, `--min-px`,
chip size, anchor classes, and a schema version constant. Change any of them and
it rebuilds rather than answering from the old one. `segmap index --list` shows
what is cached, `--refresh-index` forces a rebuild, `--no-cache` bypasses it
entirely, and `$SEGMAP_CACHE` or `--cache DIR` moves it.

```
$ segmap index --list
kind            built   build_s   size_mb  source
chips        6min ago      1071       1.5  .../data/incoming/sinai
mosaic        41s ago        19    2388.9  .../data/incoming/sinai
regions      24min ago      370    4783.8  .../data/incoming/sinai
regions      31min ago       15     268.9  .../data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif
regions      10min ago        1      32.0  .../tile_cropped_x802_y847_z0.5.tif  [SUBSET]
```

It is just a directory of `.npy` and `meta.json`; delete it to reclaim the disk.

**Pointing at one tile or at the whole AOI is the same command.** `-i <file.tif>`
is one tile, `-i <directory>` mosaics every GeoTIFF in it by affine transform.
Prefer the mosaic when you can afford it: regions are connected components, so
per-tile analysis cuts every region at the seam.

Everything from here on uses one tile. Set it once:

```bash
T=data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif
```

### When you can't afford it: `--max-mpx`

`--max-mpx N` takes a centred window of at most N megapixels **at full
resolution** and tells you exactly what it did:

```
$ segmap solve s5 -i $T --max-mpx 8 --query "count Batha minarea 200"
# input: data/incoming/sinai/tile_cropped_x802_y847_z0.5.tif -> 8192x8192 px @ 0.496 m/px, 67.1 Mpx, 100.0% classified
# SUBSET: --max-mpx 8 cropped this AOI to rows 2682:5510, cols 2682:5510 -- 2828x2828 px = 8.0 Mpx of 67.1 Mpx (11.9% of the extent), holding 11.9% of the AOI's classified pixels. Full resolution, no downsampling. Every number derived from this raster describes that window only and must not be reported as an AOI-wide figure.
# built regions index in 1.1s and cached it
# count: count Batha minarea 200
= 5.00
# evidence: region ids 492, 471, 518, 455, 462
```

It **crops rather than strides**. Striding is nearest-neighbour downsampling of
a label raster: it deletes every feature narrower than the stride — roads,
wadis and walls are 1–4 px here — merges regions that never touched, and shifts
every area and region count by an amount nobody can predict. A crop changes
exactly one thing, which ground you are looking at, and the note above says so
in every output that carries it, including the question handed to the model in
`segmap ask` and the header of `segmap report`.

The subset is part of the cache key, so a cropped index and a full one are
different entries and neither can be served for the other.

---

## 4. Ten questions that work

All of these run offline. `segmap solve s5 --query` is exactly the surface the
model gets in `segmap ask` — same code, same numbers.

The `# input:` and `# using cached` stderr lines are elided below.

### 1. What am I even looking at?

*Tests: the composition digest, and the nodata denominator.*

```
$ segmap solve s5 -i $T --query describe
# describe (20 classes present): describe
# fractions are over 67108864 classified pixels (0.0% of the extent is nodata and excluded); region counts are after the index's min-area cut, so small specks are missing from them
class_id	name	superclass	frac	area_m2	n_regions
17	Rendzina	soil	0.6860	11334244	2686
26	DryGrassland	vegetation	0.1309	2163353	5124
15	TerraRosa	soil	0.0891	1472069	513
19	ClayeyDeepSoil	soil	0.0636	1049989	873
23	Batha	vegetation	0.0118	195089	1010
12	MaralBadlands	rock	0.0086	141585	74
9	DirtRoad	road	0.0025	40602	23
29	LimestoneStoneyTerrain	rock	0.0023	37896	304
33	LimestoneTerrace	rock	0.0020	32628	140
```

Two-thirds Rendzina. This is arid rangeland with dirt tracks, no settlement.

### 2. How many badlands patches are big enough to matter?

*Tests: counting **regions** (not objects) with an area filter, and that every
count comes back with the ids it was computed from.*

```
$ segmap solve s5 -i $T --query "count MaralBadlands minarea 500"
# count: count MaralBadlands minarea 500
= 8.00
# evidence: region ids 98, 118, 69, 70, 63, 116, 85, 135
```

### 3. How much dirt road is there?

*Tests: area arithmetic from pixel counts and the GSD — the thing that was ten
orders of magnitude wrong before the geographic-CRS fix.*

```
$ segmap solve s5 -i $T --query "area DirtRoad"
# area_m2: area DirtRoad
= 40,593.83
# evidence: region ids 36, 34, 20, 18, 28, 16, 19, 17, 32, 29, 38, 31, 25, 33, 37, 21, 30, 26, 24, 27, 22, 23, 35
```

4 hectares of track across 23 regions, on a 16.6 km² tile.

### 4. Which limestone terraces, specifically?

*Tests: `find` — the verb that hands back ids and centroid pixels you can go and
look at.*

```
$ segmap solve s5 -i $T --query "find LimestoneTerrace minarea 2000"
# find (3 regions): find LimestoneTerrace minarea 2000
rid	class	area_m2	slope	cy	cx
10940	LimestoneTerrace	6843	0.0	1334	4692
10929	LimestoneTerrace	4006	0.0	854	5562
10911	LimestoneTerrace	2005	0.0	154	6678
# evidence: region ids 10940, 10929, 10911
```

Note `slope 0.0`: there is no DEM. See §7.

### 5. How close does the badlands get to the paved road?

*Tests: pixel-to-pixel closest approach, not centroid distance, with both
endpoints returned so the number can be checked on the map. This is the
expensive verb — one full distance transform per call.*

```
$ segmap solve s5 -i $T --query "distance MaralBadlands PavedRoad"
# distance_m: distance MaralBadlands PavedRoad
# closest approach, pixel to pixel
= 3,477.17
class_a	rid_a	ay	ax	class_b	rid_b	by	bx	metres
MaralBadlands	116	1687	3567	PavedRoad	15	8190	6179	3477.17
# evidence: region ids 116, 15
```

7.7 s and 2.5 GB even warm — the distance transform is over the full raster and
is not cached, because it depends on the class pair.

### 6. Where can a wheeled vehicle actually get to?

*Tests: the S4 → S5 chain, and whether the answer admits what it rests on.*

```
$ segmap solve s5 -i $T --query "corridor DirtRoad vehicle wheeled"
# corridor (wheeled, seeded from DirtRoad): corridor DirtRoad vehicle wheeled
# 99.0% of the classified area is reachable from DirtRoad without leaving ground passable to a wheeled vehicle (threshold 0.45, slope limit 25.0 deg)
= 16,356,043.88
component	area_m2	tile_frac	cy	cx
1	16356044	99.0%	4123	4119
# evidence: component ids 1
```

**This is a weak answer** and it is in §5 for that reason — but notice it says
which two guessed constants it rests on.

### 7. Slope filter with no DEM

*Tests: abstention. Every `mean_slope` here is 0.0, which is a valid-looking
number meaning "unmeasured". Filtering on it would return a confident, wrong
answer.*

```
$ segmap solve s5 -i $T --query "find TerraRosa slope > 10"
query error: no DEM is attached to this tile, so every region's slope is unmeasured (stored as 0.0). A slope filter cannot be applied; attach a DEM with --dem and ask again
```

### 8. A question the map cannot answer

*Tests: refusal in words, not an empty table. "No fences found" would be a wrong
answer, not a missing one.*

```
$ segmap solve s5 -i $T --query "count Fence"
query error: 'Fence' is outside the 47-class vocabulary: there is no fence class; a fence is also sub-metre wide and would not survive segmentation at this GSD
```

Try `count Truck`, `count School`, `count Building` too — each refuses with its
own reason.

### 9. A measured zero

*Tests: the other side of the same coin. `Car` **is** in the vocabulary, so zero
is a finding about this ground, not a limit of the tool.*

```
$ segmap solve s5 -i $T --query "count Car"
# count: count Car
= 0.00
```

### 10. Which labels contradict their own context?

*Tests: S1, the deterministic audit — and specifically whether it tells you what
it did **not** check.*

```
$ segmap solve s1 -i $T --budget 6
# S1 consistency audit -- 11058 regions, 34 root causes, 2484 candidate findings
# coverage: 8/20 classes checked, 1.4% of area. Read the coverage block before reading the findings.

## coverage -- what was actually examined
# 8 of 20 classes present on this map have at least one active substantive check (5 check families exist; 47 classes are defined).
# those classes hold 1.4% of classified region area.
#
# A CLASS WITH NO ACTIVE CHECK IS UNEXAMINED, NOT CLEAN. ...
# ABSTAINED -- no DEM: every mean_slope and aspect_circvar is 0.0, which is a valid-looking number meaning 'unmeasured'. The slope and aspect priors did not run rather than score against it.

class	area_m2	pct_map	substantive	geometry_only
Rendzina	11333220	68.61%	NONE -- UNEXAMINED	enclosed-speck
DryGrassland	2161686	13.09%	NONE -- UNEXAMINED	enclosed-speck
TerraRosa	1471895	8.91%	NONE -- UNEXAMINED	enclosed-speck
```

The coverage block is the useful part: with no DEM, 98.6% of the map's area has
no substantive check against it. (S1 is under active change; the exact counts
will move, the coverage block is the part to read either way.)

### Bonus: which chips are worth sending to a detector?

```
$ segmap solve s3 -i $T --policy oov
# S3 triage -- policy find-out-of-vocabulary@v0 (regime R3)
chips total          1024
hard-excluded        0 (0%)
selected (budget)    205 (20%)
control set          25 (3% of rejected -- MANDATORY, this is the only unbiased recall signal)
cost reduction       78%
score captured       100% of total prior mass

# WARNING: 'score captured' is prior mass, NOT recall. Recall requires detector outcomes; until then this number is a plausibility check only.
```

---

## 5. Known to be weak

Nothing here is dropped from the list above; these are the three that gave
answers I would not hand to anyone.

**`corridor` on sinai returns 99% of the tile.** Trafficability is a per-class
score plus a slope term. With no DEM the slope term is inert, and this tile is
68% Rendzina and 13% DryGrassland — both scored as easily passable — so almost
everything connects to almost everything. The number is arithmetically correct
and analytically useless. It becomes informative when a DEM is attached, and not
before. (On the synthetic fixture, which has a DEM, the same query returns
63.5%.)

**`count Car` prints `= 0.00` with no comment.** The tool layer that `segmap ask`
uses returns this as `status: empty` with the sentence "Computed, and the answer
is zero. This is a measured absence, not a failed query." The bare CLI renderer
does not repeat that. Read a zero from `solve s5` as a measured zero, but the
CLI is not helping you.

**S1's finding *counts* mean nothing yet.** 2484 candidates on one tile, with no
measured false-positive rate — that is S1's own blocking open question. The
coverage block is trustworthy; the findings list is a worklist, not a verdict.

---

## 6. The mosaic

The whole sinai AOI — 20 tiles, 31822×37535 px, 1194 Mpx, 59.3% of the extent
classified — is one command:

```bash
segmap index -i data/incoming/sinai/          # 24 min, once
segmap solve s5 -i data/incoming/sinai/ --query "count House minarea 40"
```

Building it took **24 min 24 s and 47.0 GB peak RSS**, split three ways:

| stage | cold | warm | on disk |
|---|---:|---:|---:|
| stitch 20 GeoTIFFs into one raster | 19 s | 0.0 s (mmapped) | 2.4 GB |
| region index (133,976 regions) | 370 s | 0.67 s | 4.8 GB |
| chip index (18,104 chips of 256 px) | 1071 s | 0.11 s | 1.5 MB |

Warm, on the full AOI:

```
$ segmap solve s5 -i data/incoming/sinai/ --query "count House minarea 40"
# using cached mosaic (31822x37535 px, stitched 17s ago, .segmap_cache/mosaic-b8cbe61296f8b2b3)
# using cached regions index (built 24min ago, 370s to build, ...)
= 123.00
```
**1.5 s, 1.5 GB.** So yes — the mosaic is interactive now.

| warm query on the full AOI | wall | peak RSS |
|---|---:|---:|
| `count House minarea 40` | 1.5 s | 1.5 GB |
| `find House minarea 100` (80 regions) | 1.5 s | 1.5 GB |
| `solve s3 --policy structures` (18,104 chips) | 1.0 s | 1.3 GB |
| `describe` | 6.3 s | 8.8 GB |

`describe` is the slow one because it takes `labels[valid]` — a 709 M-element
copy — before histogramming. That is pre-existing and unrelated to the cache.

**Two things to know before you wait 24 minutes.**

The chip index is 70% of the cold cost: seven full-raster distance transforms,
one per anchor class, at 1194 Mpx each. If you only want `s5` / `s1` / `s2`
questions you do not need it — those use the region index alone, and it builds
in 6 minutes. `segmap index` builds both because `s3` and `report` need chips.

And 47 GB of peak RSS is a real requirement for the cold build. On a smaller
box, index one tile at a time, or use `--max-mpx`.

---

## 7. What to trust

**Trustworthy:** everything counted, measured or connected-component'd from the
label raster — class fractions, areas, region counts, adjacency, pixel-to-pixel
distances — plus every refusal and every abstention, which are enforced in code
rather than left to a prompt.

**Not trustworthy:** anything touching slope, aspect or morphology, because no
DEM is attached and those fields are 0.0 meaning "unmeasured" (this switches off
roughly half the taxonomy and most of S1 and S4); the class definitions in
`taxonomy.py`, which are a starting point pending expert review and are the text
all the reasoning runs on; every heuristic constant under `solutions/`, which is
a guess; and the live LLM path, which has never been run against the real API.

---

## 8. The API-key path

```bash
pip install -e '.[llm]'
export ANTHROPIC_API_KEY=sk-...
segmap ask -i $T "How much of this tile is bare rock, and where is the roughest ground?"
```

`segmap ask` gives the model the six query verbs as tools and lets it plan. It
cannot state a count, an area or a distance that did not come back from a tool
call, because it has no other way to see the raster. The answer is followed by
the call log, and every entry in that log is an `s5_query` string you can rerun
yourself.

```bash
segmap tools                 # the tool definitions, no SDK and no key needed
segmap ask --digest --level l2 --with-audit "Which regions look misclassified?"
```

`--digest` is the un-grounded path: the model reads a table instead of calling
tools, so treat arithmetic in those answers as an estimate.

Without a key you get this, instead of a `TypeError` about HTTP headers:

```
$ segmap ask -i $T "How much of this tile is bare rock?"
# using cached regions index (built 32min ago, 15s to build, .segmap_cache/regions-2249af801549f08c)
# 11058 regions indexed; answering with tool calls
`segmap ask` found no API credentials: export ANTHROPIC_API_KEY and try again. Everything except the model call runs offline -- `segmap solve s5 --query '...'` computes the same numbers, `segmap tools` prints the surface the model is given, and `segmap report` writes the whole page.
```

**This path is still unexercised against the live API.** `anthropic 1.0.0` is
installed and imports; there were no credentials on the machine this was written
on (`ANTHROPIC_API_KEY` unset, no `ant` CLI), so not one real request has been
made. The first thing to run with a key is the round-trip check in
`docs/solutions/S5-query.md`: ask a question whose answer you already know from
`solve s5`, and check the model's prose against its own call log line by line.
Treat the first live run as a test of RT, not as an answer.
