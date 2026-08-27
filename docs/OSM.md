# The OSM layer — spec

> Status: **spec + stage plan**. Every assertion below is checkable, and the
> stage that proves it is named. Numbers marked `[measured]` came from a command
> in this repo against `data/incoming/`; everything else is a guess and says so.

## The problem

Two problems, one layer.

**1. The reasoning unit is wrong.** Connected components are the only spatial
unit RT has, and there are 95,170 of them on aza and 133,976 on sinai
`[measured]`. Nothing in that number is addressable by a human: "region 41,882"
is not a place. Analysts reason about *blocks* — the ground bounded by streets —
and about *intersections*. That partition already exists, in OSM, for every
populated AOI RT will ever see.

**2. The segmentation has no reference map.** RT's audit compares the labels
only against themselves (geometry, slope, neighbours). For rock it says so and
abstains — `UNDECIDABLE-NEEDS-geological-map`. For roads, buildings and water
the reference map *is available and free*, and RT was not reading it. On the aza
AOI, 28.6% of the ground under an OSM road corridor is labelled `Shadow` and a
further 24.4% `MaralBadlands` `[measured]` — the segmenter cannot see a street
it cannot see, and OSM can tell it one is there.

## Non-goals

- **OSM never edits the label raster.** Burning OSM roads into `labels` would
  destroy the only thing that makes this layer valuable — the ability to compare
  two independent maps — and would silently import OSM's errors into RT's
  output. The layer is parallel, always.
- **No routing engine.** The road graph exists to name places and to bound
  blocks, not to compute travel times.
- **No LLM.** S1–S4 only, deterministic. The layer is what an LLM would later
  read; nothing here calls one.

## D — domain requirements

| # | Assertion | Proof |
|---|---|---|
| D-1 | An OSM way is a *vector*: node ids, lon/lat geometry, tags. Tag → semantics mapping lives in one table (`osm/tags.py`) and every entry is either sourced or marked a guess. | S1 |
| D-2 | The direction of trust is **stated, not defaulted**: `osm/trust.py` holds the owner's policy — OSM is the reference for existence and identity, ST is the latest word on state — and every finding is tagged with which axis it is on. Neither map is called wrong. | S5 |
| D-3 | OSM completeness is unknown and varies by AOI and by tag. A region with no OSM feature near it is `OSM-SILENT`, never `OSM-SAYS-NO`. | S5, S6 |
| D-4 | A block is the ground enclosed by the road network, not a grid cell. Blocks tile the AOI's classified area exactly: every valid pixel is in exactly one block or in the road corridor. | S3 |
| D-5 | Road corridor width comes from tags (`width`, `lanes`, `highway` grade, in that precedence), never from a single constant. | S2 |

## F — functional requirements

| # | Assertion | Proof |
|---|---|---|
| F-1 | `osm.fetch` returns an `OsmVectors` from (a) a local Overpass-JSON / GeoJSON / `.osm` XML file, or (b) an Overpass query over the raster's own bbox, cached to disk. A cached snapshot is used without touching the network. | S1 |
| F-2 | A snapshot on disk carries its provenance: query, bbox, server, UTC fetch time, element counts. Re-running with the snapshot present reproduces byte-identically. | S1 |
| F-3 | Projection lon/lat → pixel uses the raster's own affine. A raster with no transform (`.npy`) raises with a message naming the cause; a non-4326 CRS is reprojected via rasterio, or raises if rasterio is absent. | S2 |
| F-4 | `osm.align` reports the integer pixel shift that maximises overlap between the burned road corridor and ST's road classes, and the overlap at zero shift. Measured on aza: best shift (-8, +8) px ≈ 1.0 m, +5.3% overlap over zero shift `[measured]` — i.e. co-registered. | S2 |
| F-5 | `build_blocks` returns blocks with id, area, bbox, centroid, the ST class histogram inside it, the OSM ways on its boundary, and the street *names* among them. Measured on aza: 99 blocks, median 3,896 m², 93 ≥ 200 m² holding 99.9% of the free area `[measured]`. | S3 |
| F-6 | `build_road_graph` returns intersections (nodes shared by ≥2 ways, or way endpoints) and segments between them, each with length, grade, name, surface. Measured on aza: 555 shared nodes over 404 ways `[measured]`. | S3 |
| F-7 | Both are query-independent and cached exactly like the region and chip indices, keyed on the OSM snapshot fingerprint as well as the raster's. A stale key misses rather than answering. | S4 |
| F-8 | A `blocks` digest level emits the block table and the street table as TSV, with the standard `# NOTE: N omitted` on any truncation. | S4 |
| F-9 | S1 gains reference-map checks: road corridor with no ST road under it, ST road with no OSM way near it, ST road *grade* against OSM `surface`, `House` against building footprints, `Water` against waterways. Each rolls up on a cause like every other finding. | S5 |
| F-10 | S2 gains a reference-evidence term. With no OSM layer, or where OSM is silent, S2's scores are **bit-identical to today's**. | S6 |
| F-11 | S3 gains per-chip OSM features (`osm.dist_road`, `osm.road_frac`, `osm.building_frac`, `osm.n_junctions`) usable in policy rules, and can take the **block** as the unit of spend instead of the grid square. | S7 |
| F-12 | S4 products may take OSM overlays: roads as known-passable in `trafficability`, buildings/barriers as blockers there, buildings as cover *beside* them in `concealment`, footprints and `built_landuse` in `built_fabric`, footprints resolving `Unclassified`/`Clutter` in `change_volatility`. (`drainage` and `fire_fuel`, the two products this row used to name, were retired -- `s4_products.PRODUCTS` is now exactly those four.) Every product states what the overlay changed. | S8 |
| F-13 | Class notes: a Markdown sidecar with one section per class and named fields. Of these, **only `confused_with` has a consumer today** -- `s2_adjudicate` uses it for the candidate shortlist and quotes its reason text. `season`, `never` and `scale` are recorded for planned consumers and read by nothing yet; the rest is prose for the model. Missing file = today's behaviour. | S9 |
| F-14 | Bare `segmap notes` (with `-i`, and no `--class`/`--check`/`--template`) ranks the 47 classes by pixel share in the loaded AOI against which notes fields are filled, so effort goes where the map actually is. Coverage is the default, not a flag. | S9 |

## N — non-functional requirements

| # | Assertion | Proof |
|---|---|---|
| N-1 | New hard dependencies: none. numpy + scipy + stdlib. rasterio is used when present (polygon burn, reprojection) and the pure-numpy path is used when it is not. | S1–S3 |
| N-2 | Network access is optional and never implicit: an Overpass fetch happens only when no snapshot is cached, and says so on stderr with the query it is about to run. | S1 |
| N-3 | Cost on the largest real AOI (sinai, 1194 Mpx, 6,038 ways): the burn + block build is cached, and the warm path is memory-mapped like the region index. Cold cost is stated in the output, not hidden. | S4 |
| N-4 | The block partition degrades honestly. Where OSM roads enclose nothing, the AOI is one block and the output says so — it does not invent a partition. | S3 |

## U — UX requirements

| # | Assertion | Proof |
|---|---|---|
| U-1 | Every OSM-derived number in any output is marked as OSM-derived. A reader must never have to guess which map a figure came from. | S4–S8 |
| U-2 | Blocks are named by their bounding streets when OSM has names (`block 84 — bounded by شارع النخيل, شارع الشعف`), by grade when it does not (`bounded by 3 residential ways`), and by neither only when the block touches no named or graded way. | S3 |
| U-3 | An OSM fetch failure (offline, rate-limited, empty result) is a clear message naming what to do — not a stack trace, and not a silent empty layer that reads as "OSM says there is nothing here". | S1 |
| U-4 | `segmap osm -o/--out PATH.png` writes a PNG of the road corridor and block partition over the label colourmap, because a partition is a thing you check by looking at it. | S3 |

## Stages

| S | What | Proves |
|---|---|---|
| S1 | `osm/tags.py`, `osm/vectors.py`, `osm/fetch.py` | D-1, F-1, F-2, N-2, U-3 |
| S2 | `osm/burn.py` — projection, line/polygon burn, distance field, `align` | D-5, F-3, F-4 |
| S3 | `osm/partition.py` — blocks + road graph | D-4, F-5, F-6, N-4, U-2, U-4 |
| S4 | cache + `segmap osm` + `blocks` digest level | F-7, F-8, N-3, U-1 |
| S5 | S1 reference checks | D-2, D-3, F-9 |
| S6 | S2 reference evidence | F-10 |
| S7 | S3 OSM features + block unit | F-11 |
| S8 | S4 OSM overlays | F-12 |
| S9 | class notes: parser, template, `segmap notes` | F-13, F-14 |

## Status

S1–S9 are built and exercised on the aza AOI; `tests/test_osm.py` holds the
invariants (blocks tile the ground exactly; S2 is bit-identical without a
reference map; a policy rule the unit cannot measure is reported rather than
silently dropped; a fresh notes template reads as undocumented). The trust
policy that resolves D-2 is `osm/trust.py`; the tile-addressing half of D-1 is
`partition.blocks_in_window` and `segmap osm --tile`.

Order is dependency order; S5–S8 are independent of each other and all depend
on S4.
