# OSM in the reasoning process — the four no-LLM solutions, and how the vectors get in

> PM document. Companion to [`OSM.md`](OSM.md), which is the spec. This one
> answers two questions only: **which solutions reason without an API call**, and
> **by what route OSM data enters each of them, per tile**.
>
> Not in scope here: measuring whether the answers got better. That needs
> reviewed regions and is a separate piece of work (S1's blocking question).
> Everything below is ingestion.

---

## 1. The four solutions that never call an LLM

| | Solution | Reads | Emits | Runs on |
|---|---|---|---|---|
| **S1** | audit | region index + priors | ranked worklist of root causes | `segmap solve s1` |
| **S2** | adjudicate | one region + candidate shortlist | KEEP / SWITCH / UNDECIDABLE + evidence table | `segmap solve s2 --region N` |
| **S3** | triage | chip index + a policy | ranked chips, a spend budget, a control set | `segmap solve s3 --policy P` |
| **S4** | products | label raster (+DEM) | float raster in [0,1] + per-superclass summary | `segmap solve s4 --product P` |

All four are pure code. No key, no network, no model.

Two clarifications, because "the four without an LLM" is not quite the same set as
"S1–S4":

- **S5 (query) also runs deterministically** — `segmap solve s5 --query 'corridor
  PavedRoad'` does the arithmetic in `s5_query.py`. What makes it an LLM solution
  is its *purpose*: the six verbs exist so a model can choose one
  (`tools.py`/`ask.py`). Left out of scope on your instruction, and rightly —
  its OSM work is a different shape (verbs that take a street name).
- **S6 (change) also runs deterministically**, but it needs a second date and we
  have one for nowhere. Out of scope for a different reason.

So: **S1, S2, S3, S4**.

---

## 2. What "inserting OSM into the reasoning process" actually means

Not a new solution. A **second input to stage 0**, on the same footing as the
DEM: something the segmenter never saw, joined to its output, available to every
solution downstream.

```
GeoTIFF(s) ──► loader/mosaic ──► LabelRaster ──┐
                                               ├──► RegionIndex  ─┐
DEM (optional) ────────────────────────────────┤    ChipIndex     ├──► S1 S2 S3 S4
                                               │                  │
OSM snapshot ──► project ─► burn ─► partition ──┴──► BlockIndex   ─┘
                                                    RoadGraph
                                                    WayFootprints
```

The load-bearing rule, and the one thing that must not be traded away:
**OSM never edits `labels`.** Burning roads into the label raster would be the
obvious shortcut and it destroys the only thing that makes this join valuable —
two independent maps you can hold against each other — and imports OSM's own
errors into RT's output with no way to tell them apart afterwards.

### The five ingestion steps

| | Step | Input | Output | Cost |
|---|---|---|---|---|
| **A0** | acquire | tile/mosaic affine → WGS84 bbox, padded | an Overpass-JSON snapshot pinned on disk | one network call per AOI, ever |
| **A1** | project | snapshot + the raster's own inverse affine | way vertices in pixel coordinates | free |
| **A2** | burn | vertices + tag-derived width | corridor / footprint masks, **and one ST class histogram per way** | ∝ corridor area, not raster area |
| **A3** | partition | corridor mask; shared node ids | `BlockIndex` (blocks) + `RoadGraph` (intersections, segments) | one `ndi.label` pass |
| **A4** | gate | corridor vs ST's own road classes | REGISTERED / MARGINAL / MISREGISTERED | one sampled shift search |
| **A5** | index | the above | cached beside the region and chip indices, keyed on the snapshot fingerprint | paid once per AOI |

A0 is the reproducibility unit. OSM changes under you — a partition built on
Tuesday and rebuilt on Friday can differ because somebody mapped an alley, and
nothing in the output would say so. The snapshot is written once with its query,
bbox, server and UTC time; every later run reads the file. Refetching is a
deliberate act (`--osm-refresh`), never a side effect of asking a question twice.

A4 is a **gate, not a metric**. Its job is to refuse: if the two maps are more
than a few metres out of registration, every narrow feature disagrees for
georeferencing reasons and the reference checks would emit thousands of confident
findings about a transform bug. It answers "may we compare these?", not "is the
segmentation good?".

### Per tile — and why the tile is the wrong unit for blocks

You asked for this per tile, e.g. a 1024×1024 window. That works for A0–A2 and
A4 unchanged; **A3 is where tile size bites**, and it is worth deciding
deliberately:

| tile | GSD | ground | typical OSM blocks inside it |
|---|---|---|---|
| 1024² | 0.5 m/px (sinai) | 512 × 512 m | roughly 4–25 |
| 1024² | 0.122 m/px (aza) | 125 × 125 m | **0–2** — smaller than one city block |
| 8509×8405 (aza mosaic) | 0.122 m/px | 1.04 × 1.03 km | 98 |

At aza's resolution a 1024 px tile is a third of a block. A partition built
*per tile* would therefore invent boundaries at the tile seam that exist nowhere
on the ground — the same error the mosaic exists to avoid for connected
components (one wadi across four tiles becoming four wadis with four wrong
areas).

**Recommendation: build blocks on the AOI (mosaic), address them per tile.** A
tile then reports "this window covers block 84 (bounded by شارع النخيل) and part
of block 12", which is what an analyst wants anyway, and no block gets cut by an
arbitrary raster edge. Two mechanics make it work, both already in place:

- the query bbox is **padded** (150 m by default), so blocks close against
  streets that lie just outside the raster;
- ways are addressed by **OSM way id and node id**, so a street crossing a tile
  seam is the same object in both tiles.

If you want a hard per-tile partition anyway (independent tiles, no mosaic step),
that is a one-flag change, and every block touching the tile edge must be
flagged `truncated` so its area is never reported as a block area. Say which and
I will spec it.

---

## 3. Where OSM enters each of the four

Each row is ingestion: a new input to an existing mechanism, not a new mechanism.

### S1 — audit

Today every check is self-referential: it contradicts a label using the label
raster's own geometry, slope and neighbours. It cannot say *"the thing you called
Rendzina is a street"*, because nothing in a label raster knows where streets
are.

OSM enters as **seven new check families**, alongside the existing priors:

| family | the question it asks |
|---|---|
| `osm-road-missing` | OSM maps a street here; ST labels no road class under it |
| `osm-road-occluded` | same, but ST says `Shadow` — a visibility limit, not a label error |
| `osm-road-grade` | both say road; OSM's `surface` tag and ST's road grade disagree |
| `osm-road-extra` | ST maps a road with no OSM way under it |
| `osm-building-missing` | OSM footprint, no ST `House` — keyed by what ST said instead |
| `osm-building-extra` | ST `House` with no OSM footprint |
| `osm-water-extra` | ST `Water` with no OSM water feature |

Three things make these different from a prior, and all three have to survive
into the output:

1. **A disagreement is not an error.** Neither map is truth. Every message names
   both maps and which one asserted what.
2. **Absence is not an assertion.** "OSM has no way here" can mean ST
   hallucinated a road *or* nobody has mapped that alley. The finding says so
   instead of choosing.
3. **Occlusion is its own decision.** A segmenter that cannot see a street in
   shadow has not erred; it was prevented from labelling. Rolling that in with
   real misses would bury both.

Severity is `disagreement × tags.RELIABILITY[layer]`, capped below the physical
contradictions, so nothing sourced from a volunteered map of unmeasured
completeness outranks "badlands on flat ground".

### S2 — adjudicate

Two insertions:

- **the candidate shortlist.** A polygon labelled `Rendzina` sitting on a way
  tagged `surface=asphalt` needs `PavedRoad` on the list, and `class_distance`
  will never put it there — soil-to-road is the 1.0 maximum. OSM proposes
  candidates directly.
- **a fourth evidence term**, `reference`, weighted 0.30 with the existing three
  renormalised to 0.70. Where OSM is silent the term is `None` and the score is
  **bit-identical to today's** — that identity is the acceptance criterion, not
  a hope.

The asymmetry matches the one S2 already applies to morphology: a corridor under
a region is mild evidence *for* a road label; a region wholly inside a mapped
building is strong evidence *against* an undisturbed-ground label. And the
verdict line states when the reference term is what flipped it, because that is
the one input that can be wrong for reasons nothing in the segmentation reveals.

### S3 — triage

Two insertions, and the second is the interesting one:

- **new features on the chip record** — `osm.dist_road`, `osm.road_frac`,
  `osm.building_frac`, `osm.n_intersections` — usable in the same policy-rule
  syntax as `dist_to.PavedRoad` and `entropy`. This is the cheap one: the LLM
  stays out of the per-chip loop, the policy stays a small auditable object, and
  the deterministic scorer gains a real settlement prior instead of inferring
  one from `House` density.
- **the block as the unit of spend**, instead of the 256 px grid square. A grid
  is an artefact of the detector's input size; a block is a semantic unit with a
  name. Ranking 98 named blocks is a thing a human can approve in 30 seconds,
  which is the stated design bar for a policy that discards 85% of an AOI.

One honesty requirement: the block unit does not carry every chip feature
(`dist_to`, class-pair interfaces). A policy referencing a feature the unit does
not provide must produce a `# NOTE:` naming it, not silently evaluate false.

### S4 — products

OSM enters as **named overlay terms**, and the strongest is the first:

| product | overlay |
|---|---|
| trafficability | mapped corridor = known-passable (slope still applies); buildings and walls = impassable |
| concealment | ground *beside* a mapped building gains cover — not on it; a roof is not ground |
| drainage | a mapped wadi/ditch is a channel even where the DEM is too coarse to show it, or absent |
| fire_fuel | a footprint is not fuel |

Why trafficability matters most: on aza, 28.6% of the ground under an OSM road
corridor is labelled `Shadow`, whose `traffic` is the 0.5 default. Without the
overlay a wheeled-vehicle map shows the densest street network in the AOI as
mediocre going. With it, the street is a street.

Every product states what the overlay moved, per layer, before showing any
figure — a product that quietly reasons over a different map than its caller
thinks is a wrong answer with a plausible number attached.

---

## 4. The second ask: labels that carry what the analyst knows

`TerraRosa` is one token. What an analyst means by it is a paragraph, and the
paragraph is what the model needs — the class names are the entire interface
between a segmenter that knows integers and a model that knows geology.

**Mechanism:** a Markdown sidecar, one section per class, named fields. Not
Python, not JSON — a file a domain expert edits without asking anyone.

```markdown
## TerraRosa
aka:            terra rossa; <local term>
what_it_is:     <2-4 sentences: what you mean when you assign this label>
looks_like:     <the surface cues the segmenter is responding to>
confused_with:  ClayeySoil — <how you tell them apart>; Rendzina — <...>
implies:        <operational consequence: going, cover, engineering, agriculture>
scale:          <typical size and shape as a polygon>
season:         <does this label depend on the date>
never:          <what it must never be adjacent to, or never look like>
source:         <who says so, and when>
confidence:     high | medium | low
```

Four of those fields are **machine-read**, and that is what separates this from a
glossary:

| field | consumed by |
|---|---|
| `confused_with` | S2's candidate shortlist — measured confusions beat taxonomy distance |
| `season` | S6's phenology gate |
| `never` | a candidate prior — admitted only if it is a statement about geometry, position or physics (see `RETIRED_PRIORS`: four textbook soil-genesis priors produced 9,793 false findings on sinai before they were removed) |
| `scale` | S2's area and elongation bands, which are currently guesses |

The rest is prose for the model, injected **only for the classes in play** rather
than as a 47-class wall of text — the compact legend stays compact for
high-volume calls.

**What I need from you:** the content, in any form you already have it — a Word
doc, a spreadsheet, a screenshot of an internal wiki, a voice note transcribed.
I convert. The fields above are the ask; `source` and `confidence` are not
bureaucracy — `RETIRED_PRIORS` is the story of what an unsourced claim cost.

A coverage command ranks the 47 classes by their pixel share in your AOI against
which fields are filled, so the effort goes where the map actually is: on sinai,
`Rendzina` alone is 64.9% of classified pixels, and one good paragraph there is
worth twenty on classes that never appear.

---

## 5. Status

| | Built and running | Planned |
|---|---|---|
| A0–A5 ingestion | ✅ aza AOI, snapshot cached | sinai pass |
| Tile addressing (D1) | ✅ `segmap osm --tile` / `--window` | — |
| Trust policy (D2) | ✅ `osm/trust.py`, wired into S1 + S2 | — |
| S1 reference checks | ✅ 7 families, rolled up, axis-tagged | — |
| S2 reference term | ✅ silent-safe, axis-weighted | — |
| S3 OSM chip features | ✅ + a `settlement` policy that uses them | — |
| S3 block unit of spend | ✅ `--unit block` | — |
| S4 overlays | ✅ 4 products | — |
| Class notes | ✅ parser, template, `segmap notes`, S2 wiring | **needs your content** |
| Playground | ✅ `segmap playground` — upload, live OSM, all four | — |
| Summary page | ✅ [`docs/osm-layer.html`](osm-layer.html) | — |
| Demo walkthrough | ✅ [`docs/demo.html`](demo.html) — real imagery from one run | — |

Tests: `tests/test_osm.py`, 26 cases. One PRE-EXISTING failure elsewhere in the
suite (`test_real_raster.py::test_nodata_is_not_trafficable`) — verified against
HEAD before this work started; S5's 500 m² pocket-noise floor swallows a 200 m²
fixture. Not touched here.

## 6. Risks the owner should know about

1. **OSM completeness is unmeasured for these AOIs.** Every `RELIABILITY` number
   is a guess. Converting them into measurements is one day of work: take 30
   blocks, compare OSM against the imagery, count. Until then the defaults are
   deliberately timid.
2. **The aza class-id mapping is suspect, and the OSM join is what exposed it.**
   aza's tiles carry no `ID_TO_LABEL_MAPPING` tag, so they are being read with
   the mapping from the *sinai* drop. Under that mapping, `MaralBadlands`
   ("marl badlands") is the dominant class inside dense urban blocks, and roughly
   half of all mapped buildings have no `House` under them. Either the mapping is
   wrong for aza, or the labels mean something other than their names. This
   wants settling before anything is concluded from aza — and it suggests a
   cheap use of the layer: **OSM as a check on the wire-id mapping itself.**
3. **Block granularity is GSD-dependent** (§2). Decide the unit before building
   anything else on top of it.

## 7. Decisions

### Decided

**D1 — unit of reasoning: AOI-wide blocks, addressed per tile.** *(owner,
2026-08-26)* The partition is built once on the mosaic; a tile reports which
blocks it covers and how much of each. No block is cut by a raster edge, a street
crossing a seam stays one way id, and a block has one area rather than one per
tile that saw it.

What this commits us to:
- the mosaic step is a **prerequisite** of the block partition, not an
  alternative to it — the partition cannot be built from a lone 1024² tile at
  aza's resolution and mean anything;
- a tile-addressing helper is needed and does not exist yet: given a window,
  return the blocks it intersects with each one's share of the window *and* the
  window's share of the block. Both numbers, because "this tile is 4% of block
  84" and "block 84 is 60% of this tile" are different facts and reporting one as
  the other is how a subset answer gets presented as an AOI answer;
- outputs that name a block must say when the window holds only part of it —
  the same discipline `subset_note` already enforces for `--max-mpx`.

### Open

**D2 — trust direction: OSM is the reference for the land; ST is the latest.**
*(owner, 2026-08-26)* — *"OSM is a good way to be a reference for the land, but
the ST output is considered latest unless I say other."*

That is two rankings on two questions, not one ranking, and `osm/trust.py` holds
the split:

| the disagreement is about | authority | example | how it reads |
|---|---|---|---|
| **existence** — is there a street/building here at all | OSM | OSM maps a street, ST shows no road surface | `reference-gap` |
| **identity** — which grade, which surface, what name | OSM | way tagged `surface=asphalt` under a polygon labelled soil | `identity-conflict` |
| **state** — what does that ground look like *now* | ST | OSM has a footprint, ST shows rubble | `state-change` |

Two consequences worth stating, because both **inverted** the first
implementation:

- A mapped building over a polygon ST calls `MaralBadlands` is **not** grounds
  for relabelling it `House`. It is a change candidate. S2 now scores that
  evidence too weakly to clear its switch margin, and S1 raises it as change.
  Verified on aza: a region 78% under a mapped footprint switched to House
  before the policy and keeps its label after it.
- An ST road with no OSM way under it is **not** an ST false positive by
  default. ST is the later map, so the first reading is an unmapped or new
  track — an `update-candidate`, i.e. a free OSM edit.

Reading either the other way round is still possible and sometimes right. That
is why no finding calls either map wrong.

### Decided

**D3 — class notes: build the infrastructure, content follows.** *(owner,
2026-08-26)* Done: `docs/class_notes.md` is a template with all 47 classes
stubbed, `segmap notes` reports coverage ranked by area on your AOI,
`segmap notes --check` validates, and `confused_with` already reaches S2's
candidate shortlist. Fill any subset, in any order; the seeded `what_it_is`
lines are the taxonomy's own text and count as **unfilled** until overwritten.
