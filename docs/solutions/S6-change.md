# S6 — Change reasoning between two dates

`segmap solve s6 [--second t2.tif]` · [`s6_change.py`](../../src/segmap_digest/solutions/s6_change.py)

**What it is.** A pixel diff of two label maps is mostly noise and phenology.
The value is in *classifying* each transition, because the same raw difference
means completely different things:

| Transition | Category | Meaning |
|---|---|---|
| `GreenGrassland → DryGrassland` | phenology | the season changed, the ground didn't |
| `Batha → Garigue` | succession | regrowth, or a boundary wobble |
| `DirtRoad → PavedRoad` | infrastructure | real, and someone cares |
| `Batha → House` | construction | real |
| `LimestoneStoney → DolomiteStoney` | **impossible** | bedrock cannot change lithology — this is a label flip |

That last row is what a naive change detector reports as a landslide.

---

## Naive algorithm (implemented)

```
for each (from, to) transition with area ≥ 20 m²:
    seasonal pair?                        → phenology
    lithology differs?                    → impossible          (label error, not change)
    both in an ordinal series?            → succession / infrastructure
    to ∈ built, from ∉ built?             → construction
    from ∈ built, to ∉ built?             → demolition
    either is an artifact class?          → noise (low confidence)
    class_distance < 0.4?                 → noise (near-neighbour confusion)
    else                                  → real-change
report: area by category, and "X% of the differing area is plausibly REAL"
```

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `SEASONAL_PAIRS` | 2 pairs | Should be derived from acquisition dates + phenology curves, not a hard-coded list |
| `MIN_EVENT_AREA_M2` | 20 | The co-registration error budget — see Q1 |
| `class_distance < 0.4` → noise | 0.4 | The segmenter's own confusion matrix between the two runs |
| `CONSTRUCTION_MIN_COMPACTNESS` | 0.4 | **Declared but unused** — the shape test isn't wired in yet |

---

## Open questions

1. **Co-registration is assumed and it won't be.** `compare()` requires
   identical grids and does a pixel-aligned diff. A one-pixel misalignment
   creates a boundary-following halo of fake transitions on every region edge in
   the tile. **This is the single biggest threat to S6 being useful**, and the
   current `MIN_EVENT_AREA_M2` filter does not address it — misregistration
   noise is numerous and small but *aggregates* to a large area.
   Fix directions: erode region boundaries before diffing; require change
   components to be interior, not boundary-following; register the pair first.
2. **Two dates or two model versions?** If the segmenter was retrained between
   runs, most "change" is model drift. There is currently nothing distinguishing
   them, and the answer differs completely. Log the model version per raster.
3. **What is the phenology model?** Two hard-coded pairs. Real seasonality
   spans irrigated/unirrigated, orchard leaf-on/off, grassland green-up, and soil
   moisture — all functions of acquisition date and rainfall, neither of which
   is an input today.
4. **Is `impossible` always a label error?** Yes for bedrock; but quarrying,
   large earthworks, or a landslide can genuinely expose different material.
   Rare, and the current wording is too absolute.
5. **Transitions are aggregated, not localised.** Output is a table of (from,
   to, total area, component count). Users want *where* — a change polygon layer
   with a category per polygon.
6. **Direction of succession is unverified.** Maquis → Batha in one season is
   almost certainly fire or clearance, not gradual degradation. Rate matters and
   isn't modelled.
7. **More than two dates.** A time series lets phenology be *learned* rather
   than asserted, and turns one-off flips into detectable noise.

## Options and extensions

- **Boundary-aware diffing.** Erode each region by 1–2 px before comparing;
  report only interior change. Cheapest large improvement available, and it
  directly addresses Q1.
- **Change polygons, not a transition table.** Connected-component each
  (from, to) mask, attach the category, emit GeoJSON. This is what a user
  actually consumes.
- **Wire up the shape test.** `CONSTRUCTION_MIN_COMPACTNESS` exists but isn't
  used — a compact rectilinear change component is construction; a diffuse one
  following a slope is natural. Free precision.
- **Confidence per event.** Combine component size, shape, boundary-distance,
  and class distance into a per-event confidence, so the output can be ranked
  rather than filtered.
- **Feed `impossible` back to S1.** Every impossible transition is a *confirmed*
  label error on at least one of the two dates, obtained with no reviewer
  involvement. This is the cheapest source of ground truth in the whole system
  — arguably the most valuable thing S6 produces.
- **LLM for the narrative, code for the table.** The transition table is
  mechanical; "a new access track was cut to the ridge and three structures went
  up beside it" is not. Hand the model the categorised events and let it write
  the report.

## How to evaluate

- **Null test — run it on a tile against itself.** Should report zero change.
  Then run it on the same imagery segmented twice with a different seed or model
  version; everything reported is a false positive. **Do this first; it needs no
  labels and it will be sobering.**
- **Shifted-copy test.** Diff a tile against itself shifted by 1 px. Everything
  reported is misregistration noise, and the total gives you the size of the
  problem in Q1.
- **Known-event recall.** Seed a handful of real changes you know about
  (a new building, a sealed road) and check they surface with the right category.
- **Category precision.** Of the events called `construction`, how many are
  construction? The category, not the detection, is the product.
