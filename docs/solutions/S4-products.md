# S4 — Derived decision products

`segmap solve s4 --product trafficability|concealment|built_fabric|change_volatility` · [`s4_products.py`](../../src/segmap_digest/solutions/s4_products.py)

**What it is.** The many-to-one mapping from 47 classes + slope + season into
what someone actually asks for. This is where the taxonomy stops being a
classification and starts being useful.

**Why a model helps.** Not at inference — at *authoring*. Hand-writing 47 × N
lookup tables with exceptions is tedious and error-prone, and an LLM is good at
composing rules with exceptions. But once authored, the tables are checked in
and evaluation is fully deterministic. **No LLM in the runtime path.**

---

## Naive algorithm (implemented)

```
trafficability(vehicle, wet) = surface[class] · slope_term
                               zeroed where surface < vehicle.min_surface
                               × (1 − wet_sensitivity) on water-holding classes
  slope_term = clip(1 − (slope − 5°)/(vehicle.max_slope − 5°), 0, 1)

concealment(target)  = max(canopy[class]^k, 0.6·cover(shadow), 0.5·cover(near_built))
                       + 0.25·relief_roughness·clip(relief_amplitude / target.height, 0, 1)
  k       = min(1 + target.footprint / canopy_clump, 6)
  cover(m) = mean of m over a window the size of target.footprint

built_fabric      = 0.55·base[class] + 0.45·clip(anthro_fraction_50m / 0.6, 0, 1)
                    (+0.15 inside OSM built landuse)
change_volatility = volatility[class]
```

Each returns a float raster in [0,1] plus per-superclass means and an optional
per-region table.

### `concealment` is parameterised by target — concealment *of what*

`concealment(raster, target="person"|"vehicle"|"structure")`. (Not yet a CLI
flag — `cli.py` builds `**kw` for `trafficability` only, so the command line is
stuck on the default until someone wires it; `compute` already passes it
through.) The same maquis canopy hides a crouching
man and does not hide a truck; the old single constant answered that question
silently on the caller's behalf, which made half the score an unstated
assumption wearing the costume of a terrain property. Now the size is named and
the terms respond to it:

| Target | footprint | height | effect |
|---|---|---|---|
| `person` | 0.5 m² | 1.7 m | canopy exponent ≈ 1.1 — essentially the old product |
| `vehicle` | 8 m² | 2.0 m | fractional canopy has to be near-complete; a 3 m shadow pool no longer counts |
| `structure` | 100 m² | 5.0 m | canopy and adjacency all but vanish; only broad, deep relief remains |

An unknown target raises `ValueError` naming the valid ones. There is no
default-and-hope: the previous behaviour *was* a silent default.

### `built_fabric` — bare-natural → dense-urban

The honest per-tile answer to "how populated is this", derived from the raster
itself rather than from a 100 m population grid that is coarser than a city
block and answers a different question (where people are *registered*, not what
the ground *is*). A per-class ordinal blended with the locally smoothed
anthropogenic area fraction. **Blended, not summed** — a sum cannot fix the case
the product exists for: an isolated shed scores 0.9 from its class and stays
there whatever is added. Measured on a synthetic block at 0.5 m GSD:

| Situation | score | reading |
|---|---|---|
| Inside a dense built block | 0.95 | dense urban |
| Bare courtyard inside that block | 0.45 | in a settlement, on bare ground |
| One isolated shed in open desert | 0.50 | a real building in empty ground |
| Open desert | 0.00 | bare natural |

Both middle rows land mid-scale from opposite directions, and that is the point:
neither pretends to be its own extreme.

### `change_volatility` — the baseline a change detector must beat

**This product exists to serve the change-detection work, and it is
unvalidated.** No pair of real dates has been measured to produce any of its
numbers; every one is read off a class definition. It is a prior over the
taxonomy — how much this ground is *expected* to differ between two dates from
season, phenology and illumination alone — so that a detector can require a
difference to beat its own null hypothesis before calling it an event.

| Classes | value | why |
|---|---|---|
| `GreenGrassland`, `DryGrassland` | 0.9 | *the same ground in two seasons*. The taxonomy says so itself: "a GreenGrassland/DryGrassland difference between two dates is phenology, not change" |
| `Shadow` | 0.85 | an illumination artifact, not a surface. It moves with the sun between two acquisitions for reasons that are not change at all — the least obvious entry here and the one that produces most naive false positives |
| `Car` | 0.8 | gone next week; a change of the object, not of the ground |
| Agriculture (3) | 0.7 | crop cycle: planted → grown → harvested → bare |
| `Water`, `HydromorpicSoil`, `Unclassified`, `Clutter` | 0.5 | seasonal pools; and for the last two, the segmenter is unsure, so it may flip on its own over ground that never moved |
| `Batha`, `Garigue`, `Maquis` | 0.35 | the degradation series shifts over years — what moves between dates is the segmenter's boundary between adjacent stages |
| Soils (4) | 0.2 | a winter green flush relabels bare soil; ploughing changes tone, not kind |
| Dirt roads | 0.15 | re-grading, dust, verge growth |
| `House`, `BrickWall`, `Pavement`, `PavedRoad`, rock (24) | 0.03–0.1 | if these differ between two dates, something actually happened |

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `ClassDef.traffic` (47 values) | hand-set in `taxonomy.py` | **Vehicle trials, or a doctrine manual.** Not a sweep |
| `ClassDef.canopy` (47 values) | hand-set | Canopy height / LiDAR, or field survey |
| `VEHICLES.max_slope_deg` | 25/35/45 | Vehicle spec sheets — these are real numbers someone owns |
| `VEHICLES.min_surface` | 0.45/0.20/0.05 | Coupled to the `traffic` scale; meaningless until that's fixed |
| `SLOPE_FREE_DEG` | 5.0 | Where degradation actually starts, per vehicle |
| `WET_CLASSES` | 6 classes | Soil survey; should be a soil-drainage-class join |
| `SHADOW_/BUILT_CONCEALMENT` | 0.6 / 0.5 | Depends entirely on sensor and observer geometry. Quoted for the `person` target |
| `TARGETS` footprint/height | 0.5/8/100 m², 1.7/2/5 m | **The customer's own target list** — a document that exists and that nobody has shown us |
| `CANOPY_CLUMP_M2` | 4.0 | Crown-diameter statistics per vegetation class — the same LiDAR survey that owns `ClassDef.canopy` |
| `CANOPY_EXPONENT_MAX` | 6.0 | Nowhere. It is a cap on a random-placement model that goes to zero faster than reality, because a structure is *sited* under the densest patch, not dropped at random |
| `FABRIC_BASE` (47 values) | 0.0–0.9 | An ordinal someone has to own, the way `traffic` is. Not calibrated against a census and must not be reported as population |
| **`FABRIC_WINDOW_M`** | **50.0** | **The load-bearing constant of `built_fabric`.** The block-size distribution of settlements in the AOI — measurable from OSM building footprints, never measured here. At ~10 m the density term just replays the class map (the isolated shed is dense urban again); at ~500 m a village smears into the desert and the product becomes the coarse population grid it exists to beat |
| `FABRIC_DENSITY_WEIGHT` / `_SATURATION` / `_ANTHRO_MIN` | 0.45 / 0.6 / 0.45 | Sensitivity sweep, once `FABRIC_WINDOW_M` is settled. Saturation is "what fraction of a dense block is roof and seal" and is answerable from footprints |
| `VOLATILITY` (47 values) | 0.03–0.9 | **A measured pair of dates.** Take two acquisitions over stable ground, cross-tabulate the label pairs per class, and read the off-diagonal mass. Until then the whole table is prose |
| `OSM_BUILT_LANDUSE_FABRIC` | 0.15 | Deliberately small: OSM landuse coverage is the most uneven layer we ingest, and a strong term would draw the edges of OSM's *survey effort* onto the map and call them settlement boundaries |

---

## Open questions

1. **Is a 0–1 score the right output at all?** Users may want *classes*
   (GO / SLOW-GO / NO-GO) with named thresholds, or *speed in km/h*, or a cost
   surface for routing. A float with no units is hard to argue with and hard to
   validate. **Answer this before tuning the constants.**
2. **Whose doctrine?** Trafficability is not a physical constant — it's a
   vehicle × doctrine × acceptable-risk decision. Who owns these numbers?
3. **Season is a switch, not a model.** `--wet` is binary. Real behaviour depends
   on antecedent rainfall, soil type, and how long since it stopped. Minimum
   viable improvement: take an acquisition date and a rainfall series.
4. **Slope from what DEM?** Slope is computed at the label raster's GSD. A 0.3 m
   slope is dominated by micro-relief and noise; vehicle trafficability responds
   to slope over several metres. **The DEM resolution and smoothing window are
   free parameters nobody has set.**
5. **Concealment from whom?** Half-answered. Concealment *of what* is now
   explicit (`--target`), but the observer still is not: overhead EO at nadir,
   oblique, thermal, and ground-level observers give completely different
   answers, and it remains implicitly nadir EO. The target table made the
   omission on the other side of the sensor visible; it did not fix this one.
6. **Missing products.** Line-of-sight / intervisibility (needs viewshed on the
   DEM, not just classes), diggability / engineering, dust, off-road speed,
   go-around routing. LOS is the most-requested and the most absent.
   `drainage` and `fire_fuel` were removed rather than left to rot: neither is
   relevant to this project, and a product nobody consumes is a maintenance
   cost and a source of false confidence in a demo.
7. **How do products compose?** "Concealed *and* trafficable" is the actual
   question most of the time. Min, product, or a weighted rank?
8. **Validation has no ground truth path.** Unlike S1/S3, there's no cheap
   labelling story. Expert review of a handful of AOIs may be all that's
   available — design for that.

## Options and extensions

- **Cost surface + routing.** Convert trafficability to a per-pixel traversal
  cost and run least-cost path / A*. Turns "where can I go" into "here is the
  route", which is what people want. Cheap to add on top of what exists.
- **Viewshed for LOS.** Standard DEM viewshed, modulated by canopy height from
  the class. Biggest missing product.
- **Uncertainty propagation.** A product built on a suspect region (see S1)
  should carry that doubt. Emit a confidence raster alongside every product.
- **Per-vehicle table externalised.** Move `VEHICLES` and the per-class `traffic`
  values into a YAML file that a domain owner edits without touching code — same
  argument as the class definitions.
- **LLM-authored tables with a diff review.** Have the model propose the 47
  values for a new product (say, diggability) with a one-line justification each,
  then diff against expert edits. Fast way to bootstrap a new product; terrible
  way to ship one unreviewed.
- **Wet/dry as two committed products** rather than a flag, so downstream
  consumers can't forget to set it.

## How to evaluate

Hardest of the six to evaluate, and the docs should say so rather than imply a
metric exists:

- **Expert key.** Have an analyst classify GO/SLOW-GO/NO-GO on 50–100 polygons;
  score agreement. Small-N but it's the real target.
- **Known-route check.** Existing tracks and roads should score as trafficable
  end-to-end. Any road segment scoring NO-GO is a bug — this one is free and
  should be a regression test.
- **Ablation.** Product with vs. without slope, with vs. without wet. If a term
  changes nothing, delete it.
- **`change_volatility` has the only cheap validation story here, and it should
  be built first.** Take two dates over ground known not to have changed, and
  the per-class off-diagonal rate of the label confusion is a *measurement* of
  this table — no analyst, no polygons. Every number in `VOLATILITY` is
  currently prose, and this is how it stops being prose.
- **`built_fabric` against footprint density.** OSM building footprint area per
  hectare is an independent (if incomplete) measure of the same thing. Rank
  correlation over tiles is free and would catch a badly-set
  `FABRIC_WINDOW_M`.
- **Sensitivity sweep.** Perturb each constant ±30% and see which ones move the
  output. Those are the only ones worth an expert's time.
