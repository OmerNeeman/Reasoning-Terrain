# S4 — Derived decision products

`segmap solve s4 --product trafficability|concealment|drainage|fire_fuel` · [`s4_products.py`](../../src/segmap_digest/solutions/s4_products.py)

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

concealment = max(canopy[class], shadow·0.6, near_built·0.5) + 0.25·relief_roughness
drainage    = 0.45·flatness + 0.30·lowness + 0.25·water_holding_class
fire_fuel   = canopy[class] + 0.3·dry_class + 0.25·grass
```

Each returns a float raster in [0,1] plus per-superclass means and an optional
per-region table.

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `ClassDef.traffic` (47 values) | hand-set in `taxonomy.py` | **Vehicle trials, or a doctrine manual.** Not a sweep |
| `ClassDef.canopy` (47 values) | hand-set | Canopy height / LiDAR, or field survey |
| `VEHICLES.max_slope_deg` | 25/35/45 | Vehicle spec sheets — these are real numbers someone owns |
| `VEHICLES.min_surface` | 0.45/0.20/0.05 | Coupled to the `traffic` scale; meaningless until that's fixed |
| `SLOPE_FREE_DEG` | 5.0 | Where degradation actually starts, per vehicle |
| `WET_CLASSES` | 6 classes | Soil survey; should be a soil-drainage-class join |
| `SHADOW_/BUILT_CONCEALMENT` | 0.6 / 0.5 | Depends entirely on sensor and observer geometry |
| `FIRE_DRY_BONUS` | 0.3 | Fuel-model literature (Rothermel-style), not invention |

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
5. **Concealment from whom?** Overhead EO at nadir, oblique, thermal, and
   ground-level observers give completely different answers. Currently it's
   implicitly nadir EO.
6. **Missing products.** Line-of-sight / intervisibility (needs viewshed on the
   DEM, not just classes), diggability / engineering, dust, off-road speed,
   go-around routing. LOS is the most-requested and the most absent.
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
- **Sensitivity sweep.** Perturb each constant ±30% and see which ones move the
  output. Those are the only ones worth an expert's time.
