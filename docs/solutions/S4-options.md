# S4 — The option space for derived products

Companion to [S4-products.md](S4-products.md), which documents the four products
that exist. This document is the **menu**: every decision product this taxonomy
could plausibly support, what each one costs, and which ones are honestly not
possible. Nothing here is implemented. It exists so that someone can choose.

**What we have today.** A 45-class label raster at ~0.3 m/px, the region index
(area, perimeter, compactness, elongation, neighbour graph with shared boundary
lengths), the chip index (per-chip histograms, entropy, class-pair interface
lengths, distance-to-anchor fields), and the taxonomy's structure — the
lithology × morphology grid, the `DEGRADATION_SERIES`, the `ROAD_SERIES`, the
co-occurrence `PRIORS`.

**What we do not have.** No RGB. No DEM — so every slope, elevation, aspect and
`aspect_circvar` field in `Region` is currently zero, and `_slope()` in
`s4_products.py` returns a zero array. Read that carefully: **three of the four
shipped products are running with their topographic term silently disabled.**
`trafficability` is currently a per-class lookup, `drainage` is currently 0.45 ×
1.0 + 0.30 × 0.5 + 0.25 × holding, and `concealment` has no roughness term. No
acquisition date, no rainfall series, no soil-drainage map, no geological map,
no canopy height, no second date.

Feasibility is stated against *that* baseline, not against the fixture — the
synthetic tile in `synth.py` has a DEM and the real tiles do not.

---

## Feasibility tiers

| Tier | Meaning |
|---|---|
| **NOW** | Buildable from the label raster and the indices alone, today |
| **NOW\*** | Buildable now but running degraded; a DEM materially upgrades it |
| **DEM** | Not meaningful without elevation. Do not ship a version without it |
| **DEM+** | Needs a DEM *and* a derived product on it (viewshed, flow accumulation) |
| **GAP** | Needs an input nobody in this project has: rainfall, date, canopy height, geology, a second acquisition |

## Validation motifs

There is no ground truth and there is not going to be one. These eight moves are
what is actually available; each product below cites the ones that bite. Defined
once here so the catalogue can stay short.

- **V1 — Known-route check.** `PavedRoad`, `DirtRoad` and `DirtRoadB` must score
  passable end-to-end. A road pixel scoring NO-GO is a bug. Free, deterministic,
  belongs in CI.
- **V2 — Revealed preference.** The map is a record of decisions humans already
  made. Someone judged the ground under `House` buildable; someone judged the
  ground under `UnirrigatedOrchard`, `LimestoneTerrace` and `NariTerrace`
  workable; `DirtRoad` alignments are a going map drawn by drivers over years.
  A product that scores those locations badly is either wrong or has found
  something — and you must decide which *before* you look.
- **V3 — Expert key.** 50–100 polygons scored by an analyst. Measure
  **inter-expert** agreement first: it is the ceiling. If two experts agree at
  κ = 0.5, no amount of tuning validates the product past 0.5.
- **V4 — Ablation.** Delete a term. If the output does not move, delete the term.
- **V5 — Sensitivity sweep.** Perturb each constant ±30%. Only the constants
  that move the output are worth an expert's time.
- **V6 — Physical-consistency test.** The product must obey a law it was not
  fitted to: drainage must be monotone downhill, a viewshed must be symmetric,
  fuel must be monotone along `DryGrassland → Batha → Garigue → Maquis`.
- **V7 — Stability.** One-pixel shift, tile-boundary seams, cross-tile transfer.
  If the answer moves, the product is reading segmenter noise.
- **V8 — Class-swap probe.** Synthesise a tile with one class substituted for a
  near neighbour under `class_distance()` — `Garigue`→`Batha`, or
  `LimestoneStoneyTerrain`→`DolomiteStoneyTerrain` (distance 0.45, the
  RGB-blind case). A good product moves a little. One that flips is keyed to a
  distinction the segmenter cannot reliably make, and will be unstable in
  production for reasons that look like nothing.

---

# A · Mobility & movement

### A1 — Off-road trafficability, vehicle-conditioned
*"Can my vehicle cross this ground, and where will it bog or belly out?"*

The shipped product. The taxonomy carries most of the signal directly: the
morphology axis of the rock grid **is** a surface-roughness ladder. `Boulder`
(`LimestoneBoulder`, `DolomiteBoulder`, `BasaltBoulder`) is impassable to
wheels; `RockyTerrain` is broken bedrock; `BeddedRock` is stepped micro-relief;
`StoneyTerrain` is clasts over soil and is the most trafficable rock state;
`Terrace` is a near-flat facet and is close to soil. Soils split by wet
behaviour — `ClayeyDeepSoil` and `HydromorpicSoil` are the bog risks,
`TerraRosa` and `Rendzina` are firm. Vegetation resists by density along
`Batha → Garigue → Maquis`. `MaralBadlands` is a dissection class, not a surface
class, and is impassable for a reason slope alone does not express.

- **Signal:** all 21 rock classes via `morphology`; the five `soil` classes; the
  `DEGRADATION_SERIES`; `Water`, `BrickWall`, `House` as absolute barriers.
- **Also needs:** DEM for the slope term. Vehicle spec sheets for `max_slope_deg`
  and `min_surface`. Doctrine for what risk is acceptable.
- **Feasibility:** NOW\* — surface term works today; slope term is dead code
  until a DEM exists.
- **Output:** continuous raster [0,1], per vehicle class.
- **Validate:** V1 (the free one), V5, V8. Note `LimestoneStoneyTerrain` vs
  `DolomiteStoneyTerrain` carry identical `traffic` — good, that confusion is
  operationally free, and V8 should confirm it.

### A2 — GO / SLOW-GO / NO-GO overlay
*"Show me where I can go."*

The same computation as A1 with the output question answered the other way.
S4-products.md flags this as the blocking open question and it is: a unitless
float is unfalsifiable and un-arguable, while three named classes with published
thresholds can be disputed by an expert — which is the only validation channel
that exists. Thresholds are policy, not physics, and should live in the same
externalised table as the vehicle definitions.

- **Signal:** as A1.
- **Also needs:** someone with authority to set two thresholds.
- **Feasibility:** NOW.
- **Output:** three categorical classes, as raster and as dissolved vector
  polygons.
- **Validate:** V1, V3 (this is the product V3 was made for — an analyst can
  score GO/SLOW-GO/NO-GO on 80 polygons in an afternoon; nobody can score a
  float).

### A3 — Least-cost route
*"Get me from here to there. Show me the line."*

Invert A1 into a per-pixel traversal cost and run A\*. This is the product users
actually ask for — "where can I go" answered as a line on a map rather than a
heat map. Cheap on top of what exists, and `s5_query.py`'s `corridor` verb is
already half of it. Absolute barriers (`House`, `BrickWall`, `Water`, the three
`Boulder` classes) become infinite cost; the `ROAD_SERIES` becomes a cost
discount ordered `PavedRoad < DirtRoad < DirtRoadB`.

- **Signal:** A1's cost surface plus `ROAD_SERIES` discounts.
- **Also needs:** DEM if you want the route to respect climb cost rather than
  just surface; without it, routes will happily cross a cliff face of
  `LimestoneStoneyTerrain`.
- **Feasibility:** NOW\* — and the caveat is severe. A label-only route is
  *plausible-looking and unsafe*. Ship it with the limitation on the face of the
  product.
- **Output:** a route (polyline), with a cost breakdown per segment.
- **Validate:** V2 is unusually strong here — routed paths between two points
  should converge on the existing `DirtRoad` network, because that network is
  the accumulated route-finding of everyone who drove there before you.

### A4 — Chokepoint & defile inventory
*"Where does my column get squeezed, and where would I be ambushed?"*

Distance-transform the trafficable mask; local maxima of a narrow corridor are
chokepoints. From labels alone you get **obstacle-bounded** chokepoints — gaps
between `Maquis` blocks, gaps in a `BrickWall` line, a `DirtRoad` threading
between two `LimestoneBoulder` fields, a wadi crossing. You do **not** get
topographic defiles, which are the ones that matter most, because a defile is
defined by walls that a label raster cannot see.

- **Signal:** the negative space of A1's NO-GO mask; `LimestoneBoulder`,
  `DolomiteBoulder`, `BasaltBoulder`, `Maquis`, `Water`, `BrickWall`,
  `MaralBadlands` as the confining classes.
- **Also needs:** DEM for true defiles; DEM+viewshed to rank them by
  overlook.
- **Feasibility:** NOW\* for obstacle chokepoints, DEM for the rest.
- **Output:** ranked table of points with corridor width in metres, plus vector
  polygons of the constriction.
- **Validate:** V2 — a real chokepoint usually has a road through it, so the
  detected set should intersect the road network more than chance; V7, since
  narrow-corridor detection is exactly what a 1 px shift destabilises.

### A5 — Obstacle inventory
*"List every thing that will stop me, with its size and where it is."*

The least glamorous product here and possibly the most immediately useful,
because it is a *list*, it is checkable line by line, and it does not require
any threshold to be agreed. The region index already computes everything needed:
area, elongation, compactness, centroid, neighbours.

- **Signal:** `BrickWall` (linear, high elongation), `House`, `Water`, the three
  `Boulder` classes, `MaralBadlands`, `Maquis` above a size threshold,
  `LimestoneRockDipSlope` and `NariRockDipSlope` (planar, smooth, you slide).
- **Also needs:** nothing.
- **Feasibility:** NOW.
- **Output:** vector polygons with attributes; a ranked table.
- **Validate:** V3 on a sample — this is the one product where an analyst can say
  "yes/no, that is an obstacle" without any doctrine argument; V7.

### A6 — Wet-season mobility delta
*"How much worse does this get after rain, and for how long?"*

`--wet` is currently a boolean applied to six hand-picked classes. Real
behaviour depends on antecedent rainfall, soil texture, and days since the last
event. `ClayeyDeepSoil` on a valley floor after 40 mm is a different object from
the same polygon in August. The interesting output is not the wet map — it is
the **delta**, i.e. which ground changes category, because that is what a
planner needs to know and what a dry-season recce will not tell them.

- **Signal:** `HydromorpicSoil`, `ClayeyDeepSoil`, `Clayeysoil`,
  `MaralSmoothRockSlopes` (marl goes to grease when wet — currently *not* in
  `WET_CLASSES`, which looks like an omission), `IrrigatedField`,
  `IrrigatedOrchard`, `Water`.
- **Also needs:** acquisition date + a rainfall series. A soil-drainage-class
  join would replace the hand-picked list with something defensible.
- **Feasibility:** GAP — neither input is in the pipeline. The date is nearly
  free to obtain; the rainfall series is a public dataset.
- **Output:** categorical delta raster (unchanged / degrades one class /
  degrades two).
- **Validate:** V4 first — check the wet flag changes anything at all on a real
  tile; V3.

---

# B · Observation & concealment

### B1 — Overhead concealment
*"If I park here, can it be seen from above?"*

The shipped product. The `DEGRADATION_SERIES` is doing real work: the class
definitions carry explicit structural thresholds — `Batha` <25% woody cover and
under 0.5 m, `Garigue` 25–50% at 0.5–1.5 m, `Maquis` >50% above 1.5 m. That is a
concealment ladder written into the taxonomy, and it is the reason a
`Garigue`/`Maquis` confusion matters here far more than it does for A1.
`IrrigatedOrchard` (canopy 0.6) conceals better than `UnirrigatedOrchard` (0.4)
and stays that way through summer.

- **Signal:** `DEGRADATION_SERIES`; `UnirrigatedOrchard`, `IrrigatedOrchard`;
  `Shadow`; proximity to `House` and `BrickWall`.
- **Also needs:** nothing for the nadir-EO case. A sensor and observer geometry
  to mean anything else. Canopy height to be quantitative rather than ordinal.
- **Feasibility:** NOW.
- **Output:** continuous raster; better as three classes (exposed / broken /
  concealed).
- **Validate:** V6 — must be monotone along the series; V8, since a
  `Garigue`/`Batha` swap *should* move this product noticeably, and if it does
  not the canopy values are not doing anything.
- **Caveat:** `Shadow` is scored at 0.6 concealment, but shadow is an artifact of
  one sun angle at one instant. Concealment inherited from `Shadow` is valid for
  the acquisition time and no other. This should be a separate band, not blended
  into the score.

### B2 — Line of sight / intervisibility
*"Can I see that from here — and can they see me?"*

The most-requested and most-absent product, correctly identified in
S4-products.md. It is a DEM viewshed with a canopy-height obstruction layer
draped on top; the label raster supplies the second half only, and only
ordinally. There is no version of this that runs on labels. Do not approximate
it — a wrong LOS answer is worse than no answer, because people act on it.

- **Signal:** `Maquis` (>1.5 m), `IrrigatedOrchard`, `UnirrigatedOrchard`,
  `House`, `BrickWall` as obstructions; `Garigue` as partial.
- **Also needs:** DEM (mandatory), viewshed implementation, canopy height for
  anything better than three ordinal height bins.
- **Feasibility:** DEM+.
- **Output:** binary or fractional visibility raster from a given point; a
  point-to-point boolean.
- **Validate:** V6 — reciprocity is a free, strong test: viewshed(A)∋B must imply
  viewshed(B)∋A for equal observer heights, and any implementation bug breaks it.

### B3 — Observation post siting
*"Where do I put an OP to watch this road?"*

Cumulative viewshed over a target set (say, all `PavedRoad` pixels), intersected
with B1 concealment and A1 access. A good OP sees much, is itself concealed, and
can be reached — three products composed, which makes it a good test of open
question 7 in S4-products.md.

- **Signal:** B2 × B1 × A1; `LimestoneTerrace`, `NariTerrace`, `MaralTerrace`,
  `DolomiteTerrace` as the flat, occupiable facets on otherwise steep ground —
  a terrace is where you can actually sit.
- **Also needs:** DEM + viewshed.
- **Feasibility:** DEM+.
- **Output:** ranked table of candidate points with viewshed area, concealment
  score, and access cost.
- **Validate:** V3; V2 in the strong form — if there are existing structures on
  high ground, check whether the product finds them.

### B4 — Cover from direct fire
*"If I am shot at here, what stops the round?"*

Distinct from concealment and routinely conflated with it. `Maquis` conceals and
stops nothing. `BrickWall`, `House`, and the `Boulder` classes are hard cover.
Everything else is defilade, which is a purely topographic property.

- **Signal:** `BrickWall`, `House`, `LimestoneBoulder`, `DolomiteBoulder`,
  `BasaltBoulder`, `LimestoneRockyTerrain`, `DolomiteRockyTerrain` for hard
  cover; `LimestoneBeddedRock` for stepped micro-relief.
- **Also needs:** DEM for defilade — which is most of the answer.
- **Feasibility:** NOW for the hard-cover inventory (a thin product), DEM for
  anything complete.
- **Output:** categorical (hard cover / defilade / concealment only / exposed).
- **Validate:** V3. Note the useful negative result: shipping B1 without B4
  actively misleads, and the two should be released together or B1 renamed.

### B5 — Concealed approach
*"Get me there without being seen."*

A3's routing with a visibility penalty. The composition question is real: min()
gives you the pessimist's route, product() lets high trafficability buy off
exposure. Neither is obviously right and the choice should be exposed to the
user rather than baked in.

- **Signal:** A3 cost surface × B1 (label-only version) or × B2 (real version).
- **Also needs:** DEM+viewshed for the version worth having.
- **Feasibility:** NOW\* as concealment-weighted routing; DEM+ as true
  defilade routing.
- **Output:** a route, with an exposure profile along it.
- **Validate:** V4 on the composition operator — if min() and product() give the
  same route, the question does not matter on this terrain and you can stop
  arguing about it.

### B6 — Shadow-derived object height
*"How tall is that thing?"*

`Shadow` is a first-class label and its definition says so explicitly: "shadow
length and direction are recoverable evidence about object height and sun
angle." Measure the shadow polygon's long axis, take sun elevation from the
acquisition time, get height. Applied to `Shadow` adjacent to `House` this gives
building heights; adjacent to `Clutter` it distinguishes a pylon from a tarpaulin.

- **Signal:** `Shadow` regions and their `elongation`/`bbox` from the region
  index; the adjacency graph to identify the casting class.
- **Also needs:** acquisition timestamp and sun azimuth/elevation. Nothing else.
  This is the cheapest missing input in the whole document.
- **Feasibility:** GAP, but a trivially closable one — the sun angle is two
  numbers derivable from the image metadata that `LabelRaster` does not
  currently carry.
- **Output:** a height attribute joined to adjacent `House` / `Clutter` regions;
  a number per object.
- **Validate:** V6 — all shadows in one tile must point the same way, so azimuth
  consistency across the whole `Shadow` class is a free self-check; V2 against
  any known building height.

---

# C · Hydrology & surface conditions

### C1 — Ponding and poor drainage
*"Where will water sit after rain?"*

The shipped `drainage` product is, without a DEM, 0.45 × flat(0) + 0.30 × 0.5 +
0.25 × holding — i.e. a constant plus a class mask. That is not a drainage
product, it is a soil lookup with two decorative terms. Real ponding is
depression-filling on a DEM; the labels contribute the water-holding half and,
usefully, an independent check: `HydromorpicSoil` is *defined* as seasonally
waterlogged ground in closed depressions, so it is both an input and a truth
proxy.

- **Signal:** `HydromorpicSoil`, `ClayeyDeepSoil`, `Clayeysoil`, `Water`;
  `MaralTerrace` and the other `Terrace` classes as flat facets that hold water.
- **Also needs:** DEM. Mandatory.
- **Feasibility:** DEM.
- **Output:** continuous raster, or vector polygons of predicted ponds.
- **Validate:** V6 (monotone downhill); and the strongest test available —
  predicted ponding should coincide with `HydromorpicSoil` polygons the model
  never saw, which turns a taxonomy class into free validation data.

### C2 — Ephemeral channel / wadi network
*"Where does the water run, and where do I cross?"*

Proper extraction is flow accumulation on a DEM. But the labels carry a real
partial signal that is worth extracting *now*: `MaralBadlands` is defined by high
drainage density, and linear high-elongation runs of `ClayeyDeepSoil` or
`HydromorpicSoil` between rock units trace valley floors. In arid Sinai terrain
the wadi network is also the de facto road network, so this doubles as a
mobility product.

- **Signal:** `MaralBadlands`, `ClayeyDeepSoil`, `HydromorpicSoil`, `Water`;
  region `elongation` and the neighbour graph.
- **Also needs:** DEM for anything defensible.
- **Feasibility:** NOW\* as a crude proxy, DEM for the real thing.
- **Output:** vector polylines; a crossing-point table.
- **Validate:** V2 — extracted channels should be crossed, not paralleled, by the
  `PavedRoad` network and paralleled by `DirtRoad`; V7.

### C3 — Dust generation potential
*"Will my vehicles raise a dust signature here?"*

Underrated, buildable today, and it needs no DEM at all, because dust is a
surface-material property. The taxonomy separates fine-grained and crusted
surfaces cleanly. `DirtRoad` and `DirtRoadB` are the obvious dust producers;
`ChalkSmoothRockSlopes` and `MaralSmoothRockSlopes` are fine-grained soft rock;
`Rendzina` is a shallow pale calcareous soil that powders. `NariStoneyTerrain`
is calcrete crust and produces very little. `TerraRosa` is clay and produces
little when dry-crusted but a lot when disturbed.

- **Signal:** `DirtRoad`, `DirtRoadB`, `Rendzina`, `ChalkSmoothRockSlopes`,
  `MaralSmoothRockSlopes`, `Clayeysoil`, `TerraRosa`; the `Boulder`,
  `RockyTerrain` and `BeddedRock` classes as near-zero.
- **Also needs:** nothing for a relative index. Wind and soil moisture for
  anything absolute.
- **Feasibility:** NOW.
- **Output:** three classes (low / moderate / high) along the route or as a
  raster.
- **Validate:** V3, and V2 in a weak form — heavily used `DirtRoad` corridors
  should score high by construction, which is a sanity check rather than
  evidence.

### C4 — Water point inventory
*"Where is there water, and who is using it?"*

A list, not a raster. `Water` polygons with area, plus the inference chain the
taxonomy supports: `IrrigatedField` and `IrrigatedOrchard` in a Mediterranean
summer imply a water source that may not itself be visible, which makes them
*evidence of* a water point rather than a water point. The class definition for
`IrrigatedField` states this outright ("anything rectangular and green in late
summer is almost certainly irrigated").

- **Signal:** `Water`; `IrrigatedField`, `IrrigatedOrchard`, `GreenGrassland`
  (green in a dry-season acquisition = irrigation or a spring);
  `HydromorpicSoil`.
- **Also needs:** acquisition date, to know whether "green" is informative at
  all. In the wet season `GreenGrassland` means nothing.
- **Feasibility:** NOW, with a large asterisk on the season.
- **Output:** ranked table with coordinates, area, and evidence type.
- **Validate:** V3; V7 across dates once a second acquisition exists.

### C5 — Erosion susceptibility
*"Which slopes are falling apart, and will this track survive the winter?"*

`MaralBadlands` is not a susceptibility prediction — it is erosion that has
already happened, i.e. observed ground truth for the process. That makes this
one of the few products with an internal training signal: fit susceptibility on
everything else, check whether it independently predicts where the badlands are.

- **Signal:** `MaralBadlands` (realised), `MaralSmoothRockSlopes`,
  `ChalkSmoothRockSlopes`, `Rendzina` (shallow soil on soft carbonate),
  `Clayeysoil`; low `canopy` classes as unprotected.
- **Also needs:** DEM for slope and slope length; rainfall erosivity for
  anything quantitative.
- **Feasibility:** DEM.
- **Output:** continuous raster or three classes.
- **Validate:** the held-out-`MaralBadlands` test above is the strongest
  validation in this entire document — real, cheap, and not circular provided
  the badlands class is excluded from the inputs. Plus V6.

---

# D · Hazard

### D1 — Fire fuel load
*"How much is there to burn?"*

The shipped product. The `DEGRADATION_SERIES` is a fuel-model ladder as well as a
concealment ladder: `DryGrassland` is fast-spreading fine fuel with low total
load, `Maquis` is high-load slow-ignition sclerophyll, `Batha` and `Garigue` sit
between. This is the standard Mediterranean fuel problem and the classes map
almost one-to-one onto published fuel models — so unusually for this document,
the constants have a literature to come from rather than needing invention.

- **Signal:** `DEGRADATION_SERIES`; `UnirrigatedOrchard` (dry inter-row) vs
  `IrrigatedOrchard` (green, poor fuel); `GreenGrassland` vs `DryGrassland` as a
  seasonal switch that flips this product entirely.
- **Also needs:** acquisition date — `GreenGrassland` and `DryGrassland` are the
  same vegetation and the fuel answer differs by a factor of several.
- **Feasibility:** NOW\* — buildable, but the season flag matters more here than
  in any other product, and it is currently unavailable.
- **Output:** categorical fuel model per pixel, mapping to a named standard.
- **Validate:** V6 (monotone along the series); V5.

### D2 — Fire spread / rate of spread
*"Where does it go, how fast, and what is in the way?"*

Fuel is the easy half. Spread needs slope (fire runs uphill) and wind, neither
available. Firebreaks come free from the labels: `PavedRoad`, `Pavement`,
`Water`, and the bare rock classes are non-fuel.

- **Signal:** D1 fuel raster; `PavedRoad`, `Pavement`, `Water`,
  `LimestoneRockyTerrain`, `DolomiteRockyTerrain`, `BasaltRockyTerrain`,
  `MaralBadlands` as breaks; `House` as the thing at risk.
- **Also needs:** DEM + wind field + a spread model.
- **Feasibility:** GAP (needs DEM *and* wind).
- **Output:** time-to-arrival raster from an ignition point.
- **Validate:** effectively unvalidatable here. Ship the fuel map and the
  firebreak map; do not ship a spread model you cannot test.

### D3 — Slope instability, rockfall source and runout
*"What is going to come down, and how far?"*

Source areas are recognisable from labels — `Boulder` fields *are* rockfall
deposits, so they mark where it has already happened and where it will again.
`LimestoneRockDipSlope` and `NariRockDipSlope` are planar bedding-parallel
facets: the classic slab-failure geometry, and the definitions already require
low `aspect_circvar`, meaning the taxonomy encodes the structural condition.
Runout distance is pure topography.

- **Signal:** `LimestoneBoulder`, `DolomiteBoulder`, `BasaltBoulder` (deposits);
  `LimestoneRockDipSlope`, `NariRockDipSlope` (failure planes);
  `LimestoneBeddedRock`; `MaralBadlands`.
- **Also needs:** DEM. Dip direction and angle from a geological map to know
  whether a dip slope is a failure surface or a stable bench.
- **Feasibility:** DEM (partial), full version needs geology too.
- **Output:** vector source polygons plus modelled runout zones.
- **Validate:** predicted runout should contain the observed `Boulder` polygons —
  again a class doubling as ground truth. V3.

### D4 — Flash-flood exposure
*"What gets washed away when it rains upstream?"*

The characteristic hazard in this terrain: wadis that are roads for 360 days and
torrents for five. Needs a catchment, which needs a DEM extending well beyond the
tile, plus rainfall. The label raster contributes only the exposure inventory.

- **Signal:** C2 channel network; `House`, `PavedRoad`, `DirtRoad`,
  `IrrigatedField`, `ClayeyDeepSoil` as exposed assets.
- **Also needs:** DEM covering the *upstream catchment*, not just the tile, plus
  a rainfall series. Tile-local hydrology is a category error.
- **Feasibility:** GAP.
- **Output:** vector inundation extent; a ranked exposed-asset table.
- **Validate:** V3 only, or historical event records if any exist.

### D5 — Surface-disturbance anomaly
*"Has something here been disturbed since last time?"*

Belongs to S6, listed here for completeness because it is what people ask for.
`Clutter` and `Unclassified` are the taxonomy's explicit "something is here and I
have no word for it" classes and are the natural anchors. Single-date, this
cannot work: there is no such thing as a disturbed-surface signature in a
45-class label map. Two dates plus S6's phenology filter makes it a real
question — and even then the correct output is a *review queue*, not a call.

- **Signal:** `Clutter`, `Unclassified`, `Shadow`; class transitions S6 flags as
  `impossible`.
- **Also needs:** a co-registered second acquisition. Probably RGB or a detector
  for confirmation.
- **Feasibility:** GAP.
- **Output:** ranked candidate table for human review. Never a decision.
- **Validate:** S3's control-set pattern is the only honest option — dispatch a
  fraction of rejected candidates anyway to get an unbiased recall signal.

---

# E · Engineering & construction

### E1 — Diggability / excavatability
*"Can I dig here with a shovel, with a JCB, or do I need explosives?"*

**The product this taxonomy appears to have been built for.** No other candidate
uses the class list so completely. The lithology × morphology grid resolves
almost exactly to excavation classes: `StoneyTerrain` means clasts over soil —
diggable with effort. `RockyTerrain` means extensive exposed bedrock — machine
work. `Boulder` means detached blocks — you move them, you do not dig them.
`BeddedRock` means bedding planes, which rip along the bedding and resist across
it. `Terrace` means soil-covered flat — the easiest excavation on the map.
`SmoothRockSlopes` on marl and chalk is soft rock, diggable by hand.

Nari is the case that gives the argument away. Nari is a **calcrete crust**, not
a formation, and no geological map carries it as a separate unit — it is a
near-surface hardpan a metre or two thick over softer material. The only reasons
to spend three of your 45 classes on it are that you must break through it, or
that you must build on it. Both are engineering.

- **Signal:** the whole rock grid via `morphology`; `TerraRosa`, `Rendzina`
  (shallow — you hit rock fast), `ClayeyDeepSoil` (deep, easy, but wet-unstable),
  `Clayeysoil`; the three `Nari*` classes as a distinct hardpan case.
- **Also needs:** nothing for an ordinal class. Depth-to-bedrock — which no map
  has — for anything quantitative.
- **Feasibility:** NOW.
- **Output:** four categorical classes (hand / light machine / heavy machine /
  blast), as raster and vector.
- **Validate:** V3, and V2 strongly — existing quarry workings (usually labelled
  `Clutter` per its definition, "quarry infrastructure") and existing cuttings
  along `PavedRoad` are places where someone already paid to find out.

### E2 — Bearing capacity / hardstanding suitability
*"Can I put a heavy thing here and will it still be level next year?"*

The complement of E1: E1 asks what you can remove, E2 asks what will hold.
`ClayeyDeepSoil` and `Clayeysoil` shrink and swell — the definitions say so —
and are the worst foundations on the map. The `Nari*` classes are the best
non-engineered surface available: a hard crust is precisely what you want
underneath. `HydromorpicSoil` is disqualifying.

- **Signal:** `NariRockyTerrain`, `NariStoneyTerrain`, `NariTerrace`;
  `LimestoneBeddedRock`, `DolomiteRockyTerrain`; `Clayeysoil`, `ClayeyDeepSoil`,
  `HydromorpicSoil` as negatives; the `Terrace` classes as ready-made level
  ground.
- **Also needs:** nothing for a screening product. Geotechnical data to put
  numbers on it — and a screening product is genuinely useful, because it says
  where *not* to send the drill rig.
- **Feasibility:** NOW.
- **Output:** three classes, vector polygons.
- **Validate:** V2 is decisive — `House` and `Pavement` polygons record where
  people already chose to build. If the product scores existing settlement as
  poor ground, it is wrong.

### E3 — Aggregate and borrow-material siting
*"Where do I get fill, and where is the nearest hard stone?"*

`LimestoneStoneyTerrain` and `DolomiteStoneyTerrain` are loose clast fields;
`Boulder` fields are armour stone; `ClayeyDeepSoil` is a fill and liner source.
Composed with A3 (haul distance along the road network) this becomes an actual
siting product rather than a material map.

- **Signal:** `LimestoneStoneyTerrain`, `DolomiteStoneyTerrain`,
  `BasaltStoneyTerrain`, the three `Boulder` classes, `ClayeyDeepSoil`;
  `Clutter` as an existing-quarry indicator; haul distance via `ROAD_SERIES`.
- **Also needs:** a geological map to confirm limestone vs dolomite, which
  matters for aggregate quality and — per the class definition for
  `DolomiteRockyTerrain` — is not separable in RGB at all.
- **Feasibility:** NOW for siting, geology for material quality.
- **Output:** ranked table of candidate sources with volume estimate and haul
  distance.
- **Validate:** V2 — existing quarries should rank highly; if they do not, the
  criteria are wrong, because someone with money already ran this analysis.

### E4 — Route and track construction cost
*"What would it cost to put a road through here?"*

E1 diggability plus earthwork volume plus A3 alignment. The three inputs are
independent and it is a clean composition, but earthwork volume is cut-and-fill
against a DEM and there is no substitute.

- **Signal:** E1 output; A3 alignment; `MaralBadlands` and `HydromorpicSoil` as
  ground you route around at any cost.
- **Also needs:** DEM.
- **Feasibility:** DEM.
- **Output:** a number (cost per km) along a route; a ranked comparison of
  alignments.
- **Validate:** V2 against the existing `DirtRoad` network — the alignments that
  exist are, roughly, the cheap ones.

### E5 — Helicopter landing site / drop zone siting
*"Where can this land?"*

Requirements are a size, a slope limit, a surface that will not brown out or
throw debris, and no vertical obstructions. Three of the four are available
today: size from region area, surface and debris from the class, obstructions
from `House`, `BrickWall`, `Maquis`, orchard canopy. Slope is the missing one and
it is the hard constraint, which is what puts this in the DEM tier.

- **Signal:** the four `Terrace` classes and `DryGrassland`, `Batha`,
  `NariStoneyTerrain` as candidate surfaces; C3 dust as the brownout term; the
  `Boulder` classes as debris hazards; `Maquis`, `House`, `BrickWall`,
  `IrrigatedOrchard` as obstructions.
- **Also needs:** DEM for slope and obstruction height.
- **Feasibility:** DEM.
- **Output:** vector polygons, ranked, with dimensions.
- **Validate:** V3; V1-analogue — any existing helipad or `Pavement` of adequate
  size must pass.

---

# F · Land use & human activity

### F1 — Built-up extent and settlement inventory
*"Where do people live, how big is it, and is it growing?"*

Straightforward, immediately useful, and needs nothing. Morphological closing
over `House` + `Pavement` + `BrickWall` + `PavedRoad` gives settlement polygons;
region-index attributes give area and count; the `PRIORS` entry linking `Car` to
`Pavement`/`PavedRoad`/`House` gives an occupancy cross-check.

- **Signal:** `House`, `Pavement`, `BrickWall`, `PavedRoad`, `Clutter`
  (greenhouses, tents, solar — the periphery of a settlement), `Car`.
- **Also needs:** nothing. A second date for growth.
- **Feasibility:** NOW.
- **Output:** vector polygons with building counts and footprint area; a table.
- **Validate:** V3 is easy here and V7 matters — settlement boundaries are
  threshold-sensitive and will move under a 1 px shift if the closing radius is
  wrong.

### F2 — Activity and occupancy proxy
*"Is this place in use?"*

`Car` is a strange class to include in a *terrain* taxonomy, and its presence is
informative about intent. Vehicle counts per settlement, vehicle density on
`Pavement`, vehicles present at an isolated structure — all cheap, all
single-date, all weak individually and useful in aggregate.

- **Signal:** `Car` counts and their `PRIORS` context; `Clutter`; `Pavement`
  area per `House`; `IrrigatedField` (someone is paying for water, so someone is
  present).
- **Also needs:** nothing single-date. Multi-date turns a weak proxy into a
  trend, which is where the value is.
- **Feasibility:** NOW (weak), strong only with a second date.
- **Output:** a ranked table per settlement; a number.
- **Validate:** V7 across dates. Single-date, essentially unvalidatable — state
  that rather than implying a confidence.

### F3 — Agricultural inventory and irrigation status
*"What is being farmed, how much, and who is irrigating?"*

The taxonomy makes the distinction that matters and states the inference rule
in the class definitions: `IrrigatedField` vs `UnirrigatedOrchard` vs
`IrrigatedOrchard`. In a Mediterranean summer, green and rectangular means
irrigated, which means abstraction, which means infrastructure. Season is
load-bearing and the taxonomy says so explicitly for `GreenGrassland`.

- **Signal:** `IrrigatedField`, `IrrigatedOrchard`, `UnirrigatedOrchard`,
  `GreenGrassland`, `DryGrassland`; `ClayeyDeepSoil` as the arable substrate;
  the `Terrace` classes as terraced cultivation.
- **Also needs:** acquisition date. Without it, `GreenGrassland` vs
  `DryGrassland` is uninterpretable and the whole product loses its inference.
- **Feasibility:** NOW for area accounting; the irrigation *inference* is GAP
  until a date exists.
- **Output:** a table of areas by type; vector field polygons.
- **Validate:** V3; V6 — irrigated parcels should be rectangular with low
  perimeter/area, which the region index already measures, so shape gives a free
  consistency check on the class.

### F4 — Land degradation and grazing pressure
*"Is this landscape being worn down, and where hardest?"*

`DryGrassland → Batha → Garigue → Maquis` is explicitly labelled in the taxonomy
as the Mediterranean **degradation** gradient, so a degradation index is a
straight read of the series. Single-date it gives state; the direction of travel
needs two dates. Distance to settlement gives a testable spatial prediction:
degradation should decay away from `House` clusters and along `DirtRoad`.

- **Signal:** `DEGRADATION_SERIES` as an ordinal; distance-to-`House` and
  distance-to-`DirtRoad` from the chip index's anchor distance fields;
  `Rendzina` and bare rock exposure as advanced degradation.
- **Also needs:** nothing for state. A second date for trend. Rainfall to
  separate drought from grazing — the two look identical in one frame.
- **Feasibility:** NOW for state.
- **Output:** continuous index or four ordinal classes; a distance-decay curve.
- **Validate:** V6 via the distance-decay prediction, which the product is not
  fitted to and which is a genuine test; V8.

---

# G · Ecology & vegetation

### G1 — Habitat patch quality and fragmentation
*"How much intact shrubland is left, and in what size pieces?"*

Pure region-index arithmetic: `Maquis` patch area distribution, compactness
(edge effect), and adjacency. Standard landscape-ecology metrics, no new
machinery, and the region index was built for exactly this shape of query.

- **Signal:** `Maquis`, `Garigue`; region `area_m2`, `compactness`, `perimeter_m`
  and the neighbour graph; `PavedRoad`, `DirtRoad`, `IrrigatedField` as
  fragmenting features.
- **Also needs:** nothing.
- **Feasibility:** NOW.
- **Output:** ranked patch table plus summary metrics; a number.
- **Validate:** V7 across tile boundaries — patch metrics are notoriously
  sensitive to the analysis extent, and a patch cut by a tile edge is a
  measurement artifact that must be handled explicitly.

### G2 — Ecological connectivity corridors
*"Can wildlife move between these patches?"*

Structurally identical to A3 with a different cost table: `Maquis` becomes cheap
rather than expensive, roads become barriers rather than discounts. Worth noting
because it demonstrates that the routing machinery is the reusable asset and the
cost table is the product.

- **Signal:** `DEGRADATION_SERIES` inverted; `PavedRoad`, `House`, `BrickWall`
  as barriers; `Water` as an attractor.
- **Also needs:** nothing structural. Species requirements to mean anything
  biological.
- **Feasibility:** NOW.
- **Output:** corridor polygons; a connectivity number between patch pairs.
- **Validate:** V3 with an ecologist rather than a terrain analyst — different
  expert, and the product is worthless without one.

### G3 — Biomass and carbon
*"How much carbon is standing here?"*

`canopy` in the taxonomy is a *cover fraction*, not a height, and biomass needs
height. The `Maquis` definition gives ">1.5 m", which is a floor, not a
measurement — the difference between 1.6 m and 6 m is the entire answer. Nothing
in the pipeline can close that.

- **Signal:** `DEGRADATION_SERIES` cover fractions; `IrrigatedOrchard`,
  `UnirrigatedOrchard`.
- **Also needs:** canopy height — LiDAR, stereo photogrammetry, or GEDI. Plus
  regional allometric equations.
- **Feasibility:** GAP. Do not build a number here; a cover-fraction map is
  honest and a tonnes-per-hectare figure is not.
- **Output:** at best, ordinal cover classes.
- **Validate:** not validatable without field plots.

---

# H · Navigation & orientation

### H1 — Landmark distinctiveness
*"What can I navigate by here?"*

Rarity × size × compactness over the region index. A single `BasaltRockyTerrain`
outcrop in a carbonate landscape is a landmark precisely because the taxonomy
notes basalt is "spatially disjoint from the carbonate units"; a `Maquis` patch
in a `Maquis` landscape is not. Cheap, unusual, and the region index computes
every term already.

- **Signal:** every class, weighted by inverse frequency from the `l0`
  histogram; `Water`, `House`, `BasaltRockyTerrain`, the `Boulder` classes,
  isolated `Maquis` as high-salience.
- **Also needs:** nothing.
- **Feasibility:** NOW.
- **Output:** ranked table of landmarks with coordinates and a description.
- **Validate:** V3, or the operational test — hand someone the top ten and see
  whether they can self-locate.

### H2 — Terrain-association fingerprint
*"Where am I?" / "Do these two maps show the same place?"*

Use the chip index's per-chip class histogram plus class-pair interface lengths
as a matching descriptor: a compact, illumination-invariant, season-tolerant
signature for a patch of ground. Applications in GPS-denied positioning and in
tile deduplication. The chip index was built as a triage unit but it is already
the right data structure for this.

- **Signal:** the full 45-class histogram per chip plus interface lengths;
  `class_distance()` as the histogram metric so near-miss confusions do not
  break matching.
- **Also needs:** nothing.
- **Feasibility:** NOW.
- **Output:** a match score; a ranked candidate location table.
- **Validate:** self-supervised and genuinely rigorous — hold out chips, match
  them back, measure top-1 accuracy. Alone in this document, this product has a
  real quantitative metric with no expert required. V8 is the key robustness
  test: matching must survive `Limestone`↔`Dolomite` substitution, since that
  confusion is guaranteed to happen.

### H3 — Auto-gazetteer and AOI terrain brief
*"Describe this place to me."*

Name features from class plus geometry plus relative position ("the dolomite
boulder field 400 m north-east of the wadi crossing"), then compose an AOI
narrative. This is the `l0`/`l1` digest plus the region index handed to the LLM
— the cheapest product in the catalogue and the one most likely to be what a
non-specialist actually wanted when they asked for a map.

- **Signal:** everything; region centroids, areas, and adjacency for the
  relative-position language.
- **Also needs:** the georeference for real coordinates. `LabelRaster` carries
  the transform; nothing downstream uses it yet.
- **Feasibility:** NOW.
- **Output:** a gazetteer table plus prose.
- **Validate:** V3 for factual accuracy — every claim in the prose must be
  traceable to a region id, which is enforceable at generation time and should
  be. Prose without provenance is the failure mode.

---

# 1 · Recommendation table

Value is to an end user, not to the project. Feasibility is against today's
inputs. **Score** = value (1–5) × feasibility multiplier (NOW 1.0, NOW\* 0.8,
DEM 0.5, DEM+ 0.4, GAP 0.2).

| # | Product | Group | Value | Feas. | Score | Verdict |
|---|---|---|---|---|---|---|
| A2 | GO/SLOW-GO/NO-GO overlay | mobility | 5 | NOW | **5.0** | **Build first** |
| E1 | Diggability / excavatability | engineering | 5 | NOW | **5.0** | **Build first** |
| A5 | Obstacle inventory | mobility | 4 | NOW | **4.0** | **Build first** |
| A1 | Trafficability raster | mobility | 5 | NOW\* | 4.0 | Exists; re-express as A2 |
| B1 | Overhead concealment | observation | 5 | NOW\* | 4.0 | Exists; pair with B4 |
| A3 | Least-cost route | mobility | 5 | NOW\* | 4.0 | Next, on top of A2 |
| E2 | Bearing / hardstanding | engineering | 4 | NOW | 4.0 | Cheap, strong V2 |
| F1 | Built-up extent | land use | 4 | NOW | 4.0 | Cheap, uncontroversial |
| H2 | Terrain fingerprint | navigation | 4 | NOW | 4.0 | Only product with a real metric |
| D1 | Fire fuel load | hazard | 4 | NOW\* | 3.2 | Exists; blocked on season |
| C3 | Dust potential | hydrology | 3 | NOW | 3.0 | Underrated, needs no DEM |
| E3 | Aggregate siting | engineering | 3 | NOW | 3.0 | Niche but decisive when needed |
| H3 | Auto-gazetteer / brief | navigation | 3 | NOW | 3.0 | Cheapest thing here |
| H1 | Landmark distinctiveness | navigation | 3 | NOW | 3.0 | Cheap |
| F3 | Agricultural inventory | land use | 3 | NOW | 3.0 | Area now, inference needs date |
| F4 | Degradation index | land use | 3 | NOW | 3.0 | Testable distance-decay |
| G1 | Habitat fragmentation | ecology | 3 | NOW | 3.0 | Pure region-index arithmetic |
| A4 | Chokepoints & defiles | mobility | 5 | NOW\*/DEM | 2.8 | Half of it needs a DEM |
| C4 | Water point inventory | hydrology | 3 | NOW | 2.7 | Season caveat |
| B2 | LOS / intervisibility | observation | 5 | DEM+ | 2.0 | **Most-wanted, blocked** |
| B5 | Concealed approach | observation | 5 | NOW\*/DEM+ | 2.0 | Needs B2 to be real |
| B4 | Cover from fire | observation | 4 | DEM | 2.0 | Ship with B1 or rename B1 |
| C1 | Ponding / drainage | hydrology | 4 | DEM | 2.0 | Exists but is a stub today |
| E5 | HLS / DZ siting | engineering | 4 | DEM | 2.0 | Three of four terms ready |
| G2 | Ecological corridors | ecology | 2 | NOW | 2.0 | Same engine as A3 |
| C2 | Wadi network | hydrology | 4 | NOW\*/DEM | 2.0 | Crude proxy now |
| B3 | OP siting | observation | 4 | DEM+ | 1.6 | After B2 |
| D3 | Slope instability | hazard | 3 | DEM | 1.5 | Self-validating via Boulder |
| C5 | Erosion susceptibility | hydrology | 3 | DEM | 1.5 | Best validation story here |
| E4 | Route construction cost | engineering | 3 | DEM | 1.5 | Composition of E1+A3+DEM |
| F2 | Activity proxy | land use | 3 | NOW | 1.5 | Weak single-date |
| B6 | Shadow-derived height | observation | 3 | GAP(tiny) | 1.2 | Unblocked by two numbers |
| A6 | Wet-season delta | mobility | 4 | GAP | 0.8 | Needs date + rainfall |
| D4 | Flash-flood exposure | hazard | 4 | GAP | 0.8 | Needs catchment DEM |
| D5 | Surface disturbance | hazard | 4 | GAP | 0.8 | S6's problem; two dates |
| D2 | Fire spread | hazard | 3 | GAP | 0.6 | Ship the firebreak map only |
| G3 | Biomass / carbon | ecology | 2 | GAP | 0.4 | Do not build |

**37 products. Pick these three:**

1. **A2 — GO/SLOW-GO/NO-GO.** Not new work; it is A1 with the output question
   answered. It converts an unfalsifiable float into three named classes an
   expert can argue with, which is the only validation channel that exists. It
   also answers open question 1 in S4-products.md, which currently blocks
   constant-tuning on every mobility product.
2. **E1 — Diggability.** Highest value-per-unit-effort in the catalogue. Nothing
   but the label raster, and it consumes more of the taxonomy's structure than
   any other product — see section 3. It also has an unusually good validation
   story via existing quarries and road cuttings.
3. **A5 — Obstacle inventory.** A list, not a score. No thresholds to negotiate,
   checkable line by line by a non-expert, and it is what people reach for first.
   Low ceiling but it will be right, and being right about something small is
   how the rest earns trust.

Deliberately not in the top three: **B2 (LOS)**, which is the most-requested
product in the project and stays blocked until a DEM exists. Say that plainly
rather than shipping an approximation.

---

# 2 · The dependency story: one input dominates

## A DEM. Nothing else is close.

Counting against the tiers above:

| Input | Products it unlocks outright | Products it materially upgrades | Total touched |
|---|---|---|---|
| **DEM** (+ derived viewshed, flow accumulation) | **11** — A4, B2, B3, B4, C1, C2, C5, D3, E4, E5, B5 | **6** — A1, A2, A3, D2, D4, A6 | **17 of 37** |
| Acquisition date + sun angle | 2 — B6, part of F3 | 3 — D1, C4, A6 | 5 |
| Rainfall series | 1 — A6 | 3 — C5, D4, F4 | 4 |
| Geological map | 0 | 3 — E3, D3, plus S2 label accuracy | 3 |
| Second acquisition | 1 — D5 | 3 — F1, F2, F4 | 4 |
| Canopy height | 1 — G3 | 2 — B1, B2 | 3 |

The gap is not close, and the raw count understates it, for three reasons.

**First: the DEM is not an enhancement, it is a missing operand.** Three of the
four shipped products already have a slope term wired in and it is multiplying by
a zero array. `drainage` without elevation reduces to a soil mask with two
constant terms — it is presently answering a different question than its name
claims. This is not "we could do better with a DEM"; it is "one product is
currently mislabelled and two are running on half their inputs".

**Second: the DEM validates the taxonomy itself.** Nine of the 45 classes are
defined by geometry that only a DEM can express, and the `PRIORS` table already
encodes the tests:

- `MaralBadlands` requires slope 8–90° (`Prior` with `slope_deg=(8.0, 90.0)`).
  On flat ground it is a definitional contradiction.
- `LimestoneRockDipSlope` and `NariRockDipSlope` require
  `aspect_variance_max=0.25` — a dip slope is a *coherent planar facet* and high
  aspect variance falsifies the label regardless of what the pixels looked like.
- `HydromorpicSoil` requires slope 0–5° and a topographic low.
- The four `Terrace` classes (`MaralTerrace`, `LimestoneTerrace`,
  `DolomiteTerrace`, `NariTerrace`) are defined as "a near-flat facet
  interrupting a slope" — a statement about a *slope profile*, which is
  unverifiable without one.

Without a DEM, those nine classes are unfalsifiable: the S1 audit cannot check
them, the S2 adjudicator cannot use them as evidence, and every product built on
them inherits an error nobody can measure. **The DEM is simultaneously the
largest product input and the largest accuracy lever**, and no other candidate
input is both.

**Third: it is obtainable.** Unlike canopy height (needs LiDAR or a GEDI join),
a geological map (needs a rights-cleared national survey), or a second
acquisition (needs a tasking decision and co-registration work), elevation data
for this AOI exists at multiple resolutions today. The real decision is
resolution, and S4-products.md open question 4 is right to flag it: slope
computed at 0.3 m/px is micro-relief and noise, while vehicle trafficability
responds to slope over several metres. **Specify the smoothing window before
acquiring the DEM, not after** — a 30 m global DEM will not resolve a terrace or
a dip slope, and a 0.5 m photogrammetric DEM will need aggressive smoothing for
mobility work and none at all for the morphology checks. Those are two different
derived layers from one source, and both are needed.

Runner-up worth noting for cost: **the acquisition timestamp**. It touches five
products, it unblocks B6 entirely, and it costs nothing — it is metadata that
already exists on every image and simply is not plumbed through `LabelRaster`.
It should be added regardless of what else is decided, because `GreenGrassland`
vs `DryGrassland`, `IrrigatedField`'s entire inference, and the fire fuel model
are all uninterpretable without it.

---

# 3 · What the taxonomy tells us about intent

This section is **inference from the class list**, not documented fact. Stated
confidences are my own.

## The argument

A pure land-cover scheme does not look like this. CORINE, LCCS, and every
national land-cover product would carry one or two bare-rock classes. This
taxonomy spends **21 of 45 classes on rock**, and — the load-bearing point —
splits them primarily by **morphology**, not by lithology:

| Lithology | Morphologies |
|---|---|
| Limestone | RockyTerrain, Boulder, StoneyTerrain, BeddedRock, RockDipSlope, Terrace (6) |
| Dolomite | RockyTerrain, Boulder, StoneyTerrain, Terrace (4) |
| Nari | RockyTerrain, StoneyTerrain, RockDipSlope, Terrace (4) |
| Basalt | RockyTerrain, Boulder, StoneyTerrain (3) |
| Marl | Badlands, SmoothRockSlopes, Terrace (3) |
| Chalk | SmoothRockSlopes (1) |

Four observations, each of which independently points the same way.

**1. The morphology axis is a surface-behaviour ladder, not a geological one.**
`Boulder` / `StoneyTerrain` / `RockyTerrain` is a distinction by **clast size and
bedrock exposure**. Geologists do not subdivide a formation that way — clast
size is not a lithological property, it is a *surface* property. Vehicle-going
tables and engineering-soil classifications do subdivide exactly that way,
because block size and bedrock exposure determine whether you can drive on it or
dig it. If the goal were geological mapping, you would carry the lithology axis
and drop morphology. If the goal were land cover, you would carry neither at this
resolution. Carrying both, with morphology as the *finer* axis, means the target
variable is what the ground *does*, not what it *is*.

**2. `Terrace` appears on four different lithologies.** A terrace is a flat spot
on a slope. Its physical composition barely matters and its operational meaning
is total: it is where you can drive, park, land, build, camp, or cultivate on
otherwise unusable ground. Spending four classes on a *shape* replicated across
four *materials* is only rational if the shape is what you care about.

**3. `RockDipSlope` carries a geometric constraint in its definition.** "Requires
LOW aspect variance across the polygon — high aspect variance falsifies this
label regardless of how the pixels looked." A dip slope is a smooth planar facet
parallel to bedding, and its distinguishing practical property is that you slide
off it and cannot get purchase on it. Note it exists for Limestone and Nari but
not Marl or Chalk — soft rock does not form them. That asymmetry is expert
knowledge deliberately encoded, and it is knowledge about *slabs*, not strata.

**4. Nari is the clincher.** Nari is a calcrete crust — a near-surface hardpan,
typically a metre or two thick, that forms *over* other units. It is not a
formation and does not appear as a separate unit on a geological map; the
`NariRockyTerrain` definition says so, "Nari CAPS other units". Spending three of
45 classes on a thin crust is only justified if the crust changes what you can do
on the ground: you must break through it to dig, and it is excellent bearing
material to build on. There is no land-cover reason and no mapping reason. There
is an engineering reason.

**Corroborating, outside the rock grid:**

- **Roads split by grade, not by material.** `PavedRoad` / `DirtRoad` /
  `DirtRoadB` — and `DirtRoadB` is defined purely as "narrower, rougher, or less
  maintained". That is a going distinction with no other purpose.
- **Soils split by wet behaviour.** `HydromorpicSoil` ("seasonally
  waterlogged"), `ClayeyDeepSoil` ("very poor wet trafficability"), `Clayeysoil`
  ("shrink-swell; poor trafficability when wet"). Three of five soil definitions
  mention trafficability or wetness explicitly. A pedological scheme would use
  horizons and taxonomy; this one uses behaviour.
- **`Car` is a class in a terrain taxonomy.** Terrain does not contain cars.
  Its presence means the product is about what is *happening* on the terrain, not
  only what the terrain *is*.
- **`Clutter` is defined by function, not appearance** — "greenhouses, pylons,
  fences, tents, solar panels, debris, quarry infrastructure ... the segmenter
  saying 'something is here and I lack a class for it'". A land-cover scheme
  puts unknowns in a null class. This one treats them as a detection target.
- **What is absent.** No species, no crop types (only irrigation *status*), no
  building types, no impervious/pervious split, no wetland classes, no soil
  horizons. Every one of those is standard in a land-cover or ecological scheme
  and none is here.

## The read

**The taxonomy was designed for cross-country movement and ground engineering,
with concealment as a strong secondary.** It is a going-and-going-surface scheme
wearing land-cover clothes. On that reading, the real target products are:

1. **A1/A2 trafficability and mobility classes** — the morphology ladder, the
   road grades, and the soil wet-behaviour splits all feed this and little else.
2. **E1/E2 diggability and bearing** — the only reading under which Nari,
   `BeddedRock`, and the `StoneyTerrain`/`RockyTerrain`/`Boulder` split are all
   worth their class budget simultaneously.
3. **B1/B4 concealment and cover** — the vegetation series carries explicit cover
   and height thresholds, which is a concealment ladder.

**Confidence: ~0.8** that mobility and surface engineering are the primary design
target. The rock morphology grid is the strong evidence and it is not standard
anything — I know of no published land-cover or geological scheme that
subdivides four lithologies by clast size and facet geometry the way this does.
Someone chose that, and clast size and facet geometry are what tracks and shovels
care about.

**Confidence: ~0.5** that concealment was a first-class design goal rather than
inherited. `DryGrassland → Batha → Garigue → Maquis` is the standard
Mediterranean formation series used throughout Israeli and Levantine vegetation
mapping — it is Zohary-lineage phytogeography, and its presence is what you would
expect from *any* regional scheme in this area. It happens to be a concealment
ladder; that may be a fortunate coincidence rather than a design decision. The
explicit cover-percentage and height thresholds in the definitions push me toward
"designed", but not far.

**Confidence: ~0.35** on hydrology as a design target. `Water`,
`HydromorpicSoil`, `ClayeyDeepSoil` and `MaralBadlands` support drainage work,
but there is no channel class, no wadi class, no floodplain class — a striking
omission in a Sinai/Negev taxonomy where the wadi network is the dominant
landform and the de facto road network. If hydrology were a target, that gap
would not exist. I read the hydrological classes as *consequences* of the soil
and lithology axes rather than as intent.

**One dissenting signal, recorded honestly.** The lithology axis itself
(Limestone / Dolomite / Nari / Basalt / Chalk / Marl) does *not* serve mobility
or engineering well — as the `DolomiteRockyTerrain` definition concedes,
limestone and dolomite are "effectively indistinguishable in RGB", and they have
near-identical `traffic` values. A pure mobility scheme would have collapsed
them into "hard carbonate" and saved eight classes. The most likely explanation
is that the lithology axis was **inherited from an existing regional geological
legend** and the morphology axis was **added for the use case**, giving a
cross-product where one axis is legacy and one is intent. If that is right, the
practical implication is direct: collapsing `Limestone`/`Dolomite` for every
product in this catalogue costs nothing operationally and removes the single
worst confusion pair in the map. **Confidence ~0.6**, and it is cheap to test —
ask whoever wrote the class list where the lithology names came from.
