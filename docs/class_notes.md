# Class notes — what the analyst knows that the name does not say

> One section per class. Fill what you know, skip what you do not; a
> blank field reads as unfilled, not as empty. Lines starting with `>`
> are instructions and are ignored by the parser.
>
> **Exactly one field is read by code today: `confused_with`.** The other
> three named below are parsed, stored, shown and validated, but no
> solution consumes them yet — filling them changes no output. That is
> not a reason to skip them: they are the content the planned consumers
> are waiting for, and writing them down now is cheaper than
> reconstructing them later. It *is* a reason not to expect a number to
> move when you do.
>
> - `confused_with` — `OtherClass — how you tell them apart; ...`
>   **Consumed today**, by `solutions/s2_adjudicate.py`: it seeds S2's
>   candidate shortlist and the reason text is quoted back in the
>   evidence. A measured confusion beats the taxonomy's distance metric
>   every time.
> - `season` — does this label depend on the date the imagery was taken?
>   **Recorded, not yet consumed.** S6 has its own hard-coded seasonal
>   pair table (`s6_change.py`) and does not read this field.
> - `never` — what this can never be adjacent to, or never look like.
>   **Recorded, not yet consumed.** Intended as a candidate prior.
>   Admitted ONLY if it is a statement about geometry, position or
>   physics — a statement about how the thing came to exist is a
>   statement about the world, not about the label. See
>   `taxonomy.RETIRED_PRIORS` for what happened last time that line was
>   crossed.
> - `scale` — typical size and shape as a polygon. **Recorded, not yet
>   consumed.** Intended to replace the guessed area and elongation bands
>   in S2, which still come from `taxonomy.PRIORS`.
>
> Note that `segmap notes` prints a `# machine-read fields:` line naming
> all four. That header is `class_notes.MACHINE_READ`, which is a list of
> intent, not of wiring; only `confused_with` is wired.
>
> `source` and `confidence` are not bureaucracy: an attributed claim can
> be checked with its author, an unattributed one has to be re-derived or
> thrown away.

> `what_it_is` is seeded below with the one-line definition currently in
> `taxonomy.py`. Those were written by a programmer from a textbook and
> are exactly what needs replacing — overwrite them.

## Unclassified
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] No confident label. Either genuinely ambiguous imagery or a target outside the 47-class vocabulary. High-value hunting ground for out-of-vocabulary detection.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Clutter
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Man-made or anomalous material the taxonomy has no word for: greenhouses, pylons, fences, tents, solar panels, debris, quarry infrastructure. Not noise -- it is the segmenter saying 'something is here and I lack a class for it'.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Shadow
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Illumination artifact, not a surface. Whatever is underneath is unlabelled. Shadow length and direction are recoverable evidence about object height and sun angle.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## BrickWall
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Built masonry wall. Linear, high compactness, blocks vehicle movement and line of sight.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## House
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Building footprint. Compact, rectilinear, usually clustered and adjacent to Pavement or PavedRoad.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## GreenGrassland
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Herbaceous cover, photosynthetically active. In a Mediterranean climate this implies either the wet season or irrigation -- season is load-bearing for this label.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Car
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Individual vehicle. Small (roughly 4-25 m2), rectangular, found on Pavement, PavedRoad, DirtRoad, or beside House.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Pavement
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Sealed non-road surface: parking, yards, plazas, hardstanding. Adjacent to House.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## PavedRoad
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Sealed road. Long, narrow, high elongation, network-connected.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DirtRoad
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Unsealed graded track, main grade. Linear, follows terrain contours more closely than a paved road.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DirtRoadB
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Unsealed track, secondary/lower grade -- narrower, rougher, or less maintained than DirtRoad.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Water
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Open water: reservoir, pool, channel. Occupies topographic lows or is an engineered impoundment.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## MaralBadlands
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Marl badlands: intensely dissected, high drainage density, steep bare slopes, sparse vegetation. Requires real relief -- badlands on flat DEM is a contradiction.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## MaralSmoothRockSlopes
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Smooth, planar marl slopes with little surface rock. Low roughness, moderate to steep.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## MaralTerrace
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Structural bench developed on marl: near-flat facet interrupting a slope.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## TerraRosa
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Terra rossa: a strongly red, fine-textured, clay-rich Mediterranean soil. Low stone content at the surface, blocky structure, shallow to moderate depth. The label is a SOIL TYPE: it describes the soil, and carries no claim about the parent rock under or beside it.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## ClayeySoil
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Clay-rich soil, moderate depth. Shrink-swell; poor trafficability when wet.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Rendzina
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Rendzina: a shallow, pale (grey to buff), stony, strongly calcareous soil with a thin profile and abundant carbonate fragments. The label is a SOIL TYPE: it describes the soil, and carries no claim about the parent rock under or beside it.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## HydromorpicSoil
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Hydromorphic soil: seasonally waterlogged, gleyed. Occupies drainage lows and closed depressions. Should be topographically low and near Water or a drainage line.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## ClayeyDeepSoil
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Deep clay profile, typically on valley floors and alluvial fills. Flat, agriculturally productive, very poor wet trafficability.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## UnirrigatedOrchard
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Rainfed tree crop (olive, almond, carob). Regular planting geometry, wide tree spacing, dry inter-row in summer.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## IrrigatedOrchard
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Irrigated tree crop. Regular geometry, denser and greener canopy than the unirrigated form, green through the dry season.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## IrrigatedField
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Irrigated field crop. Rectangular, sharp straight edges, green in summer. Anything rectangular and green in late summer is almost certainly irrigated.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Batha
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Batha: dwarf-shrub garrigue, the most degraded stage of the Mediterranean formation series. Low woody cover (<25%), height under ~0.5 m.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Garigue
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Garigue: open low shrubland, intermediate degradation stage between batha and maquis. Woody cover roughly 25-50%, height ~0.5-1.5 m.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## Maquis
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Maquis: dense evergreen sclerophyll shrubland/woodland, the least degraded stage. Woody cover >50%, height >1.5 m. Favours north-facing slopes and wetter aspects.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DryGrassland
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Senescent herbaceous cover. The dry-season expression of GreenGrassland -- a GreenGrassland/DryGrassland difference between two dates is phenology, not change.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneRockyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Limestone with extensive exposed bedrock and rubble; broken, irregular surface.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneBoulder
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Limestone boulder field: large detached blocks. Impassable to wheeled vehicles.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneStoneyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Limestone stony ground: abundant small clasts over soil, bedrock largely covered.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneHardRockLineament
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Linear hard-limestone outcrop: a resistant bed or fracture-controlled ridge expressed as a narrow, highly elongated strip. Geometry is the label -- a compact blob of this class contradicts its own definition.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneBeddedRock
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Exposed limestone bedding planes; visible layering, stepped micro-relief.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneRockDipSlope
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Limestone dip slope: a coherent planar facet parallel to bedding. Requires LOW aspect variance across the polygon -- high aspect variance falsifies this label regardless of how the pixels looked.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## LimestoneTerrace
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Structural bench on limestone: near-flat step in a slope profile.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DolomiteRockyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Dolomite with extensive exposed bedrock. Effectively indistinguishable from limestone in RGB -- separation requires the geological map.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DolomiteBoulder
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Dolomite boulder field: large detached blocks, impassable.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DolomiteStoneyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Dolomite stony ground: abundant clasts over soil.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## DolomiteTerrace
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Structural bench developed on dolomite.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## NariRockyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Nari (calcrete crust) with exposed rock. Nari CAPS other units -- expect thin bands at plateau edges and slope crests, not large valley-floor blobs.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## NariStoneyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Nari stony ground: calcrete fragments over soil.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## NariRockDipSlope
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Nari dip slope: planar calcrete facet. Same low-aspect-variance requirement as any dip slope.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## NariTerrace
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Structural bench capped by nari crust.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## BasaltRockyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Basalt with exposed rock. Dark-toned; volcanic terrain, spatially disjoint from the carbonate units.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## BasaltBoulder
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Basalt boulder field. Impassable.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## BasaltStoneyTerrain
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Basalt stony ground: basalt clasts over soil.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## ChalkSmoothRockSlopes
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Chalk smooth slopes: soft white carbonate, low surface roughness. Chalk forms no boulder field, no dip slope and no rocky terrain -- only smooth slopes and terraces.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:

## ChalkTerrace
aka:
what_it_is: [FROM taxonomy.py, NEEDS REVIEW] Structural bench developed on chalk: near-flat step interrupting a smooth chalk slope.
looks_like:
confused_with:
implies:
scale:
season:
never:
source:
confidence:
