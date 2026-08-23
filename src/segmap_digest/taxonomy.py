"""The 45-class Smart Terrain taxonomy, plus the structure hiding inside it.

The class list is flat on the wire but is really five overlapping ontologies:
artifacts, anthropogenic objects, hydrology, pedology, a lithology x
geomorphology grid, vegetation formations, and land use. The reasoning layer
works over that structure, not over the flat list.

DEFINITIONS ARE A STARTING POINT AND NEED EXPERT REVIEW. They are the highest
leverage artifact in the whole pipeline -- an LLM reasons over these words, not
over the integer class ids the segmenter learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- sub-ontology groups ---------------------------------------------------

ARTIFACT = "artifact"
ANTHROPOGENIC = "anthropogenic"
HYDROLOGY = "hydrology"
SOIL = "soil"
ROCK = "rock"
VEGETATION = "vegetation"
LANDUSE = "landuse"


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str
    group: str
    definition: str
    lithology: str | None = None
    morphology: str | None = None
    # Coarse trafficability hint for a wheeled vehicle on flat ground.
    # 0 = impassable, 1 = trivially passable. Slope is applied separately.
    traffic: float = 0.5
    # Fractional woody/canopy cover, for concealment reasoning.
    canopy: float = 0.0


def _c(*args, **kwargs) -> ClassDef:
    return ClassDef(*args, **kwargs)


CLASSES: list[ClassDef] = [
    _c(0, "Unclassified", ARTIFACT,
       "No confident label. Either genuinely ambiguous imagery or a target outside "
       "the 45-class vocabulary. High-value hunting ground for out-of-vocabulary detection.",
       traffic=0.5),
    _c(1, "Clutter", ARTIFACT,
       "Man-made or anomalous material the taxonomy has no word for: greenhouses, pylons, "
       "fences, tents, solar panels, debris, quarry infrastructure. Not noise -- it is the "
       "segmenter saying 'something is here and I lack a class for it'.",
       traffic=0.4),
    _c(2, "Shadow", ARTIFACT,
       "Illumination artifact, not a surface. Whatever is underneath is unlabelled. Shadow "
       "length and direction are recoverable evidence about object height and sun angle.",
       traffic=0.5),

    _c(3, "BrickWall", ANTHROPOGENIC,
       "Built masonry wall. Linear, high compactness, blocks vehicle movement and line of sight.",
       traffic=0.0),
    _c(4, "House", ANTHROPOGENIC,
       "Building footprint. Compact, rectilinear, usually clustered and adjacent to Pavement "
       "or PavedRoad.", traffic=0.0),
    _c(5, "GreenGrassland", VEGETATION,
       "Herbaceous cover, photosynthetically active. In a Mediterranean climate this implies "
       "either the wet season or irrigation -- season is load-bearing for this label.",
       traffic=0.85, canopy=0.0),
    _c(6, "Car", ANTHROPOGENIC,
       "Individual vehicle. Small (roughly 4-25 m2), rectangular, found on Pavement, PavedRoad, "
       "DirtRoad, or beside House.", traffic=0.0),
    _c(7, "Pavement", ANTHROPOGENIC,
       "Sealed non-road surface: parking, yards, plazas, hardstanding. Adjacent to House.",
       traffic=1.0),
    _c(8, "PavedRoad", ANTHROPOGENIC,
       "Sealed road. Long, narrow, high elongation, network-connected.", traffic=1.0),
    _c(9, "DirtRoad", ANTHROPOGENIC,
       "Unsealed graded track, main grade. Linear, follows terrain contours more closely than "
       "a paved road.", traffic=0.9),
    _c(10, "DirtRoadB", ANTHROPOGENIC,
        "Unsealed track, secondary/lower grade -- narrower, rougher, or less maintained than "
        "DirtRoad.", traffic=0.75),
    _c(11, "Water", HYDROLOGY,
        "Open water: reservoir, pool, channel. Occupies topographic lows or is an engineered "
        "impoundment.", traffic=0.0),

    # Marl ("Maral") -- soft, fine-grained sedimentary rock. Dissects into badlands.
    _c(12, "MaralBadlands", ROCK,
        "Marl badlands: intensely dissected, high drainage density, steep bare slopes, sparse "
        "vegetation. Requires real relief -- badlands on flat DEM is a contradiction.",
        lithology="Marl", morphology="Badlands", traffic=0.1),
    _c(13, "MaralSmoothRockSlopes", ROCK,
        "Smooth, planar marl slopes with little surface rock. Low roughness, moderate to steep.",
        lithology="Marl", morphology="SmoothRockSlopes", traffic=0.35),
    _c(14, "MaralTerrace", ROCK,
        "Structural bench developed on marl: near-flat facet interrupting a slope.",
        lithology="Marl", morphology="Terrace", traffic=0.7),

    _c(15, "TerraRosa", SOIL,
        "Terra rossa: red residual decalcification clay. Forms on HARD carbonate (limestone, "
        "dolomite, nari) under Mediterranean climate, typically in karst pockets and on gentle "
        "slopes. Terra rossa surrounded by chalk or marl is a strong misclassification signal.",
        traffic=0.7),
    _c(16, "Clayeysoil", SOIL,
        "Clay-rich soil, moderate depth. Shrink-swell; poor trafficability when wet.",
        traffic=0.6),
    _c(17, "Rendzina", SOIL,
        "Rendzina: shallow, pale, calcareous soil formed on SOFT carbonate -- chalk and marl. "
        "The complement of terra rossa. Rendzina over basalt is implausible.", traffic=0.7),
    _c(18, "HydromorpicSoil", SOIL,
        "Hydromorphic soil: seasonally waterlogged, gleyed. Occupies drainage lows and closed "
        "depressions. Should be topographically low and near Water or a drainage line.",
        traffic=0.3),
    _c(19, "ClayeyDeepSoil", SOIL,
        "Deep clay profile, typically on valley floors and alluvial fills. Flat, agriculturally "
        "productive, very poor wet trafficability.", traffic=0.55),

    _c(20, "UnirrigatedOrchard", LANDUSE,
        "Rainfed tree crop (olive, almond, carob). Regular planting geometry, wide tree spacing, "
        "dry inter-row in summer.", traffic=0.6, canopy=0.4),
    _c(21, "IrrigatedOrchard", LANDUSE,
        "Irrigated tree crop. Regular geometry, denser and greener canopy than the unirrigated "
        "form, green through the dry season.", traffic=0.55, canopy=0.6),
    _c(22, "IrrigatedField", LANDUSE,
        "Irrigated field crop. Rectangular, sharp straight edges, green in summer. Anything "
        "rectangular and green in late summer is almost certainly irrigated.",
        traffic=0.7, canopy=0.1),

    _c(23, "Batha", VEGETATION,
        "Batha: dwarf-shrub garrigue, the most degraded stage of the Mediterranean formation "
        "series. Low woody cover (<25%), height under ~0.5 m.", traffic=0.7, canopy=0.2),
    _c(24, "Garigue", VEGETATION,
        "Garigue: open low shrubland, intermediate degradation stage between batha and maquis. "
        "Woody cover roughly 25-50%, height ~0.5-1.5 m.", traffic=0.5, canopy=0.4),
    _c(25, "Maquis", VEGETATION,
        "Maquis: dense evergreen sclerophyll shrubland/woodland, the least degraded stage. "
        "Woody cover >50%, height >1.5 m. Favours north-facing slopes and wetter aspects.",
        traffic=0.2, canopy=0.8),
    _c(26, "DryGrassland", VEGETATION,
        "Senescent herbaceous cover. The dry-season expression of GreenGrassland -- a "
        "GreenGrassland/DryGrassland difference between two dates is phenology, not change.",
        traffic=0.85, canopy=0.0),

    # Limestone -- hard carbonate, the fullest morphology set.
    _c(27, "LimestoneRockyTerrain", ROCK,
        "Limestone with extensive exposed bedrock and rubble; broken, irregular surface.",
        lithology="Limestone", morphology="RockyTerrain", traffic=0.25),
    _c(28, "LimestoneBoulder", ROCK,
        "Limestone boulder field: large detached blocks. Impassable to wheeled vehicles.",
        lithology="Limestone", morphology="Boulder", traffic=0.05),
    _c(29, "LimestoneStoneyTerrain", ROCK,
        "Limestone stony ground: abundant small clasts over soil, bedrock largely covered.",
        lithology="Limestone", morphology="StoneyTerrain", traffic=0.55),
    _c(30, "LimestoneBeddedRock", ROCK,
        "Exposed limestone bedding planes; visible layering, stepped micro-relief.",
        lithology="Limestone", morphology="BeddedRock", traffic=0.3),
    _c(31, "LimestoneRockDipSlope", ROCK,
        "Limestone dip slope: a coherent planar facet parallel to bedding. Requires LOW aspect "
        "variance across the polygon -- high aspect variance falsifies this label regardless of "
        "how the pixels looked.", lithology="Limestone", morphology="RockDipSlope", traffic=0.2),
    _c(32, "LimestoneTerrace", ROCK,
        "Structural bench on limestone: near-flat step in a slope profile.",
        lithology="Limestone", morphology="Terrace", traffic=0.65),

    # Dolomite -- hard carbonate, spectrally near-identical to limestone in RGB.
    _c(33, "DolomiteRockyTerrain", ROCK,
        "Dolomite with extensive exposed bedrock. Effectively indistinguishable from limestone "
        "in RGB -- separation requires the geological map.",
        lithology="Dolomite", morphology="RockyTerrain", traffic=0.25),
    _c(34, "DolomiteBoulder", ROCK,
        "Dolomite boulder field: large detached blocks, impassable.",
        lithology="Dolomite", morphology="Boulder", traffic=0.05),
    _c(35, "DolomiteStoneyTerrain", ROCK,
        "Dolomite stony ground: abundant clasts over soil.",
        lithology="Dolomite", morphology="StoneyTerrain", traffic=0.55),
    _c(36, "DolomiteTerrace", ROCK,
        "Structural bench developed on dolomite.",
        lithology="Dolomite", morphology="Terrace", traffic=0.65),

    # Nari -- calcrete crust. Caps other units; occurs as thin plateau-edge bands.
    _c(37, "NariRockyTerrain", ROCK,
        "Nari (calcrete crust) with exposed rock. Nari CAPS other units -- expect thin bands at "
        "plateau edges and slope crests, not large valley-floor blobs.",
        lithology="Nari", morphology="RockyTerrain", traffic=0.3),
    _c(38, "NariStoneyTerrain", ROCK,
        "Nari stony ground: calcrete fragments over soil.",
        lithology="Nari", morphology="StoneyTerrain", traffic=0.6),
    _c(39, "NariRockDipSlope", ROCK,
        "Nari dip slope: planar calcrete facet. Same low-aspect-variance requirement as any "
        "dip slope.", lithology="Nari", morphology="RockDipSlope", traffic=0.25),
    _c(40, "NariTerrace", ROCK,
        "Structural bench capped by nari crust.",
        lithology="Nari", morphology="Terrace", traffic=0.65),

    _c(41, "BasaltRockyTerrain", ROCK,
        "Basalt with exposed rock. Dark-toned; volcanic terrain, spatially disjoint from the "
        "carbonate units.", lithology="Basalt", morphology="RockyTerrain", traffic=0.2),
    _c(42, "BasaltBoulder", ROCK,
        "Basalt boulder field. Impassable.",
        lithology="Basalt", morphology="Boulder", traffic=0.05),
    _c(43, "BasaltStoneyTerrain", ROCK,
        "Basalt stony ground: basalt clasts over soil.",
        lithology="Basalt", morphology="StoneyTerrain", traffic=0.5),

    _c(44, "ChalkSmoothRockSlopes", ROCK,
        "Chalk smooth slopes: soft white carbonate, low surface roughness. Chalk appears in the "
        "taxonomy ONLY in this morphology -- it does not form boulder fields or dip slopes. "
        "Strongly associated with Rendzina soil.",
        lithology="Chalk", morphology="SmoothRockSlopes", traffic=0.4),
]

assert len(CLASSES) == 45, f"expected 45 classes, got {len(CLASSES)}"

BY_ID: dict[int, ClassDef] = {c.id: c for c in CLASSES}
BY_NAME: dict[str, ClassDef] = {c.name: c for c in CLASSES}
NAMES: list[str] = [c.name for c in CLASSES]
N_CLASSES = len(CLASSES)


def cid(name: str) -> int:
    return BY_NAME[name].id


# --- the lithology x geomorphology grid ------------------------------------
# Sparse and asymmetric on purpose: chalk has one morphology, limestone six.
# Any predicted combination outside this grid is a definitional error, not a
# judgement call.

LITHOLOGY_GRID: dict[str, dict[str, int]] = {}
for c in CLASSES:
    if c.lithology and c.morphology:
        LITHOLOGY_GRID.setdefault(c.lithology, {})[c.morphology] = c.id

HARD_CARBONATE = ("Limestone", "Dolomite", "Nari")
SOFT_CARBONATE = ("Chalk", "Marl")


# --- superclasses (collapse 45 -> 9 for coarse reasoning and legible maps) --

SUPERCLASS: dict[str, tuple[str, ...]] = {
    "artifact": ("Unclassified", "Clutter", "Shadow"),
    "built": ("BrickWall", "House", "Pavement"),
    "road": ("PavedRoad", "DirtRoad", "DirtRoadB"),
    "vehicle": ("Car",),
    "water": ("Water",),
    "soil": ("TerraRosa", "Clayeysoil", "Rendzina", "HydromorpicSoil", "ClayeyDeepSoil"),
    "agriculture": ("UnirrigatedOrchard", "IrrigatedOrchard", "IrrigatedField"),
    "vegetation": ("GreenGrassland", "DryGrassland", "Batha", "Garigue", "Maquis"),
    "rock": tuple(c.name for c in CLASSES if c.group == ROCK),
}

SUPERCLASS_OF: dict[int, str] = {}
for _sc, _names in SUPERCLASS.items():
    for _n in _names:
        SUPERCLASS_OF[cid(_n)] = _sc
assert len(SUPERCLASS_OF) == N_CLASSES, "every class must map to exactly one superclass"


# --- ordinal series --------------------------------------------------------
# Confusions inside a series are minor; confusions across groups are not. A CNN
# trained with plain cross-entropy cannot tell the difference.

DEGRADATION_SERIES: tuple[str, ...] = (
    "DryGrassland", "Batha", "Garigue", "Maquis",
)
ROAD_SERIES: tuple[str, ...] = ("DirtRoadB", "DirtRoad", "PavedRoad")


def class_distance(a: int, b: int) -> float:
    """0.0 = same class, 1.0 = maximally different. Ordinal series get credit for
    being near-misses; cross-superclass confusions do not."""
    if a == b:
        return 0.0
    na, nb = BY_ID[a].name, BY_ID[b].name
    for series in (DEGRADATION_SERIES, ROAD_SERIES):
        if na in series and nb in series:
            return abs(series.index(na) - series.index(nb)) / len(series)
    da, db = BY_ID[a], BY_ID[b]
    if da.lithology and db.lithology:
        if da.lithology == db.lithology:
            return 0.35          # same rock, different morphology
        if da.morphology == db.morphology:
            return 0.45          # same morphology, different rock (the RGB-blind case)
        return 0.7
    if SUPERCLASS_OF[a] == SUPERCLASS_OF[b]:
        return 0.4
    return 1.0


# --- anchor classes for distance fields ------------------------------------
# Query-independent, computed once per tile, reused by every detection policy.

ANCHOR_CLASSES: tuple[str, ...] = (
    "PavedRoad", "DirtRoad", "House", "Pavement", "Water", "Maquis", "BrickWall",
)


# --- co-occurrence priors --------------------------------------------------
# World knowledge the segmenter never had access to. Used by the consistency
# audit, and as worked examples of what the reasoning layer can check.

@dataclass(frozen=True)
class Prior:
    subject: str
    kind: str            # "expects" | "contradicts"
    others: tuple[str, ...]
    why: str
    slope_deg: tuple[float, float] | None = None
    aspect_variance_max: float | None = None


PRIORS: tuple[Prior, ...] = (
    Prior("TerraRosa", "expects",
          ("LimestoneRockyTerrain", "LimestoneStoneyTerrain", "DolomiteRockyTerrain",
           "DolomiteStoneyTerrain", "NariStoneyTerrain"),
          "Terra rossa is a residual decalcification clay of hard carbonate."),
    Prior("TerraRosa", "contradicts",
          ("ChalkSmoothRockSlopes", "MaralBadlands", "MaralSmoothRockSlopes",
           "BasaltRockyTerrain"),
          "Terra rossa does not form on soft carbonate or on basalt."),
    Prior("Rendzina", "expects",
          ("ChalkSmoothRockSlopes", "MaralSmoothRockSlopes", "MaralTerrace"),
          "Rendzina is the shallow calcareous soil of chalk and marl."),
    Prior("Rendzina", "contradicts",
          ("BasaltRockyTerrain", "BasaltStoneyTerrain", "BasaltBoulder"),
          "Rendzina requires a carbonate parent material."),
    Prior("MaralBadlands", "expects", (),
          "Badlands require dissected relief; near-flat ground falsifies the label.",
          slope_deg=(8.0, 90.0)),
    Prior("LimestoneRockDipSlope", "expects", (),
          "A dip slope is a coherent planar facet: aspect must be consistent across it.",
          aspect_variance_max=0.25),
    Prior("NariRockDipSlope", "expects", (),
          "A dip slope is a coherent planar facet: aspect must be consistent across it.",
          aspect_variance_max=0.25),
    Prior("HydromorpicSoil", "expects", ("Water", "ClayeyDeepSoil"),
          "Hydromorphic soils occupy drainage lows and closed depressions.",
          slope_deg=(0.0, 5.0)),
    Prior("Car", "expects", ("Pavement", "PavedRoad", "DirtRoad", "House"),
          "Vehicles occur on trafficable surfaces near built-up areas."),
    Prior("Maquis", "expects", ("Garigue", "Batha"),
          "The Mediterranean formation series grades continuously; abrupt maquis/bare "
          "boundaries with no intermediate stage are suspicious."),
)

PRIORS_BY_SUBJECT: dict[str, list[Prior]] = {}
for _p in PRIORS:
    PRIORS_BY_SUBJECT.setdefault(_p.subject, []).append(_p)


def legend(compact: bool = True) -> str:
    """The legend an LLM gets. `compact` omits definitions (for high-volume calls)."""
    if compact:
        return "\n".join(f"{c.id}\t{c.name}\t{c.group}" for c in CLASSES)
    lines = []
    for c in CLASSES:
        extra = f" [{c.lithology}/{c.morphology}]" if c.lithology else ""
        lines.append(f"{c.id}\t{c.name}\t{c.group}{extra}\t{c.definition}")
    return "\n".join(lines)
