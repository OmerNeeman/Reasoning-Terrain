"""S4 -- derived decision products.

This is where 47 classes become useful: a many-to-one mapping from the taxonomy
plus slope plus season into the things someone actually asks for. Tedious to
hand-code for 47 x N combinations, which is exactly why it is worth having a
model help author the tables -- but the evaluation must stay deterministic.

Products implemented (naive):
    trafficability     can a vehicle class cross this
    concealment        how well is a target of a NAMED SIZE hidden from above
    built_fabric       bare-natural -> dense-urban, from the raster itself
    change_volatility  how much this ground is expected to differ between two
                       dates for reasons that are not change

Each returns a float raster in [0, 1] plus a per-region summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..index import RegionIndex
from ..loader import LabelRaster
from ..taxonomy import BY_ID, N_CLASSES, SUPERCLASS_OF, cid

# --- tunable heuristics (all guesses -- see docs/solutions/S4-products.md) --


@dataclass(frozen=True)
class Vehicle:
    name: str
    max_slope_deg: float
    # Minimum surface trafficability the vehicle needs (taxonomy `traffic`).
    min_surface: float
    # How much wet ground degrades performance, 0 = unaffected.
    wet_sensitivity: float
    # Physical width in metres. S5's corridor query uses it to reject
    # passages narrower than the vehicle (single-pixel percolation paths).
    # 0.0 = no width gate (foot). Guesses -- where these should come from:
    # vehicle spec sheets, the same document that owns max_slope_deg.
    width_m: float = 0.0


VEHICLES = {
    "wheeled": Vehicle("wheeled", max_slope_deg=25.0, min_surface=0.45,
                       wet_sensitivity=0.7, width_m=2.5),
    "tracked": Vehicle("tracked", max_slope_deg=35.0, min_surface=0.20,
                       wet_sensitivity=0.4, width_m=3.5),
    "foot": Vehicle("foot", max_slope_deg=45.0, min_surface=0.05,
                    wet_sensitivity=0.15, width_m=0.0),
}

# Slope response: full score below SLOPE_FREE, zero at the vehicle limit.
SLOPE_FREE_DEG = 5.0

# Classes that hold water / stay wet, driving the wet penalty.
WET_CLASSES = ("HydromorpicSoil", "ClayeyDeepSoil", "ClayeySoil", "Water",
               "IrrigatedField", "IrrigatedOrchard")

# --- concealment: concealment OF WHAT? --------------------------------------
#
# The same maquis canopy hides a crouching person and does not hide a truck.
# Until now this file answered that question on the caller's behalf with a
# constant, which is the quiet kind of wrong: the number looked like a property
# of the ground when half of it was an unstated assumption about the target.
# Naming the target does not make the numbers right -- it makes the question
# visible, and it makes the answer a function of something the caller owns.


@dataclass(frozen=True)
class Target:
    name: str
    # Plan-view footprint in m2 -- the area that has to be covered.
    footprint_m2: float
    # Height in metres -- what the relief has to be deeper than.
    height_m: float


# Where these should come from: the customer's own target list, which is a
# document that exists and that nobody has shown us. These three are round
# numbers for a crouching man, a pickup/technical, and a small building.
TARGETS = {
    "person": Target("person", footprint_m2=0.5, height_m=1.7),
    "vehicle": Target("vehicle", footprint_m2=8.0, height_m=2.0),
    "structure": Target("structure", footprint_m2=100.0, height_m=5.0),
}

# Concealment: canopy comes from the taxonomy; these add the rest. All three are
# quoted for the *person* target and scaled down for bigger ones -- see
# `concealment` for how, and Open Question 5 in the doc for the observer
# geometry that none of them account for.
SHADOW_CONCEALMENT = 0.6
BUILT_CONCEALMENT = 0.5      # adjacent to buildings/walls
RELIEF_CONCEALMENT_WEIGHT = 0.25   # steep, broken ground hides things too

# Characteristic plan area of ONE canopy clump -- a single Mediterranean tree or
# shrub crown, ~2 m across. It sets how fast fractional cover stops helping as
# the target grows: for a target of footprint A dropped at random on canopy of
# fractional cover c with clumps of area A_c, the chance of being fully covered
# goes roughly as c ** (1 + A/A_c). Where this should come from: crown-diameter
# statistics per vegetation class, which is the same LiDAR survey that owns
# `ClassDef.canopy`. 4 m2 is a guess.
CANOPY_CLUMP_M2 = 4.0
# ...capped, because that random-placement model goes to zero much faster than
# reality does: a structure is *sited*, not dropped at random, and gets put under
# the densest patch on purpose. The cap is where the model stops being trusted.
CANOPY_EXPONENT_MAX = 6.0

# Window (px) over which slope roughness is measured for the relief term.
# Where this should come from: the footprint of the object being hidden
# divided by the GSD -- a fixed pixel count is only defensible at one GSD.
# It is now a floor rather than the value: `concealment` widens it to the
# target's own footprint, which is what that comment was asking for.
ROUGHNESS_WINDOW_PX = 5

# --- built fabric ----------------------------------------------------------
#
# Per-class position on a bare-natural -> dense-urban ordinal. This is the
# honest per-tile answer to "how populated is this": it is derived from the
# ground the segmenter actually saw, at the segmenter's GSD, rather than from a
# 100 m population grid that is coarser than a city block and that answers a
# different question (where people are registered) than the one being asked
# (what kind of ground is this).
#
# The values are an ordinal, not a density. Nothing here is calibrated against a
# census, and the gaps between the steps are chosen so the ordering survives the
# neighbourhood blend below -- not because 0.35 is a measured property of an
# orchard.
FABRIC_BASE_DEFAULT = 0.0     # bare rock, sand and soil: the natural end
FABRIC_BASE = {
    # Natural vegetation: ground people use and do not build.
    "GreenGrassland": 0.1, "DryGrassland": 0.1,
    "Batha": 0.1, "Garigue": 0.1, "Maquis": 0.1,
    # Open water scores natural even when the reservoir is engineered: the axis
    # is "what kind of ground is this", and water is not ground. The dam wall
    # beside it is House/BrickWall and scores as built, which is the right split.
    "Water": 0.0,
    # Managed land, not settlement -- ordinally between the two, and the step
    # most likely to be wrong, since an intensive greenhouse belt and a rainfed
    # olive terrace are both "agriculture" here.
    "UnirrigatedOrchard": 0.35, "IrrigatedOrchard": 0.35, "IrrigatedField": 0.35,
    # Surfaced ground: someone paid to seal it, but nobody lives on it.
    "Pavement": 0.6, "PavedRoad": 0.6,
    # Unsealed tracks sit lower than the owner's 0.6 on purpose: a graded dirt
    # road is a weaker statement about settlement than a kerb is -- half the
    # tracks in an arid AOI serve grazing, not a village.
    "DirtRoad": 0.5, "DirtRoadB": 0.45,
    "Car": 0.6,
    # Buildings: the dense-urban end.
    "House": 0.9, "BrickWall": 0.9,
    # Clutter is the segmenter saying "man-made, and I have no word for it":
    # greenhouses, pylons, tents, solar farms, quarry plant. Anthropogenic, but
    # not settlement, so it sits below a roof.
    "Clutter": 0.7,
    # Shadow and Unclassified assert nothing about the surface, so they take the
    # natural default from the class term and let the neighbourhood term decide.
    # That is the right answer in both directions: a shadow between two houses
    # reads as city, a shadow in a wadi reads as bare.
    "Shadow": 0.0, "Unclassified": 0.0,
}

# THE LOAD-BEARING CONSTANT OF THIS PRODUCT. The class term alone is a recolour
# of the label map; everything that makes this a *fabric* score rather than a
# per-pixel lookup happens inside this window.
#   Too small (~10 m): the window sits inside one courtyard or one roof and the
#   density term just reproduces the class term -- an isolated shed reads as
#   dense urban again.
#   Too large (~500 m): a village smears into the desert around it and the
#   product degrades into exactly the coarse population grid it exists to beat.
# 50 m is roughly one built block plus its street -- the scale at which "this is
# a settlement" first becomes true. Where this should come from: the actual
# block-size distribution of the settlements in the AOI, which is measurable
# from OSM building footprints and has never been measured here.
FABRIC_WINDOW_M = 50.0
# How much of the final score the neighbourhood carries. At 0 an isolated shed
# is dense urban; at 1 the class under your feet stops mattering at all.
FABRIC_DENSITY_WEIGHT = 0.45
# Anthropogenic area fraction at which fabric is "as dense as it gets" -- the
# rest of a dense block being yards, gardens, shadow and vegetation. Guess.
FABRIC_SATURATION = 0.6
# Class-term score at or above which a pixel counts as anthropogenic for the
# density term. Set so that agriculture (0.35) does not count as settlement.
FABRIC_ANTHRO_MIN = 0.45

# --- change volatility -----------------------------------------------------
#
# How much this ground is EXPECTED to look different between two dates for
# reasons that are not change: season, phenology, illumination. It is the
# baseline a change detector has to beat before it is allowed to call a
# difference an event.
#
# UNVALIDATED. This exists to serve the change-detection work in the next
# session, and no pair of real dates has been measured to produce any of these
# numbers -- they are read off the taxonomy's own definitions.
VOLATILITY_DEFAULT = 0.05     # rock and masonry: if it differs, something happened
VOLATILITY = {
    # The same ground in two seasons. The taxonomy says so in its own words:
    # "a GreenGrassland/DryGrassland difference between two dates is phenology,
    # not change". Anything that flags this pair is flagging the calendar.
    "GreenGrassland": 0.9, "DryGrassland": 0.9,
    # Crop cycle: planted, grown, harvested, bare, planted. The label can walk a
    # long way inside one agricultural year without a single decision changing.
    "IrrigatedField": 0.7, "IrrigatedOrchard": 0.7, "UnirrigatedOrchard": 0.7,
    # The one that matters most and is the least obvious. Shadow is not a
    # surface -- it is an illumination artifact, and it MOVES between two
    # acquisitions because the sun did, at a different time of day or a
    # different month. Every pixel a shadow crosses reads as two different
    # classes on two dates while the ground under it did not change at all.
    # Shadow edges are where a naive change detector produces most of its
    # false positives, and the shadow raster is also the cheapest thing to
    # co-register, so this is the term worth getting right first.
    "Shadow": 0.85,
    # A parked car is gone next week. That IS a change of the object and is not
    # a change of the ground, which is what these products describe.
    "Car": 0.8,
    # The degradation series shifts over years, not seasons: batha does not
    # become maquis between two acquisitions. What does move is the segmenter's
    # boundary between adjacent stages, which is a classifier difference dressed
    # as a landscape one -- hence 0.35 rather than the near-zero the ecology
    # alone would justify.
    "Batha": 0.35, "Garigue": 0.35, "Maquis": 0.35,
    # Seasonal pools, reservoir drawdown, a wadi that runs three days a year.
    "Water": 0.5,
    # Waterlogged by its own definition -- gleyed in winter, cracked and pale in
    # summer, and green with a spring flush in between.
    "HydromorpicSoil": 0.5,
    # Bare soil is not masonry: a winter green flush relabels it GreenGrassland,
    # and ploughing changes its tone without changing what it is.
    "TerraRosa": 0.2, "ClayeySoil": 0.2, "Rendzina": 0.2, "ClayeyDeepSoil": 0.2,
    # An unsealed track re-grades, gets dusty, and greens along its verges.
    "DirtRoad": 0.15, "DirtRoadB": 0.15,
    # Built surfaces. If these differ between two dates, something actually
    # happened -- which is the whole point of keeping them near zero.
    "House": 0.05, "BrickWall": 0.05, "Pavement": 0.05, "PavedRoad": 0.03,
    # The segmenter is unsure here, so it may flip on its own between two runs
    # over ground that did not move. This is model volatility, not terrain
    # volatility, and it is in the same raster because a change detector has to
    # survive both.
    "Unclassified": 0.5, "Clutter": 0.5,
}

# --- OSM overlays ----------------------------------------------------------
#
# A product is a decision surface, and the decision does not care which map the
# evidence came from -- but a reader does, so every overlay is applied as a named
# term and `summarise` states how much of the AOI it moved.
#
# The strongest of these is the first: a street the segmenter could not see is
# still a street. On the aza AOI 28.6% of the ground under an OSM road corridor
# is labelled `Shadow` [measured], whose `traffic` is the 0.5 default -- so
# without this overlay a wheeled-vehicle map shows the densest street network in
# the AOI as mediocre going.

# Trafficability of a mapped road corridor, before slope. Not 1.0: OSM says a
# way exists and roughly where, not that it is passable today -- rubble, a
# checkpoint and a collapsed culvert are all invisible to it.
OSM_ROAD_TRAFFIC = 0.90
# ...and this much for a track/path grade, which is a weaker claim again.
OSM_TRACK_TRAFFIC = 0.70

# A mapped building or wall is impassable, and a mapped footprint is a stronger
# statement about obstruction than about anything else in this file.
OSM_BUILDING_TRAFFIC = 0.0

# Concealment from a mapped building: the same value the label-derived built
# term uses, so the two are commensurable.
OSM_BUILT_CONCEALMENT = BUILT_CONCEALMENT

# Built fabric: how far a mapped residential/industrial landuse polygon may lift
# the score on its own. Small on purpose. OSM landuse coverage is the most
# uneven layer we ingest -- dense in one town, absent in the next valley -- so a
# strong term here would draw the edges of OSM's survey effort onto the map and
# call them settlement boundaries. It corroborates the raster; it never carries
# the answer.
OSM_BUILT_LANDUSE_FABRIC = 0.15

# Change volatility: inside a mapped building footprint, a pixel the segmenter
# could not name is a building, not an unknown -- so it takes the built (low)
# volatility instead of the 0.5 the segmenter's own uncertainty would earn it.
OSM_BUILT_VOLATILITY = 0.05


def _slope(raster: LabelRaster) -> np.ndarray:
    if raster.dem is None:
        return np.zeros(raster.shape, dtype=np.float32)
    gy, gx = np.gradient(raster.dem.astype(np.float64), raster.gsd)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _per_class(attr) -> np.ndarray:
    return np.array([attr(BY_ID[c]) for c in range(N_CLASSES)], dtype=np.float32)


def _mask_of(labels: np.ndarray, names) -> np.ndarray:
    return np.isin(labels, [cid(n) for n in names])


def _class_table(values: dict[str, float], default: float) -> np.ndarray:
    """Per-class lookup array from a name -> value dict, `default` elsewhere.

    Names go through `cid`, so a typo in one of the tables above is a KeyError
    at import rather than a class silently taking the default for ever.
    """
    table = np.full(N_CLASSES, default, dtype=np.float32)
    for name, value in values.items():
        table[cid(name)] = value
    return table


def _odd_window_px(metres: float, gsd: float) -> int:
    """Window size in pixels for a length in metres. Odd, so it is centred."""
    return max(3, int(round(metres / gsd)) | 1)


def trafficability(raster: LabelRaster, vehicle: str = "wheeled",
                   wet: bool = False, osm=None) -> np.ndarray:
    v = VEHICLES[vehicle]
    surface = _per_class(lambda d: d.traffic)[raster.labels]
    slope = _slope(raster)

    slope_term = np.clip(
        1.0 - (slope - SLOPE_FREE_DEG) / max(v.max_slope_deg - SLOPE_FREE_DEG, 1e-6),
        0.0, 1.0,
    )
    score = surface * slope_term
    score[surface < v.min_surface] = 0.0
    if wet:
        score = np.where(_mask_of(raster.labels, WET_CLASSES),
                         score * (1.0 - v.wet_sensitivity), score)

    if osm is not None and osm.burned.has("road"):
        # The road network wins where it exists: `max`, not a blend, because the
        # question "can a vehicle get down this street" is answered by the street
        # being there, and the label under it (Shadow, rubble, a badlands
        # misread) is what this overlay exists to correct. Slope still applies --
        # a mapped road up a 40 deg face is a mapped road a tank cannot climb.
        corridor = osm.burned.mask("road")
        base = np.where(_track_only(osm), OSM_TRACK_TRAFFIC, OSM_ROAD_TRAFFIC)
        road_score = np.float32(base) * slope_term
        score = np.where(corridor, np.maximum(score, road_score), score)
    if osm is not None:
        for name, val in (("building", OSM_BUILDING_TRAFFIC),
                          ("barrier", OSM_BUILDING_TRAFFIC)):
            if osm.burned.has(name):
                score = np.where(osm.burned.mask(name), val, score)
    return score.astype(np.float32)


def _track_only(osm) -> bool:
    """True when every mapped road in this AOI is a track or a path.

    One flag rather than a per-pixel grade raster: an int8 grade layer over the
    sinai mosaic is 1.2 GB to distinguish two constants, and an AOI whose entire
    network is tracks is the case that actually matters (open desert, where
    treating a goat path as a road is how a trafficability map lies)."""
    grades = {fp.grade for fp in osm.burned.by_kind("road")}
    return bool(grades) and grades <= {"track", "path", "footway", "bridleway"}


def _local_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Local standard deviation over a size x size window.

    Vectorised as sqrt(E[x^2] - E[x]^2) with two uniform_filter passes.
    Numerically equivalent -- edges included, both replicate the border via
    mode="nearest" -- to `ndi.generic_filter(arr, np.std, size, mode="nearest")`,
    which invokes a Python callback per pixel and takes hours at mosaic scale.
    """
    a = arr.astype(np.float64)      # float32 E[x^2]-E[x]^2 cancels catastrophically
    m = ndi.uniform_filter(a, size=size, mode="nearest")
    m2 = ndi.uniform_filter(a * a, size=size, mode="nearest")
    # Rounding can push the variance a hair below zero on flat ground; sqrt of
    # that is NaN, so clamp.
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _target_cover(mask: np.ndarray, side_px: int) -> np.ndarray:
    """Fraction of a target's own footprint that falls inside `mask`.

    A concealing feature only conceals if it persists across the whole thing
    being concealed: a 3 m pool of shadow hides a man and does not hide a 10 m
    building, however dark it is. One uniform filter at the target's own scale
    says that for every feature in the product, which is why shadow, built
    adjacency and mapped footprints all go through here.
    """
    if side_px <= 1:
        return mask.astype(np.float32)
    return ndi.uniform_filter(mask.astype(np.float32), size=side_px, mode="nearest")


def concealment(raster: LabelRaster, target: str = "person",
                osm=None) -> np.ndarray:
    """How well is a target of a NAMED SIZE hidden from an overhead observer.

    `target` is not decoration. Every concealment constant in this file was
    quoted for something person-sized, and the same maquis canopy that hides a
    crouching man does not hide a truck. The size is the caller's to state.
    """
    try:
        t = TARGETS[target]
    except KeyError:
        raise ValueError(
            f"unknown concealment target {target!r}; valid targets are: "
            f"{', '.join(sorted(TARGETS))}"
        ) from None

    # The target's own footprint as a square, in pixels. Everything below is
    # measured against this length.
    side_px = max(1, int(round(t.footprint_m2 ** 0.5 / raster.gsd)))

    # Canopy: fractional cover, so the bigger the target the closer to complete
    # that cover has to be. See CANOPY_CLUMP_M2 for the model and its limits.
    # At the person target the exponent is ~1.1, i.e. essentially the old
    # behaviour -- this generalises the previous product, it does not replace it.
    canopy = _per_class(lambda d: d.canopy)[raster.labels]
    exponent = min(1.0 + t.footprint_m2 / CANOPY_CLUMP_M2, CANOPY_EXPONENT_MAX)
    score = np.power(canopy, exponent, dtype=np.float32)

    shadow = _mask_of(raster.labels, ("Shadow",))
    score = np.maximum(score, SHADOW_CONCEALMENT * _target_cover(shadow, side_px))

    built = _mask_of(raster.labels, ("House", "BrickWall"))
    near_built = ndi.binary_dilation(built, np.ones((9, 9))) & ~built
    score = np.maximum(score, BUILT_CONCEALMENT * _target_cover(near_built, side_px))

    if osm is not None and osm.burned.has("building"):
        # Beside a mapped building, not on it: the score is "how well is a ground
        # object hidden", and a roof is not ground.
        b = osm.burned.mask("building")
        near = ndi.binary_dilation(b, np.ones((9, 9))) & ~b
        score = np.maximum(score, OSM_BUILT_CONCEALMENT * _target_cover(near, side_px))

    # Relief. Two target-dependent things happen here. The roughness window
    # widens to the target's footprint -- ground has to be broken at the scale of
    # the thing it is hiding, and 5 px of micro-relief is nothing to a truck --
    # and the term is gated by whether the local relief amplitude actually
    # exceeds the target's height. A 0.4 m ripple field hides nothing that stands
    # 2 m tall, however rough it scores.
    slope = _slope(raster)
    window_px = max(ROUGHNESS_WINDOW_PX, side_px)
    roughness = _local_std(slope, window_px) if slope.any() else np.zeros_like(slope)
    if roughness.max() > 0:
        relief = roughness / roughness.max()
        if raster.dem is not None:
            amplitude_m = _local_std(raster.dem.astype(np.float64), window_px)
            relief = relief * np.clip(amplitude_m / t.height_m, 0.0, 1.0)
        score = score + RELIEF_CONCEALMENT_WEIGHT * relief
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def built_fabric(raster: LabelRaster, osm=None) -> np.ndarray:
    """What kind of ground this is, bare-natural (0) -> dense-urban (1).

    The honest per-tile answer to "how populated is this". It is not a
    population count and must never be reported as one -- it says what the
    ground looks like, at the GSD the ground was actually seen at, which is the
    one thing a 100 m population raster cannot say: that grid is coarser than a
    city block, so it cannot tell a courtyard from a field, and it counts people
    where they are registered rather than where the buildings are.

    Two terms. The per-class score says what is under this pixel; the smoothed
    anthropogenic fraction says what kind of place this pixel is in. They are
    BLENDED, not summed, and that is the whole design: a sum cannot fix the case
    the product exists for -- an isolated shed in open desert scores 0.9 from its
    class and stays there no matter what is added to it. Blended, the shed lands
    mid-scale (a real building in empty ground, which is what it is) and the bare
    courtyard inside a city block lands mid-scale from the other direction.
    Both are honestly "in between", and neither pretends to be its own extreme.
    """
    base = _class_table(FABRIC_BASE, FABRIC_BASE_DEFAULT)[raster.labels]

    anthropogenic = (base >= FABRIC_ANTHRO_MIN).astype(np.float32)
    if osm is not None and osm.burned.has("building"):
        # A mapped footprint is a building even where the segmenter called it
        # Shadow, so it joins the density term before the smoothing. This is
        # corroboration of the same claim the class term makes, not a new one.
        anthropogenic = np.maximum(anthropogenic,
                                   osm.burned.mask("building").astype(np.float32))

    window_px = _odd_window_px(FABRIC_WINDOW_M, raster.gsd)
    density = ndi.uniform_filter(anthropogenic, size=window_px, mode="nearest")
    density = np.clip(density / FABRIC_SATURATION, 0.0, 1.0)

    score = ((1.0 - FABRIC_DENSITY_WEIGHT) * base
             + FABRIC_DENSITY_WEIGHT * density)

    if osm is not None and osm.burned.has("built_landuse"):
        # Mild and additive, never decisive: see OSM_BUILT_LANDUSE_FABRIC. A
        # residential polygon agrees that this is a settlement; it does not know
        # which pixels inside it are roof and which are the wadi behind the
        # houses, and the raster does.
        score = score + OSM_BUILT_LANDUSE_FABRIC * osm.burned.mask("built_landuse")
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def change_volatility(raster: LabelRaster, osm=None) -> np.ndarray:
    """How much this ground is EXPECTED to differ between two dates anyway.

    Season, phenology and illumination move the label on their own. This is the
    baseline a change detector has to beat before it is entitled to call a
    difference an event: on 0.9 ground (grassland between wet and dry season, a
    shadow that walked with the sun) a label difference is the null hypothesis,
    and on 0.05 ground (masonry, seal, bedrock) the same difference is a report.

    Purely per-class -- no imagery, no dates, no pair. It is a prior over the
    taxonomy, and it is infrastructure for the change-detection work rather than
    a product anyone should be shown on its own. UNVALIDATED: every number comes
    from reading the class definitions, not from measuring a pair of dates.
    """
    out = _class_table(VOLATILITY, VOLATILITY_DEFAULT)[raster.labels]
    if osm is not None and osm.burned.has("building"):
        # Narrow on purpose. Only the classes whose 0.5 is the segmenter's own
        # uncertainty get resolved by a mapped footprint. Shadow keeps its 0.85
        # even on a building, because the shadow of that building is exactly the
        # thing that moves between two acquisitions.
        unsure = _mask_of(raster.labels, ("Unclassified", "Clutter"))
        out = np.where(unsure & osm.burned.mask("building"),
                       OSM_BUILT_VOLATILITY, out)
    return out.astype(np.float32)


PRODUCTS = {
    "trafficability": trafficability,
    "concealment": concealment,
    "built_fabric": built_fabric,
    "change_volatility": change_volatility,
}


def compute(raster: LabelRaster, product: str, osm=None, **kw) -> np.ndarray:
    arr = PRODUCTS[product](raster, osm=osm, **kw)
    if raster.valid is not None:
        # Nodata carries class id 0 (Unclassified), whose traffic score is 0.5.
        # Left alone, a third of an arid crop scores as moderately driveable open
        # ground, and S5's corridor query drives straight across it.
        #
        # Zero is the safe reading for three of the four products (impassable,
        # unconcealed, bare). For `change_volatility` it inverts: zero there
        # means "any difference here is real", which is the least safe thing to
        # say about ground nobody has data for. A change detector consuming this
        # raster must mask on `raster.valid` itself and not lean on this line.
        arr = np.where(raster.valid, arr, 0.0).astype(np.float32)
    return arr


def osm_delta(raster: LabelRaster, product: str, osm, **kw) -> str:
    """What the overlay changed. A product that quietly reads a different map
    than the one its caller thinks it read is a wrong answer with a plausible
    number attached, so this is printed whenever an overlay is applied."""
    if osm is None:
        return ""
    return osm_delta_from(raster, compute(raster, product, osm=None, **kw),
                          compute(raster, product, osm=osm, **kw), osm)


def osm_delta_from(raster: LabelRaster, before: np.ndarray, after: np.ndarray,
                   osm) -> str:
    """Same report, from arrays the caller already has.

    `osm_delta` computes the product twice, which is the honest thing for a
    one-shot CLI call and pure waste for a caller that already holds both -- and
    at four products that was eight full-raster passes instead of four. The
    playground renders both rasters anyway, so it goes through here.
    """
    if osm is None:
        return ""
    ok = raster.valid
    diff = np.abs(after - before)
    moved = diff > 1e-6
    if ok is not None:
        moved &= ok
    denom = max(raster.n_valid, 1)
    lines = [
        f"# OSM overlay: {moved.sum() / denom:.2%} of classified pixels moved, "
        f"mean {before[moved].mean() if moved.any() else 0:.2f} -> "
        f"{after[moved].mean() if moved.any() else 0:.2f} on those pixels "
        f"(AOI mean {before[ok].mean() if ok is not None else before.mean():.3f} "
        f"-> {after[ok].mean() if ok is not None else after.mean():.3f})",
    ]
    for name in ("road", "building", "barrier", "built_landuse"):
        if osm.burned.has(name):
            m = osm.burned.mask(name)
            if ok is not None:
                m = m & ok
            if not m.any():
                continue
            lines.append(f"#   {name:13} {m.sum() / denom:6.2%} of the AOI, "
                         f"{before[m].mean():.2f} -> {after[m].mean():.2f}")
    lines.append("# every figure below is the WITH-overlay product; the classes "
                 "under a mapped road or footprint are no longer what drives it "
                 "there")
    return "\n".join(lines)


def summarise(raster: LabelRaster, arr: np.ndarray, ridx: RegionIndex | None = None,
              top_k: int = 12) -> str:
    """Per-superclass means, plus the worst/best regions if an index is given."""
    ok = raster.valid
    vals = arr if ok is None else arr[ok]
    lines = [
        f"# value distribution over classified pixels: mean {vals.mean():.2f}, "
        f"p10 {np.percentile(vals, 10):.2f}, p90 {np.percentile(vals, 90):.2f}",
        "",
        "## mean by superclass",
        "superclass\tmean\tarea_frac",
    ]
    sc_of = np.array([SUPERCLASS_OF[c] for c in range(N_CLASSES)])
    per_px = sc_of[raster.labels]
    denom = max(raster.n_valid, 1)
    for sc in sorted(set(SUPERCLASS_OF.values())):
        m = per_px == sc
        if ok is not None:
            m &= ok
        if not m.any():
            continue
        lines.append(f"{sc}\t{arr[m].mean():.2f}\t{m.sum() / denom:.3f}")

    if ridx is not None:
        lines += ["", f"## top {top_k} regions by mean value", "rid\tclass\tarea_m2\tmean"]
        vals = ndi.mean(arr, ridx.label_array,
                        index=[r.id for r in ridx.regions]) if ridx.regions else []
        pairs = sorted(zip(ridx.regions, vals), key=lambda p: -p[1])[:top_k]
        for r, v in pairs:
            lines.append(f"{r.id}\t{r.class_name}\t{r.area_m2:.0f}\t{float(v):.2f}")
    return "\n".join(lines)


def to_png(arr: np.ndarray, path: str) -> None:
    """Viridis-ish ramp without pulling in matplotlib."""
    from PIL import Image

    x = np.clip(arr, 0, 1)
    rgb = np.stack([
        (np.clip(1.6 * x - 0.35, 0, 1) * 255),
        (np.clip(1.2 * x + 0.05, 0, 1) * 255),
        (np.clip(1.1 - 1.3 * x, 0, 1) * 255),
    ], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(path)
