"""OSM tags -> what they mean for a 47-class Smart Terrain map.

One table per question, all of them in this file, because the alternative is the
same `highway in ("residential", "service")` test written slightly differently
in five modules.

Three of these tables are the whole reason the layer is worth building:

  `SURFACE_EXPECTS` turns OSM's `surface=*` into an expectation about ST's
    ROAD_SERIES. That is a reference map for the paved/dirt distinction, which
    is otherwise a judgement call about texture -- exactly the role the
    geological map plays for Limestone/Dolomite/Nari.
  `HALF_WIDTH_M` decides how wide the road corridor is, which decides where the
    blocks are. Get it wrong and either the corridor swallows the block or the
    block leaks through the road.
  `RELIABILITY` is the honest half: OSM is a volunteered map, its completeness
    varies by tag and by country, and a disagreement between two maps of unknown
    completeness is not evidence that ST is wrong.

EVERY NUMBER IN THIS FILE IS A GUESS unless its comment says otherwise. They are
grouped so they can be swept as a set, and the docs table in `docs/OSM.md` names
where each should actually come from.
"""

from __future__ import annotations

# --- what counts as a road -------------------------------------------------
#
# Ordinal, coarse-to-fine, because a corridor's grade is what sets its width and
# what ST class it should carry. `motorway` down to `service` is the vehicle
# network; `path`/`footway`/`steps` are not, and burning them as road corridors
# fragments a block partition into confetti -- on the aza AOI, paths and
# footways are 35 of 404 ways and every one of them is an alley too narrow for
# ST to resolve as a road at all.

VEHICLE_GRADES: tuple[str, ...] = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service", "track", "road",
)
# `*_link` ramps carry the grade of the road they serve.
LINK_SUFFIX = "_link"

FOOT_GRADES: tuple[str, ...] = (
    "footway", "path", "pedestrian", "steps", "bridleway", "cycleway", "corridor",
)

# Grades that exist in the data but are not ground you can drive on now.
NON_TRAFFIC_GRADES: tuple[str, ...] = ("construction", "proposed", "raceway", "escape")


def road_grade(tags: dict) -> str | None:
    """The `highway` value, with `*_link` folded onto its parent. None = not a road."""
    v = tags.get("highway")
    if not v:
        return None
    if v.endswith(LINK_SUFFIX):
        v = v[: -len(LINK_SUFFIX)]
    return v


def is_vehicle_road(tags: dict) -> bool:
    return road_grade(tags) in VEHICLE_GRADES


# --- corridor width --------------------------------------------------------
#
# HALF width in metres, i.e. how far the surface extends either side of the
# centreline. OSM ways are centrelines; ST sees surfaces. Precedence, highest
# first, is `width` tag -> `lanes` tag -> this table, because the first two are
# statements about *this* way and the table is a statement about its grade.
#
# WHERE THIS SHOULD COME FROM: the road-design standard in force in the AOI, or
# a measurement of ST's own road polygons per grade -- which this layer can
# produce (`segmap osm align --by-grade`) once anyone trusts it enough to
# calibrate against.

HALF_WIDTH_M: dict[str, float] = {
    "motorway": 11.0,
    "trunk": 8.0,
    "primary": 7.0,
    "secondary": 6.0,
    "tertiary": 5.0,
    "unclassified": 4.0,
    "residential": 4.0,
    "living_street": 3.5,
    "service": 3.0,
    "track": 2.5,
    "road": 4.0,
    # foot grades, only burned when the caller asks for them
    "pedestrian": 2.5,
    "footway": 1.0,
    "path": 1.0,
    "steps": 1.0,
    "cycleway": 1.0,
    "bridleway": 1.0,
    "corridor": 1.0,
}
HALF_WIDTH_DEFAULT = 3.0

# Metres per lane, for the `lanes` tag. 3.5 m is the common design figure for a
# through lane; narrower in dense old-city fabric, which is exactly where the
# `lanes` tag is least likely to be present.
LANE_WIDTH_M = 3.5

# A corridor narrower than this is not a thing ST could have resolved as a road
# at 0.5 m/px, so a "ST missed this road" finding over one is noise.
MIN_CREDIBLE_HALF_WIDTH_M = 1.0


def half_width_m(tags: dict, default: float = HALF_WIDTH_DEFAULT) -> float:
    """Half the surface width of this way, in metres.

    `width` wins because it is a measurement of this way. `lanes` is a floor
    rather than the answer: a 4-lane residential street is at least 4*3.5/2 wide,
    but a `width` tag saying 6 m on the same way is the better number.
    """
    grade = road_grade(tags)
    hw = HALF_WIDTH_M.get(grade, default) if grade else default

    raw = tags.get("width") or tags.get("est_width")
    if raw is not None:
        try:
            # "6", "6 m", "6.5m", "20'" -- take the leading number, metres only.
            txt = str(raw).strip().replace("m", " ").split()[0]
            w = float(txt)
            if 0.5 <= w <= 60.0:               # anything outside is a tagging error
                return max(w / 2.0, MIN_CREDIBLE_HALF_WIDTH_M)
        except (ValueError, IndexError):
            pass

    lanes = tags.get("lanes")
    if lanes is not None:
        try:
            n = float(str(lanes).split(";")[0])
            if 1.0 <= n <= 12.0:
                hw = max(hw, n * LANE_WIDTH_M / 2.0)
        except ValueError:
            pass
    return max(hw, MIN_CREDIBLE_HALF_WIDTH_M)


# --- surface -> the ST road class this way should carry ---------------------
#
# The one place OSM is a genuine reference map for a distinction ST has to
# guess. `PavedRoad` / `DirtRoad` / `DirtRoadB` is `taxonomy.ROAD_SERIES`, so a
# mismatch is graded by `class_distance` and a Paved/DirtB disagreement is worse
# than a Paved/Dirt one.
#
# `DirtRoadB` is the secondary grade, so an unpaved *track* maps to it and an
# unpaved *road* maps to DirtRoad; that split is a guess about what the
# segmenter's annotators meant, and it is the shakiest entry in this file.

SURFACE_EXPECTS: dict[str, str] = {
    # sealed
    "paved": "PavedRoad",
    "asphalt": "PavedRoad",
    "concrete": "PavedRoad",
    "concrete:plates": "PavedRoad",
    "concrete:lanes": "PavedRoad",
    "chipseal": "PavedRoad",
    "paving_stones": "PavedRoad",
    "sett": "PavedRoad",
    "cobblestone": "PavedRoad",
    "metal": "PavedRoad",
    # `interlock` is not in the OSM wiki's list but is the commonest surface
    # value on the aza AOI's residential streets [measured]: interlocking
    # concrete pavers, which is a sealed surface.
    "interlock": "PavedRoad",
    "interlocking": "PavedRoad",
    "bricks": "PavedRoad",
    "brick": "PavedRoad",
    "unhewn_cobblestone": "PavedRoad",
    "wood": "PavedRoad",
    # graded but unsealed
    "compacted": "DirtRoad",
    "gravel": "DirtRoad",
    "fine_gravel": "DirtRoad",
    "unpaved": "DirtRoad",
    "dirt": "DirtRoad",
    "earth": "DirtRoad",
    "ground": "DirtRoad",
    "mud": "DirtRoad",
    "sand": "DirtRoadB",
    "grass": "DirtRoadB",
    "rock": "DirtRoadB",
    "pebblestone": "DirtRoadB",
    "woodchips": "DirtRoadB",
}

# When `surface` is absent -- 88% of the aza ways [measured] -- the grade itself
# is a weak prior. Weak enough that it must not drive a finding on its own; it
# is here so the *expectation* is stated rather than invented at the call site.
GRADE_EXPECTS: dict[str, str] = {
    "motorway": "PavedRoad",
    "trunk": "PavedRoad",
    "primary": "PavedRoad",
    "secondary": "PavedRoad",
    "tertiary": "PavedRoad",
    "residential": "PavedRoad",
    "living_street": "PavedRoad",
    "unclassified": "DirtRoad",
    "service": "PavedRoad",
    "track": "DirtRoad",
}

ROAD_CLASSES: tuple[str, ...] = ("PavedRoad", "DirtRoad", "DirtRoadB")


def expected_road_class(tags: dict) -> tuple[str | None, str]:
    """(ST class this way should carry, why). `(None, reason)` = no expectation.

    Returns the *source of the expectation* alongside it, because a finding that
    cites `surface=asphalt` is actionable and one that cites "residential roads
    are usually paved" is a prior with an unmeasured error rate.
    """
    surf = tags.get("surface")
    if surf:
        key = str(surf).split(";")[0].strip().lower()
        if key in SURFACE_EXPECTS:
            return SURFACE_EXPECTS[key], f"surface={key}"
        return None, f"surface={key} not in the mapping table"
    grade = road_grade(tags)
    if grade in GRADE_EXPECTS:
        return GRADE_EXPECTS[grade], f"highway={grade}, no surface tag (weak)"
    return None, "no surface tag and no grade expectation"


# --- everything else RT has a class for ------------------------------------
#
# Only features ST could plausibly have labelled. OSM knows about a great deal
# that the 47 classes have no word for; that is `Clutter`'s job, not this table's.

def is_building(tags: dict) -> bool:
    v = tags.get("building")
    return bool(v) and v not in ("no", "roof")           # a roof has no footprint below


def is_water(tags: dict) -> bool:
    return bool(tags.get("waterway")) or tags.get("natural") == "water" \
        or tags.get("landuse") in ("reservoir", "basin") \
        or tags.get("water") is not None


def is_barrier(tags: dict) -> bool:
    """Something that stops a vehicle and blocks line of sight."""
    return tags.get("barrier") in ("wall", "fence", "retaining_wall", "city_wall",
                                   "hedge", "guard_rail", "jersey_barrier")


# Waterways that are a *line* of flow rather than a body of water. `drainage`
# cares about these; `Water` (the class) mostly does not, since a 2 m wadi is
# below what ST resolves as open water.
FLOW_WATERWAYS: tuple[str, ...] = ("stream", "ditch", "drain", "river", "canal", "wadi")

# Landuse polygons that say "this is built-up ground" -- context for S2 and a
# triage prior for S3, never a class expectation on its own.
BUILT_LANDUSE: tuple[str, ...] = (
    "residential", "industrial", "commercial", "retail", "construction",
    "military", "garages", "cemetery",
)
AGRI_LANDUSE: tuple[str, ...] = (
    "farmland", "orchard", "vineyard", "greenhouse_horticulture", "plant_nursery",
    "meadow", "farmyard",
)


def feature_kind(tags: dict) -> str:
    """One coarse kind per way, for provenance and for the reliability table.

    Order matters: a `building` with a `barrier` tag is a building.
    """
    if is_building(tags):
        return "building"
    if is_water(tags):
        return "water"
    if road_grade(tags) in FOOT_GRADES:
        return "footway"
    if is_vehicle_road(tags):
        return "road"
    if road_grade(tags):
        return "other_highway"
    if is_barrier(tags):
        return "barrier"
    if tags.get("landuse") in BUILT_LANDUSE:
        return "built_landuse"
    if tags.get("landuse") in AGRI_LANDUSE:
        return "agri_landuse"
    return "other"


# --- how much to believe OSM ----------------------------------------------
#
# 0 = ignore, 1 = treat as truth. Used to scale S2's reference evidence and to
# grade S1's disagreement severity. These are NOT accuracy figures -- nobody has
# measured OSM against these AOIs -- they are a statement of relative confidence
# in *completeness*, which is the failure mode that matters here: a missing way
# manufactures a "ST invented a road" finding, and there is no way to tell that
# from the data alone.
#
# WHERE THIS SHOULD COME FROM: a completeness study on a sample of the AOI --
# take 30 blocks, have someone compare OSM against the imagery, and count. That
# is a day of work and it converts this whole table from a guess into a
# measurement. Until then the defaults are deliberately timid.

RELIABILITY: dict[str, float] = {
    "road": 0.75,             # the vehicle network is the best-mapped thing in OSM
    "building": 0.55,         # good in cities, absent in villages, stale after conflict
    "water": 0.50,
    "barrier": 0.30,          # sparsely mapped almost everywhere
    "footway": 0.35,
    "built_landuse": 0.45,
    "agri_landuse": 0.40,
    "other_highway": 0.40,
    "other": 0.20,
}
RELIABILITY_DEFAULT = 0.30

# `surface` is a tag on a way that IS mapped, so it does not carry the
# completeness problem -- only a staleness one. Believed more than the geometry
# it hangs off.
SURFACE_RELIABILITY = 0.80
