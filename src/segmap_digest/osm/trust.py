"""Which map wins, and about what. The owner's decision, made explicit.

> *"OSM is a good way to be a reference for the land, but the ST output is
> considered latest unless I say other."* -- owner, 2026-08-26

That is not a single trust ranking, and reading it as one would get every check
in this package wrong in one direction or the other. It is **two rankings on two
different questions**, and every disagreement between the two maps has to be
sorted onto the right one before it can be resolved:

    EXISTENCE  -- is there a street / a building / water here at all?
    IDENTITY   -- what is it: which grade, which surface, what is it called?
                  ==> OSM IS THE REFERENCE. It is the base map of the land. A
                      segmenter that sees no road under a mapped street has not
                      disproved the street.

    STATE      -- what does that ground look like NOW?
                  ==> ST IS THE LATEST. A footprint in OSM is a statement that a
                      building was there when somebody mapped it. If ST shows
                      rubble, the rubble is the newer fact, and "switch this
                      label to House because OSM has a footprint" would be RT
                      overwriting an observation with a memory.

The practical consequences, in one table, are `AXIS` below. Two of them are worth
stating in words because they invert what the first implementation did:

  A mapped building over a polygon ST calls `MaralBadlands` is **not** strong
    evidence that the polygon is a House. Under this policy it is a CHANGE
    CANDIDATE -- exactly the signal you want out of an AOI where buildings stop
    existing -- and S2 must not flip the label on it.
  An ST road with no OSM way under it is **not** an ST false positive by
    default. ST is the later map; the first reading is that the road is new or
    was never mapped, which makes it an OSM update candidate.

Reading the disagreement the other way round is still possible and sometimes
right -- OSM's geometry can be wrong, ST's label can be wrong. That is why every
finding names both maps and neither is called an error.
"""

from __future__ import annotations

from dataclasses import dataclass

AXIS_EXISTENCE = "existence"
AXIS_IDENTITY = "identity"
AXIS_STATE = "state"

REFERENCE_MAP = "OSM"
LATEST_MAP = "ST"

# Who is authoritative on each axis. Flip these two lines and the whole package
# changes its mind coherently -- that is the point of having the table.
AUTHORITY = {
    AXIS_EXISTENCE: REFERENCE_MAP,
    AXIS_IDENTITY: REFERENCE_MAP,
    AXIS_STATE: LATEST_MAP,
}


@dataclass(frozen=True)
class Resolution:
    axis: str
    authority: str
    reading: str          # the short tag that goes in front of every message
    text: str             # the sentence that goes in the rolled-up cause


# What each check family is actually disagreeing about, and therefore how to
# read it. The `reading` tag is what a reviewer sorts on:
#
#   reference-gap     the base map has a feature the latest observation does not
#                     show -- occlusion, change, or a miss, in that order of
#                     likelihood
#   update-candidate  the latest observation has something the base map lacks --
#                     the base map is probably behind
#   state-change      both maps agree something is there and disagree about what
#                     it currently is -- ST wins on state, and the disagreement
#                     is the finding
#   identity-conflict both maps agree something is there and disagree about what
#                     KIND it is -- OSM is the reference here
AXIS: dict[str, Resolution] = {
    "osm-road-missing": Resolution(
        AXIS_EXISTENCE, REFERENCE_MAP, "reference-gap",
        "The reference map (OSM) has a street corridor here; the latest "
        "observation (ST) shows no road surface under it. The corridor is taken "
        "as real -- read this as a change in what the surface looks like, or as "
        "a missed label, before reading it as a street that is not there."),
    "osm-road-occluded": Resolution(
        AXIS_EXISTENCE, REFERENCE_MAP, "reference-gap",
        "The reference map has a street here and ST could not see the ground "
        "(Shadow/Unclassified). A visibility limit, not a disagreement: neither "
        "map is contradicted."),
    "osm-road-grade": Resolution(
        AXIS_STATE, LATEST_MAP, "state-change",
        "Both maps agree there is a road; they disagree about its surface. ST is "
        "the later observation and its grade stands as the current state -- so "
        "the first reading is that OSM's `surface` tag is stale, and the second "
        "is that ST misread the texture."),
    "osm-road-extra": Resolution(
        AXIS_EXISTENCE, LATEST_MAP, "update-candidate",
        "ST maps a road the reference map does not have. ST is the later "
        "observation, so the first reading is a new or never-mapped track -- an "
        "OSM edit -- and only then an ST false positive."),
    "osm-building-missing": Resolution(
        AXIS_STATE, LATEST_MAP, "state-change",
        "The reference map has a building footprint; the latest observation does "
        "not show a building on it. ST is the newer fact, so this is read as "
        "change on the ground first, and as a missed label second. It is NOT "
        "grounds for relabelling the polygon House."),
    "osm-building-extra": Resolution(
        AXIS_EXISTENCE, LATEST_MAP, "update-candidate",
        "ST maps a building the reference map does not have: new construction, "
        "or a footprint nobody has mapped."),
    "osm-water-extra": Resolution(
        AXIS_STATE, LATEST_MAP, "update-candidate",
        "ST maps water the reference map does not have. Seasonal water is "
        "systematically absent from OSM, so this finds phenology at least as "
        "readily as it finds errors."),
}

DEFAULT = Resolution(AXIS_STATE, LATEST_MAP, "disagreement",
                     "The two maps disagree and this check family has not "
                     "declared which axis it is on.")


def resolve(kind: str) -> Resolution:
    return AXIS.get(kind, DEFAULT)


# --- how much a disagreement is worth, by axis -----------------------------
#
# A finding on an axis where OSM is the authority is a statement about the
# LABEL and belongs high in a review worklist. A finding on an axis where ST is
# the authority is a statement about the WORLD -- something changed -- and it is
# just as interesting, but it is not a label defect and must not be presented as
# one. Same rank, different sentence.
SEVERITY_SCALE = {
    AXIS_EXISTENCE: 1.00,
    AXIS_IDENTITY: 1.00,
    AXIS_STATE: 0.85,
}


# --- what a candidate label gets from the reference map, by axis -----------
#
# S2's reference term, before reliability blending. Neutral is 0.5.
#
# The identity numbers are the strong ones: OSM saying "this way is tagged
# surface=asphalt" is the closest thing to ground truth in this whole pipeline
# for the PavedRoad/DirtRoad split, which is otherwise a judgement about texture.
#
# The state numbers are deliberately timid, and that is the owner's decision
# expressed as arithmetic: 0.62 support cannot clear SWITCH_MARGIN on its own
# even before reliability is applied, so a mapped footprint can put House on the
# shortlist and argue for it, and cannot overturn what ST currently sees.
SUPPORT = {
    AXIS_EXISTENCE: 0.80,
    AXIS_IDENTITY: 0.85,
    AXIS_STATE: 0.62,
}
CONTRADICT = {
    AXIS_EXISTENCE: 0.20,
    AXIS_IDENTITY: 0.15,
    AXIS_STATE: 0.40,
}


def support(axis: str) -> float:
    return SUPPORT.get(axis, 0.6)


def contradict(axis: str) -> float:
    return CONTRADICT.get(axis, 0.4)


def policy_note() -> str:
    """One paragraph, printed wherever reference findings are shown."""
    return (
        "# trust policy: OSM is the REFERENCE for the land (where streets and "
        "buildings are, what they are called); ST is the LATEST observation of "
        "what the ground looks like now. Disagreements about existence and "
        "identity resolve towards OSM; disagreements about current state "
        "resolve towards ST, and are reported as change rather than as label "
        "error. Neither map is called wrong. See osm/trust.py."
    )
