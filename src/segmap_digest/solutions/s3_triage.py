"""S3 -- cost-aware detection triage.

Use the cheap segmentation as a prior to decide which chips get sent to an
expensive detector. This is the money solution.

The load-bearing architectural claim: the LLM is NOT in the per-chip loop. It
compiles a detection query into a small, cacheable, human-auditable policy --
once per query type. A deterministic scorer then applies that policy to
millions of chips for effectively zero marginal cost. Put an LLM call per chip
and you have replaced one expensive call with another and saved nothing.

Regimes, because the economics differ by an order of magnitude:
    R1  target IS a class        near-free; the mask is the answer
    R2  target is context-bound  strong prior from co-occurring classes
    R3  target invisible         hunt Clutter/Unclassified/Shadow + high entropy

A query is TWO separable things per unit of ground, and collapsing them is how
you get a confident wrong answer:

    applicability  could the thing asked about plausibly be here AT ALL?
    priority       given that it could, how much does this unit deserve spend?

    worth_a_look = applicability x priority

"Find a missing tree" is a real question over maquis and a category error over
a basalt boulder field. A unit with zero applicability scores zero however
interesting it looks -- and "the query does not apply to this ground" is
reported as a different answer from "nothing here", because they are different
answers and conflating them is the silent wrong answer this module exists to
avoid.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..index import Chip, ChipIndex
from ..taxonomy import (
    ANCHOR_CLASSES,
    BY_ID,
    CLASSES,
    DEGRADATION_SERIES,
    N_CLASSES,
    PRIORS_BY_SUBJECT,
    ROAD_SERIES,
    SUPERCLASS,
    SUPERCLASS_OF,
    cid,
    class_distance,
)

# --- tunable heuristics (all guesses -- see docs/solutions/S3.md) ----------

# A chip is hard-excluded when an excluded class covers more than this fraction.
EXCLUDE_DOMINANCE = 0.80

# Entropy weight by regime: R3 leans on interface-richness because it has no
# positive class signal to work with.
ENTROPY_WEIGHT = {"R1": 0.0, "R2": 0.15, "R3": 0.5}

# Fraction of rejected chips sent to the detector anyway, to get an unbiased
# recall estimate. Without this there is NO signal that you have over-pruned.
CONTROL_SET_FRACTION = 0.03

# Below this many control chips the recall estimate has confidence intervals
# too wide to act on (one chip cannot distinguish 95% recall from 60%), and
# render() must say so instead of presenting it as a measurement. Where this
# should come from: the CI width required on the recall estimate -- the same
# decision that should set CONTROL_SET_FRACTION.
CONTROL_MIN_INFORMATIVE = 30


# --- tunables for the query compiler and for applicability -----------------
# These decide WHERE A QUERY MEANS ANYTHING, which is a different question from
# how interesting a chip looks. Every one of them is a guess, and the control
# set is what will eventually falsify them.

# Woody-cover fraction (taxonomy `canopy`) at which a class counts as carrying
# trees. WHERE THIS SHOULD COME FROM: the class owner's answer to "what does a
# user mean by tree" -- Batha at 0.2 is a dwarf shrub, not a tree, and that
# judgement is the only thing setting this number.
TREE_MIN_CANOPY = 0.3

# Trafficability at which a surface is plausible ground for a vehicle. Same
# value as s5_query.CORRIDOR_MIN_TRAFFIC on purpose: one physical claim should
# not have two thresholds. WHERE THIS SHOULD COME FROM: vehicle mobility tables.
TRAFFICABLE_MIN = 0.45

# Applicability credit given to merely-trafficable ground for a vehicle query.
# It exists so an off-road vehicle query is not pruned to the road network --
# the taxonomy's own surviving prior is "a vehicle sits on something
# trafficable", not "a vehicle sits on a road".
TRAFFICABLE_HABITAT_CREDIT = 0.35

# How much co-occurrence CONTEXT counts toward applicability, relative to the
# target class itself being present. WHERE THIS SHOULD COME FROM: measured
# p(target present | context present, target class absent) per query family.
HABITAT_CONTEXT_WEIGHT = 0.6

# Priority weight a compiled policy gives a context class (vs 1.0 for a target).
# WHERE THIS SHOULD COME FROM: the same calibration loop that should replace
# every hand-written weight in BUILTIN.
CONTEXT_WEIGHT = 0.45

# Scale applied to (1 - class_distance) when a non-target class is a near-miss
# of a target -- a Garigue chip is partial evidence for a Maquis query, a
# BasaltBoulder chip is not. WHERE THIS SHOULD COME FROM: the segmenter's
# confusion matrix; class_distance is a stand-in for it.
NEAR_MISS_WEIGHT = 0.6

# Below this applicability the query is not merely unpromising here, it is
# MEANINGLESS here: the unit is dropped and that is stated as the reason.
# WHERE THIS SHOULD COME FROM: the hit rate inside dropped units, measured
# through the control set. If that rate is not ~0, this is set too high.
APPLICABILITY_MIN = 0.02

# Distance out to which adjacency to habitat still makes ground plausible: the
# bare patch 30 m inside a maquis edge is exactly where a missing tree is.
# WHERE THIS SHOULD COME FROM: the spatial scale of the target's context, which
# differs per query family (a car near a road, a felled tree near a canopy).
APPLICABILITY_NEAR_M = 120.0

# Adjacency to habitat is weaker evidence than being habitat. This caps what
# proximity alone can contribute.
PROXIMITY_APPLICABILITY = 0.5

# An absence query wants habitat the target does NOT already occupy, so the
# priority of a unit is discounted by how much of it the target already fills:
# gap = (1 - target_fraction) ** ABSENCE_GAP_EXPONENT.
#
# This was a saturation ("above 25% canopy, a tree is not missing here") and
# that was a cliff, kept here as a worked example of how a plausible-sounding
# constant silently deletes an answer: a 38 m chip inside maquis is ALWAYS more
# than 25% canopy and a missing tree is 5 m across, so the rule scored every
# unit of a forested tile at exactly zero and the ranking became arbitrary --
# the flagship query returning a confident, meaningless order. Linear until
# outcome data says otherwise. WHERE THIS SHOULD COME FROM: the detector's
# minimum interpretable clearing size relative to the unit area, which makes
# this a function of chip size and GSD rather than a constant at all.
ABSENCE_GAP_EXPONENT = 1.0

# Compiled context rules: how near counts as near, what a hit is worth, and how
# many rules a policy may carry before it stops being readable in 30 seconds --
# which is the stated bar for approving something that discards 85% of an AOI.
COMPILED_NEAR_M = 100.0
COMPILED_NEAR_FACTOR = 1.4
MAX_COMPILED_RULES = 3

# Morphologies that are hostile ground for anything built, driven, or rooted.
# Physics, not history -- the same line taxonomy.py draws for its priors.
HOSTILE_MORPHOLOGIES = ("Boulder", "Badlands", "RockDipSlope", "SmoothRockSlopes")

# Per-superclass target geometry, used only to fill a compiled policy's
# `target_size_m2` / `min_gsd_m`. WHERE THIS SHOULD COME FROM: the detector's
# own spec sheet, per target type. These are order-of-magnitude guesses.
TARGET_SIZE_M2: dict[str, tuple[float, float]] = {
    "vehicle": (4.0, 30.0),
    "built": (20.0, 5000.0),
    "road": (50.0, 1e9),
    "artifact": (2.0, 5000.0),
    "vegetation": (25.0, 1e9),
    "agriculture": (100.0, 1e9),
    "water": (25.0, 1e9),
    "soil": (50.0, 1e9),
    "rock": (50.0, 1e9),
}
TARGET_MIN_GSD_M: dict[str, float] = {
    "vehicle": 0.3, "artifact": 0.3, "built": 0.5, "road": 0.5,
    "vegetation": 1.0, "agriculture": 1.0, "water": 1.0, "soil": 1.0, "rock": 1.0,
}


@dataclass
class ContextRule:
    when: str            # "dist_to.PavedRoad < 100" | "interface.Maquis_DryGrassland > 40"
    factor: float
    note: str = ""

    def evaluate(self, chip: Chip) -> bool:
        try:
            lhs, op, rhs = self.when.split()
            value = _resolve(chip, lhs)
            if value is None:
                return False
            return value < float(rhs) if op == "<" else value > float(rhs)
        except (ValueError, KeyError):
            return False


def _resolve(chip: Chip, expr: str) -> float | None:
    if expr.startswith("osm."):
        # Features from a joined reference map. `None` when the layer was not
        # joined or the unit does not measure this -- which makes the rule
        # evaluate false, so `chipfeat.missing_features` reports it rather than
        # letting a policy quietly lose a rule its author approved.
        return chip.osm.get(expr.split(".", 1)[1])
    if expr.startswith("dist_to."):
        return chip.dist_to.get(expr.split(".", 1)[1])
    if expr.startswith("interface."):
        a, b = expr.split(".", 1)[1].split("_", 1)
        try:
            ka, kb = cid(a), cid(b)
        except KeyError:
            return None
        return chip.interfaces.get((min(ka, kb), max(ka, kb)), 0.0)
    if expr == "entropy":
        return chip.entropy
    if expr.startswith("frac."):
        return chip.frac(cid(expr.split(".", 1)[1]))
    return None


@dataclass
class Policy:
    """Cacheable per (query, taxonomy version, detector version, gsd band).

    Auditable on purpose: if you are about to discard 85% of an AOI, a human
    should be able to read and approve the rule in 30 seconds.
    """
    policy_id: str
    query: str
    regime: str                                   # R1 | R2 | R3 | R1+R2 ...
    class_weights: dict[str, float] = field(default_factory=dict)
    hard_exclude: list[str] = field(default_factory=list)
    exclude_rationale: str = ""
    context_boosts: list[ContextRule] = field(default_factory=list)
    target_size_m2: tuple[float, float] = (1.0, 1e9)
    min_gsd_m: float = 1.0
    review_required: bool = True

    # --- what the query compiler understood, in words --------------------
    # All defaulted: a hand-written policy leaves them empty and behaves
    # exactly as it did before any of this existed. A compiled policy fills
    # them, and `render` prints them, because a policy nobody can read is a
    # policy nobody can approve.
    query_terms: list[str] = field(default_factory=list)     # phrase -> classes
    applies_to: list[str] = field(default_factory=list)      # the target classes
    context_classes: list[str] = field(default_factory=list)  # explicit habitat
    absence: bool = False                                    # "missing", "no X"
    unmatched: list[str] = field(default_factory=list)       # words not understood
    notes: list[str] = field(default_factory=list)           # warnings for the reviewer
    # Parsed but NOT consumed by anything here. A change-detection query
    # ("where was a house destroyed", "where was a road paved") is the same
    # shape as a detection query -- a query scored per unit of ground -- and it
    # compiles into this same Policy. The next work session scores it across
    # two segmentation maps; on ONE map the compiler falls back to the
    # present-day half of the transition and says so. "*" means an endpoint the
    # query left open.
    transition: tuple[str, str] | None = None

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d["context_boosts"] = [asdict(r) for r in self.context_boosts]
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "Policy":
        d = json.loads(Path(path).read_text())
        d["context_boosts"] = [ContextRule(**r) for r in d.get("context_boosts", [])]
        d["target_size_m2"] = tuple(d.get("target_size_m2", (1.0, 1e9)))
        # JSON has no tuple: a round-tripped transition comes back as a list and
        # would compare unequal to the tuple the compiler produced.
        if d.get("transition") is not None:
            d["transition"] = tuple(d["transition"])
        return Policy(**d)

    def understood(self) -> str:
        """One line a human can check the compiler against."""
        if not self.query_terms and not self.unmatched:
            return "hand-written policy (no query was compiled)"
        parts = ["; ".join(self.query_terms) or "nothing"]
        if self.absence:
            parts.append("ABSENCE sense (looking for where the target is NOT)")
        if self.transition:
            parts.append(f"transition {self.transition[0]} -> {self.transition[1]}")
        if self.context_classes:
            parts.append("context " + "+".join(self.context_classes))
        if self.unmatched:
            parts.append("NOT understood: " + " ".join(self.unmatched))
        return " | ".join(parts)

    def weight_vector(self) -> np.ndarray:
        v = np.zeros(N_CLASSES)
        for name, w in self.class_weights.items():
            v[cid(name)] = w
        return v


# --- built-in policies -----------------------------------------------------
# These are hand-written stand-ins for what an LLM would compile from the query
# text. Hand-writing one first is how you validate the scorer without an LLM in
# the loop at all.

BUILTIN: dict[str, Policy] = {
    "vehicles": Policy(
        policy_id="find-vehicles@v0",
        query="find vehicles, including off-road and partially concealed",
        regime="R1+R2",
        target_size_m2=(4.0, 30.0), min_gsd_m=0.3,
        class_weights={
            "Car": 1.0, "PavedRoad": 0.9, "Pavement": 0.85, "DirtRoad": 0.7,
            "DirtRoadB": 0.6, "Clutter": 0.5, "Shadow": 0.45, "House": 0.4,
            "DryGrassland": 0.25, "Batha": 0.2, "GreenGrassland": 0.2,
        },
        hard_exclude=[
            "Water", "MaralBadlands", "LimestoneBoulder", "DolomiteBoulder",
            "BasaltBoulder", "LimestoneRockDipSlope", "NariRockDipSlope",
            "ChalkSmoothRockSlopes", "MaralSmoothRockSlopes",
        ],
        exclude_rationale=(
            "vehicles cannot traverse or park on boulder fields, steep smooth rock "
            "faces, dissected badlands, or water; slope and surface roughness make "
            "these physically inaccessible"
        ),
        context_boosts=[
            ContextRule("dist_to.PavedRoad < 100", 1.6, "near sealed road"),
            ContextRule("dist_to.DirtRoad < 50", 1.4, "near track"),
            ContextRule("dist_to.House < 60", 1.3, "near built-up"),
        ],
    ),
    "structures": Policy(
        policy_id="find-structures@v0",
        query="find buildings and built structures",
        regime="R1",
        target_size_m2=(20.0, 5000.0), min_gsd_m=0.5,
        class_weights={
            "House": 1.0, "Pavement": 0.7, "BrickWall": 0.6, "Clutter": 0.5,
            "Shadow": 0.4, "PavedRoad": 0.3,
        },
        hard_exclude=["Water", "MaralBadlands", "LimestoneBoulder", "BasaltBoulder"],
        exclude_rationale="buildings are not sited on water, boulder fields, or badlands",
        context_boosts=[ContextRule("dist_to.PavedRoad < 150", 1.4, "road access")],
    ),
    "settlement": Policy(
        policy_id="find-in-settlement@v0",
        query="find ground objects inside the built-up street network",
        regime="R2",
        target_size_m2=(2.0, 500.0), min_gsd_m=0.3,
        class_weights={
            "Clutter": 0.9, "Car": 0.9, "Pavement": 0.7, "House": 0.6,
            "Shadow": 0.55, "Unclassified": 0.5, "PavedRoad": 0.5,
            "DirtRoad": 0.4, "DryGrassland": 0.2,
        },
        hard_exclude=["Water"],
        exclude_rationale="open water cannot host a ground object",
        # The reference-map rules. `osm.dist_road` is a fact about where the
        # streets are; the alternative -- inferring settlement from ST's own
        # `House` density -- infers it from the same map being triaged, which is
        # exactly the circularity the join removes.
        #
        # This is a CHIP-unit policy. `osm.dist_road` is meaningless on the block
        # unit (a block is the complement of the corridor, so every block is
        # zero metres from a road) and `run_blocks` reports it as a rule that did
        # not fire rather than quietly scoring without it.
        context_boosts=[
            ContextRule("osm.dist_road < 30", 1.6, "on the mapped street network"),
            ContextRule("osm.building_frac > 0.15", 1.4, "mapped built-up ground"),
            ContextRule("osm.n_junctions > 0", 1.2, "contains a junction"),
        ],
    ),
    "oov": Policy(
        policy_id="find-out-of-vocabulary@v0",
        query="find anything the taxonomy has no word for (greenhouses, pylons, "
              "fences, tents, disturbed earth)",
        regime="R3",
        target_size_m2=(2.0, 5000.0), min_gsd_m=0.3,
        class_weights={"Clutter": 1.0, "Unclassified": 0.95, "Shadow": 0.5},
        hard_exclude=["Water"],
        exclude_rationale="open water cannot host a concealed ground object",
        context_boosts=[ContextRule("entropy > 2.5", 1.3, "interface-rich chip")],
    ),
}


# --- the query vocabulary --------------------------------------------------
# The taxonomy IS the vocabulary. Almost everything below is read out of
# taxonomy.py, so adding a class there extends the compiler for free. The
# synonym table is the exception -- it is hand-written, therefore it is short,
# it lives in exactly one place, and every compile prints back which entry of it
# fired so a reviewer can catch a wrong mapping in one line.

_LOWER_NAME: dict[str, str] = {c.name.lower(): c.name for c in CLASSES}

# Canopy-bearing classes, DERIVED rather than listed: "tree" means whatever the
# taxonomy says carries woody canopy above TREE_MIN_CANOPY.
TREE_CLASSES: tuple[str, ...] = tuple(
    c.name for c in CLASSES if c.canopy >= TREE_MIN_CANOPY)

# Singular keys only -- the resolver strips a trailing "s"/"es" before giving up.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "tree": TREE_CLASSES,
    "canopy": TREE_CLASSES,
    "forest": ("Maquis", "Garigue"),
    "woodland": ("Maquis", "Garigue"),
    "shrub": ("Maquis", "Garigue", "Batha"),
    "scrub": ("Maquis", "Garigue", "Batha"),
    "bush": ("Maquis", "Garigue", "Batha"),
    "orchard": ("UnirrigatedOrchard", "IrrigatedOrchard"),
    "grove": ("UnirrigatedOrchard", "IrrigatedOrchard"),
    "olive": ("UnirrigatedOrchard", "IrrigatedOrchard"),
    "almond": ("UnirrigatedOrchard", "IrrigatedOrchard"),
    "building": ("House", "BrickWall", "Pavement"),
    "house": ("House", "BrickWall", "Pavement"),
    "structure": ("House", "BrickWall", "Pavement"),
    "settlement": ("House", "Pavement", "BrickWall"),
    "village": ("House", "Pavement", "BrickWall"),
    "wall": ("BrickWall",),
    "fence": ("BrickWall", "Clutter"),
    "car": ("Car",),
    "vehicle": ("Car",),
    "truck": ("Car",),
    "track": ("DirtRoad", "DirtRoadB"),
    "trail": ("DirtRoad", "DirtRoadB"),
    "parking": ("Pavement",),
    "water": ("Water",),
    "pool": ("Water",),
    "reservoir": ("Water",),
    "pond": ("Water",),
    "rubble": ("Clutter", "Unclassified"),
    "debris": ("Clutter", "Unclassified"),
    "spoil": ("Clutter", "Unclassified"),
    "anomaly": ("Clutter", "Unclassified"),
    "crop": ("IrrigatedField",),
    "field": ("IrrigatedField",),
    "grass": ("GreenGrassland", "DryGrassland"),
    "rock": SUPERCLASS["rock"],
}

# Words that map to a class but ask for something the map never predicted. The
# class is still the right ground to search; the note says what the answer
# cannot contain. s5_query.UNANSWERABLE_CONCEPTS is the canonical list -- this
# is the handful that matter for triage, kept local so S3 does not break when
# that table moves.
OVERSPECIFIC: dict[str, str] = {
    "truck": "the taxonomy has one vehicle class, Car -- vehicle TYPE was never "
             "predicted, so this cannot be narrower than 'a vehicle'",
    "fence": "there is no fence class, and a fence is sub-metre wide; BrickWall "
             "and Clutter are the closest ground, not the answer",
    "house": "the building class is called House and carries no function, height "
             "or occupancy",
    "building": "the building class is called House and carries no function, "
                "height or occupancy",
}

# The ordinal series as query words. A series word selects the WHOLE series --
# "degradation" is a question about the vegetation gradient, not about one stage.
SERIES_WORDS: dict[str, tuple[str, ...]] = {
    "degradation": DEGRADATION_SERIES,
    "succession": DEGRADATION_SERIES,
    "formation series": DEGRADATION_SERIES,
    "vegetation series": DEGRADATION_SERIES,
    "road series": ROAD_SERIES,
    "road grade": ROAD_SERIES,
}

# Absence sense. The owner's own example ("a missing tree") is an absence query,
# and an absence query is NOT a presence query with the sign flipped: it targets
# ground where the class's CONTEXT is present and the class itself is not.
ABSENCE_WORDS = frozenset((
    "missing", "no", "without", "absent", "absence", "removed", "cleared",
    "gone", "lost", "destroyed", "demolished", "razed", "felled", "cut",
    "lacking", "not", "disappeared", "vanished",
))

# "near roads" is a statement about context, not about the target.
PROXIMITY_WORDS = frozenset((
    "near", "beside", "alongside", "along", "next", "adjacent", "around",
    "within", "close",
))

# Change-detection verbs. "$" = substitute the class the query named; "*" = an
# endpoint the query left open. Nothing consumes `transition` yet -- see the
# field comment on Policy.
TRANSITION_VERBS: dict[str, tuple[str, str]] = {
    "destroyed": ("$", "Clutter"),
    "demolished": ("$", "Clutter"),
    "razed": ("$", "Clutter"),
    "removed": ("$", "*"),
    "felled": ("$", "*"),
    "cleared": ("$", "*"),
    "paved": ("DirtRoad", "PavedRoad"),
    "sealed": ("DirtRoad", "PavedRoad"),
    "planted": ("*", "$"),
    "built": ("*", "$"),
    "constructed": ("*", "$"),
    "appeared": ("*", "$"),
}

# Query glue. Not vocabulary, not an error either -- it must not end up in
# `unmatched`, which exists to say "I did not understand this word".
STOPWORDS = frozenset((
    "find", "search", "look", "looking", "show", "list", "get", "where", "was",
    "were", "is", "are", "be", "been", "a", "an", "the", "in", "on", "of",
    "for", "to", "at", "by", "with", "from", "me", "us", "any", "all", "some",
    "that", "this", "these", "those", "and", "or", "it", "its", "there", "has",
    "have", "had", "do", "does", "did", "please", "which", "what", "who",
    "detect", "spot", "identify", "locate", "area", "areas", "place", "places",
    "ground", "unit", "units", "chip", "chips", "m", "km",
))

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_ARROW_RE = re.compile(
    r"^(.*?)\s*(?:->|=>|-->|→|\bbecame\b|\bbecomes\b|\bturned into\b)\s*(.*)$")


def _lookup(phrase: str) -> tuple[str, ...] | None:
    """One phrase -> class names. Exact class names win over everything."""
    p = phrase.strip()
    if not p:
        return None
    cands = [p]
    if p.endswith("es"):
        cands.append(p[:-2])
    if p.endswith("s"):
        cands.append(p[:-1])
    for cand in cands:
        key = cand.replace(" ", "").replace("_", "")
        if key in _LOWER_NAME:
            return (_LOWER_NAME[key],)
        if cand in SUPERCLASS:
            return SUPERCLASS[cand]
        if cand in SERIES_WORDS:
            return SERIES_WORDS[cand]
        if cand in SYNONYMS:
            return SYNONYMS[cand]
    return None


def _scan(tokens: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Greedy longest-phrase scan. -> (classes, terms understood, unmatched)."""
    classes: list[str] = []
    terms: list[str] = []
    unmatched: list[str] = []
    i = 0
    while i < len(tokens):
        hit = None
        for n in (3, 2, 1):
            if i + n <= len(tokens):
                phrase = " ".join(tokens[i:i + n])
                got = _lookup(phrase)
                if got:
                    hit = (phrase, got, n)
                    break
        if hit is None:
            t = tokens[i]
            if t not in STOPWORDS and not t.isdigit():
                unmatched.append(t)
            i += 1
            continue
        phrase, got, n = hit
        terms.append(f"{phrase} -> {'+'.join(got)}")
        for g in got:
            if g not in classes:
                classes.append(g)
        i += n
    return classes, terms, unmatched


def _habitat(targets: tuple[str, ...]) -> dict[str, float]:
    """Classes that make ground PLAUSIBLE for `targets`, and how plausible.

    Three sources, all of them statements the taxonomy already makes:
      * `PRIORS` co-occurrence -- "a vehicle sits on something trafficable,
        near built-up ground". These are the surviving priors: geometry and
        physics, not history (see the retired soil-genesis priors).
      * `class_distance` -- a near-miss on an ordinal series is partial
        evidence. Garigue is plausible maquis ground; a boulder field is not.
      * trafficability, for vehicle queries only, so an off-road vehicle query
        is not silently pruned down to the road network.
    """
    out: dict[str, float] = {}
    tset = set(targets)

    def bump(name: str, v: float) -> None:
        if name not in tset and v > 0:
            out[name] = max(out.get(name, 0.0), float(v))

    for t in targets:
        for p in PRIORS_BY_SUBJECT.get(t, ()):
            if p.kind == "expects":
                for other in p.others:
                    bump(other, 1.0)
        ti = cid(t)
        for c in CLASSES:
            d = class_distance(ti, c.id)
            if d < 1.0:
                bump(c.name, NEAR_MISS_WEIGHT * (1.0 - d))

    if any(SUPERCLASS_OF[cid(t)] == "vehicle" for t in targets):
        for c in CLASSES:
            if c.traffic >= TRAFFICABLE_MIN:
                bump(c.name, TRAFFICABLE_HABITAT_CREDIT)
    return out


@lru_cache(maxsize=128)
def _query_vectors(applies_to: tuple[str, ...], context: tuple[str, ...]
                   ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """(target vector, habitat vector, proximity anchors) over N_CLASSES.

    Cached per query shape, because the alternative is rebuilding three tables
    per chip and the whole claim of this module is that the per-chip cost is
    arithmetic. Returned arrays are shared -- read them, never write them.
    """
    tgt = np.zeros(N_CLASSES)
    hab = np.zeros(N_CLASSES)
    for n in applies_to:
        tgt[cid(n)] = 1.0
    habitat = _habitat(applies_to)
    for n, v in habitat.items():
        hab[cid(n)] = v
    for n in context:                      # explicit "near X" / transition source
        if n not in applies_to:
            hab[cid(n)] = 1.0
    # Only ANCHOR_CLASSES have a distance field. Strong habitat only: being
    # 40 m from something 0.15-plausible is not evidence of anything.
    anchors = tuple(n for n in list(applies_to) + list(context) +
                    [k for k, v in habitat.items() if v >= 1.0]
                    if n in ANCHOR_CLASSES)
    return tgt, hab, tuple(dict.fromkeys(anchors))


def _hard_exclude(targets: tuple[str, ...],
                  habitat: dict[str, float]) -> tuple[list[str], str]:
    """Ground the target physically cannot occupy, with the sentence that says why.

    Never excludes a target or its habitat: a query for boulder fields must not
    hard-exclude boulder fields, and the check is cheap.
    """
    supers = {SUPERCLASS_OF[cid(t)] for t in targets}
    hostile = [c.name for c in CLASSES if c.morphology in HOSTILE_MORPHOLOGIES]
    ex: list[str] = []
    why: list[str] = []
    if supers & {"built", "road", "vehicle"}:
        ex += ["Water"] + hostile
        why.append("nothing is built on or driven over open water, boulder "
                   "fields, dissected badlands or steep smooth rock faces")
    if supers & {"vegetation", "agriculture"}:
        ex += ["Water"] + hostile
        why.append("woody vegetation does not root on open water, boulder "
                   "fields, badlands or bare rock faces")
    if supers & {"artifact"}:
        ex += ["Water"]
        why.append("open water cannot host a concealed ground object")
    if not ex:
        return [], ("nothing excluded: the target is itself a terrain surface, "
                    "so no ground type is implausible for it")
    keep = [n for n in dict.fromkeys(ex)
            if n not in targets and habitat.get(n, 0.0) < 0.5]
    return keep, "; ".join(why)


def _context_rules(targets: tuple[str, ...], context: tuple[str, ...],
                   regime_r3: bool) -> list[ContextRule]:
    """Distance rules over the anchor fields the chip index already computed.

    Only ANCHOR_CLASSES have a distance field, so only they can appear here.
    Capped at MAX_COMPILED_RULES: a policy nobody reads is a policy nobody
    approved.
    """
    order: list[str] = []
    for n in list(context) + list(targets):
        if n in ANCHOR_CLASSES and n not in order:
            order.append(n)
    for t in targets:                       # what the taxonomy says t sits near
        for p in PRIORS_BY_SUBJECT.get(t, ()):
            if p.kind == "expects":
                for other in p.others:
                    if other in ANCHOR_CLASSES and other not in order:
                        order.append(other)
    rules = [ContextRule(f"dist_to.{n} < {COMPILED_NEAR_M:.0f}",
                         COMPILED_NEAR_FACTOR, f"near {n}")
             for n in order[:MAX_COMPILED_RULES]]
    if regime_r3 and len(rules) < MAX_COMPILED_RULES:
        rules.append(ContextRule("entropy > 2.5", 1.3, "interface-rich chip"))
    return rules


def _geometry(targets: tuple[str, ...]) -> tuple[tuple[float, float], float]:
    supers = [SUPERCLASS_OF[cid(t)] for t in targets] or ["artifact"]
    lo = min(TARGET_SIZE_M2.get(s, (1.0, 1e9))[0] for s in supers)
    hi = max(TARGET_SIZE_M2.get(s, (1.0, 1e9))[1] for s in supers)
    return (lo, hi), min(TARGET_MIN_GSD_M.get(s, 1.0) for s in supers)


def compile_query(text: str, gsd: float | None = None) -> Policy:
    """Natural query -> auditable Policy. DETERMINISTIC: there is no LLM here.

    That is deliberate and temporary. The architecture says a model compiles the
    query into a small cacheable policy once, and a deterministic scorer applies
    it to millions of chips. This function is the hand-written stand-in for the
    compile step -- the same role BUILTIN plays for a single query -- so the
    scorer, the applicability model and the audit surface can be validated with
    no model in the loop at all. Replacing it with a model-compiled policy
    changes what fills the Policy, not what consumes it.

    Understood: exact class names, superclass names, the ordinal series, a small
    synonym table, an absence sense ("missing tree", "no trees", "where trees
    were removed"), a proximity clause ("near roads"), and transition queries
    ("House -> Clutter", "where was a road paved") which parse into
    `Policy.transition` and are not otherwise consumed yet.
    """
    raw = (text or "").strip()
    low = raw.lower()
    notes: list[str] = []
    terms: list[str] = []
    unmatched: list[str] = []
    targets: list[str] = []
    context: list[str] = []
    transition: tuple[str, str] | None = None

    # 1. An explicit transition -- "X -> Y", "X became Y". Parsed first because
    #    the arrow splits the query into two halves that mean different things.
    arrow = _ARROW_RE.match(low) if ("->" in low or "=>" in low or "→" in low
                                     or " became " in low or " becomes " in low
                                     or " turned into " in low) else None
    if arrow:
        left, right = arrow.group(1), arrow.group(2)
        lc, lt, lu = _scan(_TOKEN_RE.findall(left))
        rc, rt, ru = _scan(_TOKEN_RE.findall(right))
        terms += lt + rt
        unmatched += lu + ru
        if lc or rc:
            transition = (lc[0] if lc else "*", rc[0] if rc else "*")
            # On ONE map only the present-day half is observable: the `to` side
            # is what should be visible now, the `from` side is context.
            targets = rc or lc
            context = [n for n in lc if n not in targets]
            if not rc:
                notes.append("transition target is open ('*'): scored as the "
                             "ABSENCE of the source class on this single map")
            terms.append(f"transition {transition[0]} -> {transition[1]} "
                         "(parsed; change detection does not consume it yet)")
        absence = not rc
    else:
        tokens = _TOKEN_RE.findall(low)
        # 2. Absence and transition verbs, pulled out before the class scan so
        #    they are recorded as understood rather than reported as noise.
        absence = any(t in ABSENCE_WORDS for t in tokens)
        verbs = [t for t in tokens if t in TRANSITION_VERBS]
        for t in dict.fromkeys(t for t in tokens if t in ABSENCE_WORDS):
            terms.append(f"{t} -> absence sense")
        rest = [t for t in tokens if t not in ABSENCE_WORDS
                and t not in TRANSITION_VERBS]

        # 3. A proximity clause is context, not target.
        cut = next((i for i, t in enumerate(rest) if t in PROXIMITY_WORDS), None)
        head = rest if cut is None else rest[:cut]
        tail = [] if cut is None else rest[cut + 1:]
        targets, t1, u1 = _scan(head)
        context, t2, u2 = _scan(tail)
        terms += t1
        if cut is not None:
            terms.append(f"{rest[cut]} -> context, not target")
            terms += t2
        unmatched += u1 + u2

        # 4. Verb-implied transitions. "$" takes the class the query named.
        if verbs:
            v = verbs[0]
            src, dst = TRANSITION_VERBS[v]
            named = targets[0] if targets else "*"
            src = named if src == "$" else src
            dst = named if dst == "$" else dst
            transition = (src if src != "$" else "*", dst if dst != "$" else "*")
            terms.append(f"{v} -> transition {transition[0]} -> {transition[1]} "
                         "(parsed; change detection does not consume it yet)")

    targets = list(dict.fromkeys(targets))
    context = list(dict.fromkeys(n for n in context if n not in targets))
    for t in terms:
        head_word = t.split(" ->")[0]
        if head_word in OVERSPECIFIC:
            notes.append(f"'{head_word}': {OVERSPECIFIC[head_word]}")

    tt = tuple(targets)
    habitat = dict(_habitat(tt)) if tt else {}
    for n in context:
        habitat[n] = 1.0

    # 5. Weights. Presence ranks the target and its context; ABSENCE ranks the
    #    CONTEXT and gives the target itself nothing, because a chip full of
    #    maquis is the last place a missing tree is.
    weights: dict[str, float] = {}
    if absence:
        for n, v in habitat.items():
            weights[n] = round(v, 3)
    else:
        for n in targets:
            weights[n] = 1.0
        for n, v in habitat.items():
            weights[n] = max(weights.get(n, 0.0), round(CONTEXT_WEIGHT * v, 3))

    ex, why = (_hard_exclude(tt, habitat) if tt else
               ([], "nothing to exclude: no target class was understood"))
    if absence:
        regime = "R2"                      # context-bound by construction
    elif tt and set(tt) <= {"Clutter", "Unclassified", "Shadow"}:
        regime = "R3"
    elif tt:
        regime = "R1"
    else:
        regime = "none"
    rules = _context_rules(tt, tuple(context), regime == "R3") if tt else []
    if rules and regime == "R1":
        regime = "R1+R2"
    size, min_gsd = _geometry(tt) if tt else ((1.0, 1e9), 1.0)

    if not tt:
        # Loud, not silent. An empty policy that "matches everything" is how a
        # triage system quietly returns the top 20% of nothing.
        notes.insert(0, "NOTHING IN THIS QUERY MATCHED THE TAXONOMY. No target "
                        "class was identified, so this policy applies NOWHERE "
                        "and selects nothing. Rephrase using class names, a "
                        "superclass (" + ", ".join(sorted(SUPERCLASS)) + "), or "
                        "a known synonym -- do not treat an empty result as "
                        "'no targets found'.")
    if gsd is not None and gsd > min_gsd:
        notes.append(f"GSD {gsd:.2f} m is coarser than the {min_gsd:.2f} m this "
                     f"target needs: the segmentation may not resolve it at all, "
                     f"and a triage score cannot tell you that it did not.")
    if unmatched and tt:
        notes.append("ignored words: " + " ".join(unmatched) +
                     " -- check that none of them changed the meaning")

    # Cacheable per (query, taxonomy version, detector version, GSD band) is the
    # design; the GSD goes into the key here, the other two do not exist as
    # version strings yet and pretending otherwise would be worse than not.
    key = f"{low}|{','.join(targets)}|{absence}|{gsd if gsd is None else round(gsd, 3)}"
    slug = re.sub(r"[^a-z0-9]+", "-", low).strip("-")[:28] or "nomatch"
    return Policy(
        policy_id=f"q-{slug}@{hashlib.sha1(key.encode()).hexdigest()[:8]}",
        query=raw,
        regime=regime,
        class_weights=weights,
        hard_exclude=ex,
        exclude_rationale=why,
        context_boosts=rules,
        target_size_m2=size,
        min_gsd_m=min_gsd,
        query_terms=terms,
        applies_to=targets,
        context_classes=context,
        absence=absence,
        unmatched=unmatched,
        notes=notes,
        transition=transition,
    )


# --- applicability ---------------------------------------------------------

def _proximity(chip: Chip, names: tuple[str, ...]) -> float:
    """0..1 evidence that habitat is nearby, from the anchor distance fields.

    Only ANCHOR_CLASSES have a distance field, and the block unit has none at
    all -- both cases return 0, which is "no evidence", not "far away".
    """
    if not chip.dist_to:
        return 0.0
    best = min((chip.dist_to.get(n, float("inf")) for n in names),
               default=float("inf"))
    if not np.isfinite(best):
        return 0.0
    return float(max(0.0, 1.0 - best / APPLICABILITY_NEAR_M))


def applicability(chip: Chip, policy: Policy) -> float:
    """Could the thing being asked about plausibly be here AT ALL? 0..1.

    This is the half that is NOT priority. A tree query over a basalt boulder
    field returns ~0 and no amount of interface richness or entropy can raise
    it, because the question is meaningless on that ground -- which is a
    different statement from "searched and found nothing".

    Presence: target mass, plus discounted habitat mass (co-occurrence context,
    ordinal near-misses, trafficable ground for vehicles), floored by proximity
    to habitat.
    Absence:  is this the kind of ground the target BELONGS on -- and a forest
    is maximally tree country, evidenced by its trees. Where the gap actually is
    belongs to the priority half, not here (see `score_query`). Getting this
    wrong is easy and was: an earlier reading killed applicability off as the
    target filled the unit, so "find missing trees" scored 0 in a forest AND 0
    on a desert road, which is precisely the distinction the whole mechanism
    exists to draw.

    A hand-written BUILTIN policy has no `applies_to` and therefore no
    applicability model; it returns 1.0, which means "not assessed" and is
    reported as such rather than as a measurement.
    """
    if not policy.applies_to:
        return 0.0 if policy.query_terms or policy.unmatched or policy.notes else 1.0
    tgt, hab, anchors = _query_vectors(tuple(policy.applies_to),
                                       tuple(policy.context_classes))
    t = float(np.dot(tgt, chip.class_frac))
    h = float(np.dot(hab, chip.class_frac))
    near = PROXIMITY_APPLICABILITY * _proximity(chip, anchors)
    if policy.absence:
        # Habitat OR the target itself: both are evidence that this is ground
        # where the thing belongs, which is what applicability asks.
        return float(np.clip(min(1.0, max(t, h, near)), 0.0, 1.0))
    a = min(1.0, t + HABITAT_CONTEXT_WEIGHT * h)
    return float(np.clip(max(a, near), 0.0, 1.0))


@dataclass
class ScoredChip:
    chip: Chip
    score: float
    excluded: bool
    reason: str = ""
    # 1.0 means "not assessed" for a hand-written policy, and a measured
    # applicability for a compiled query. `render` only prints the column when
    # a query was actually compiled -- printing 1.00 for a BUILTIN policy would
    # be inventing a measurement nobody made.
    applicability: float = 1.0


def score_chips(cidx: ChipIndex, policy: Policy) -> list[ScoredChip]:
    """Pure arithmetic over precomputed layers. Milliseconds over millions."""
    w = policy.weight_vector()
    ex = [cid(n) for n in policy.hard_exclude]
    ent_w = ENTROPY_WEIGHT.get(policy.regime.split("+")[0], 0.15)
    out: list[ScoredChip] = []

    for ch in cidx.chips:
        ex_frac = float(ch.class_frac[ex].sum()) if ex else 0.0
        if ex_frac > EXCLUDE_DOMINANCE:
            out.append(ScoredChip(ch, 0.0, True,
                                  f"{ex_frac:.0%} excluded classes"))
            continue
        s = float(np.dot(w, ch.class_frac))
        # A dominance gate alone is too coarse on heterogeneous chips: a chip
        # that is 60% boulder field still scores on its other 40%. Discount by
        # the excluded mass so partial exclusion is worth something.
        s *= (1.0 - ex_frac)
        for rule in policy.context_boosts:
            if rule.evaluate(ch):
                s *= rule.factor
        s *= (1.0 + ent_w * ch.entropy / 4.0)
        out.append(ScoredChip(ch, s, False))
    return out


@dataclass
class Selection:
    selected: list[ScoredChip]
    rejected: list[ScoredChip]
    control: list[ScoredChip]
    budget_frac: float
    score_captured: float

    @property
    def cost_reduction(self) -> float:
        n = len(self.selected) + len(self.rejected)
        return 1.0 - (len(self.selected) + len(self.control)) / max(n, 1)

    @property
    def n_not_applicable(self) -> int:
        return sum(1 for s in self.rejected if s.applicability < APPLICABILITY_MIN)

    @property
    def cost_reduction_searchable(self) -> float:
        """The saving over ground the query could actually apply to.

        `cost_reduction` divides by every unit including the ones the query does
        not apply to, so a query that is meaningless over half an AOI prints a
        75% saving where the real figure over searchable ground is 50%. That is
        a search that never happened, counted as a search that was skipped --
        and this module's own docstring says a caller reporting without reading
        `n_not_applicable` "is reporting a search that never happened".
        """
        live = len(self.selected) + len(self.rejected) - self.n_not_applicable
        if live <= 0:
            return 0.0
        return 1.0 - (len(self.selected) + len(self.control)) / live


def select(
    scored: list[ScoredChip],
    budget_frac: float = 0.20,
    control_frac: float = CONTROL_SET_FRACTION,
    seed: int = 0,
) -> Selection:
    """Naive: top-K by score.

    The real version targets *calibrated expected recall* -- fit p(hit | score)
    on outcome data, then take the smallest set reaching the recall target. That
    gives users a knob they can reason about ("95% of targets at 11% of cost")
    instead of an arbitrary K. Not implementable until outcomes exist; see
    docs/solutions/S3.md.
    """
    live = [s for s in scored if not s.excluded]
    live.sort(key=lambda s: -s.score)
    k = max(1, int(round(len(scored) * budget_frac)))
    selected, rest = live[:k], live[k:]
    rejected = rest + [s for s in scored if s.excluded]

    rng = np.random.default_rng(seed)
    n_ctl = int(round(len(rejected) * control_frac))
    if rejected and control_frac > 0:
        # Rounding can hit 0 for pools under ~17 rejected chips, which would
        # silently delete the only unbiased recall signal. One control chip is
        # weak evidence but never zero evidence; render() states the weakness.
        n_ctl = max(1, n_ctl)
    idx = rng.choice(len(rejected), size=min(n_ctl, len(rejected)), replace=False) \
        if rejected else []
    control = [rejected[i] for i in idx]

    total = sum(s.score for s in scored) or 1.0
    return Selection(selected, rejected, control, budget_frac,
                     sum(s.score for s in selected) / total)


NOT_APPLICABLE = "the query does not apply to this ground"


def _is_compiled(policy: Policy) -> bool:
    """Did a query compiler produce this policy? Hand-written ones leave every
    one of these empty, and then `render` prints exactly what it always did."""
    return bool(policy.query_terms or policy.applies_to or policy.unmatched
                or policy.notes)


def score_query(cidx: ChipIndex, policy: Policy) -> list[ScoredChip]:
    """worth_a_look = applicability x priority, per unit.

    `score_chips` computes the priority half and is left exactly as it was.
    This wraps it: a unit the query does not apply to is dropped with that
    stated as the reason, which is a different rejection from "scored low" and
    a different rejection again from "hard-excluded by class".
    """
    scored = score_chips(cidx, policy)
    if not _is_compiled(policy):
        return scored
    for s in scored:
        a = applicability(s.chip, policy)
        s.applicability = a
        if a < APPLICABILITY_MIN:
            s.score = 0.0
            # Keep a hard-exclude reason if there already is one: "boulder
            # field" is more informative than "not applicable", and the counts
            # in render() are taken from `applicability`, not from the string.
            if not s.excluded:
                s.excluded = True
                s.reason = f"{NOT_APPLICABLE} (applicability {a:.2f})"
        else:
            s.score *= a
            if policy.absence:
                # The gap term. A unit already packed with the target is a
                # worse place to hunt for a missing one -- but only worse, not
                # disqualified, which is why this is linear and not a
                # threshold (see ABSENCE_GAP_EXPONENT). Applied here rather
                # than in `applicability` so each half keeps its meaning:
                # applicability says the query makes sense on this ground,
                # priority says where on that ground to look.
                tgt, _hab, _anch = _query_vectors(tuple(policy.applies_to),
                                                  tuple(policy.context_classes))
                t = float(np.dot(tgt, s.chip.class_frac))
                s.score *= float(max(0.0, 1.0 - t) ** ABSENCE_GAP_EXPONENT)
    return scored


def run_query(cidx: ChipIndex, query: str, budget_frac: float = 0.20,
              gsd: float | None = None) -> tuple[Policy, Selection, dict]:
    """Compile a query, score every unit by applicability x priority, select.

    Returns (policy, selection, diagnostics). The diagnostics are not decoration:
    `n_not_applicable` is the count of units the question was MEANINGLESS on,
    and a caller that reports "nothing found" without reading it is reporting a
    search that never happened.
    """
    policy = compile_query(query, gsd=gsd if gsd is not None else cidx.gsd)
    scored = score_query(cidx, policy)
    sel = select(scored, budget_frac=budget_frac)
    n_na = sum(1 for s in scored if s.applicability < APPLICABILITY_MIN)
    n_ex = sum(1 for s in scored
               if s.excluded and s.applicability >= APPLICABILITY_MIN)
    live = [s.applicability for s in scored
            if s.applicability >= APPLICABILITY_MIN]
    diag = {
        "query": policy.query,
        "policy_id": policy.policy_id,
        "regime": policy.regime,
        "understood": policy.understood(),
        "matched_vocabulary": bool(policy.applies_to),
        "applies_to": list(policy.applies_to),
        "absence": policy.absence,
        "transition": policy.transition,
        "unmatched": list(policy.unmatched),
        "n_units": len(scored),
        "n_not_applicable": n_na,
        "frac_not_applicable": n_na / max(len(scored), 1),
        "n_hard_excluded": n_ex,
        "n_selected": len(sel.selected),
        "n_control": len(sel.control),
        # Which constraint bound: the budget, or the question not applying? If
        # this is False the budget was never the limit -- every unit the query
        # applies to was dispatched, and buying more budget buys nothing.
        "budget_binding": len(sel.selected) >= max(
            1, int(round(len(scored) * budget_frac))),
        "mean_applicability_where_applicable": (
            float(np.mean(live)) if live else 0.0),
        "cost_reduction": sel.cost_reduction,
        "notes": list(policy.notes),
    }
    return policy, sel, diag


def render(sel: Selection, policy: Policy, top_k: int = 15,
           unit: str = "chip", missing: list[str] | None = None,
           labels: dict[int, str] | None = None) -> str:
    compiled = _is_compiled(policy)
    everything = sel.selected + sel.rejected
    n = len(everything)
    n_na = sum(1 for s in everything if s.applicability < APPLICABILITY_MIN)
    n_ex = sum(1 for s in sel.rejected
               if s.excluded and s.applicability >= APPLICABILITY_MIN)
    # The actual fraction, not the target: on small pools the floor of one
    # control chip makes them differ, and printing the target would lie.
    ctl_frac = len(sel.control) / max(len(sel.rejected), 1)
    lines = [
        f"# S3 triage -- policy {policy.policy_id} (regime {policy.regime})",
        f"# query: {policy.query}",
        f"# hard-exclude rationale: {policy.exclude_rationale}",
        f"# unit of spend: {unit}",
    ]
    if compiled:
        lines.insert(2, f"# understood: {policy.understood()}")
        for note in policy.notes:
            lines.append(f"# WARNING: {note}")
    lines += [
        "",
        f"{unit + 's total':21}{n}",
    ]
    if compiled:
        # NOT the same statement as "hard-excluded" and NOT the same statement
        # as "scored low". These units were never a place the question meant
        # anything, and reporting them as "searched, nothing found" is the
        # silent wrong answer.
        lines.append(
            f"not applicable       {n_na} ({n_na / max(n, 1):.0%}) -- "
            f"{NOT_APPLICABLE}")
    lines += [
        f"hard-excluded        {n_ex} ({n_ex / max(n, 1):.0%})",
        f"selected (budget)    {len(sel.selected)} ({len(sel.selected) / max(n, 1):.0%})",
        f"control set          {len(sel.control)} "
        f"({ctl_frac:.0%} of rejected, target {CONTROL_SET_FRACTION:.0%} -- "
        f"MANDATORY, this is the only unbiased recall signal)",
    ]
    if missing:
        lines.append("")
        lines.append(
            f"# WARNING: {len(missing)} policy rule(s) reference a feature the "
            f"`{unit}` unit does not measure, so they did not fire and this "
            f"selection is NOT the policy its author approved:")
        for m in missing:
            lines.append(f"#   {m}")
        lines.append("# Either run this policy on the chip unit, or rewrite the "
                     "rule against a feature this unit has.")
    if 0 < len(sel.control) < CONTROL_MIN_INFORMATIVE:
        lines.append(
            f"# WARNING: a control set of {len(sel.control)} {unit}(s) estimates "
            f"recall with very wide confidence intervals -- treat it as a smoke "
            f"test, not a recall measurement (needs >= {CONTROL_MIN_INFORMATIVE})."
        )
    if compiled and n and n_na == n:
        # The one case where the cost line below is actively misleading: 97% of
        # nothing is not a saving. Whatever the caller does with this result, it
        # must not be reported as "searched the AOI and found nothing".
        lines.append(
            f"# WARNING: the query applies to NO {unit} in this AOI. The cost "
            f"figure below is not a saving -- nothing was searched. This is "
            f"'the question does not apply to this ground', which is a "
            f"different answer from 'nothing is here'.")
    lines += [
        f"cost reduction       {sel.cost_reduction:.0%}"
        + (f"  (over ALL units; {sel.cost_reduction_searchable:.0%} over the "
           f"{n - sel.n_not_applicable} units the query applies to)"
           if compiled and sel.n_not_applicable else ""),
        f"score captured       {sel.score_captured:.0%} of total prior mass",
        "",
        "# WARNING: 'score captured' is prior mass, NOT recall. Recall requires "
        "detector outcomes; until then this number is a plausibility check only.",
        "",
        f"## top {top_k} selected {unit}s",
        f"{unit}\tr\tc\tscore" + ("\tappl" if compiled else "") + "\ttop_classes"
        + ("\tbounded_by" if unit == "block" else ""),
    ]
    for s in sel.selected[:top_k]:
        order = np.argsort(-s.chip.class_frac)[:3]
        top = " ".join(f"{BY_ID[int(k)].name}:{s.chip.class_frac[k]:.2f}"
                       for k in order if s.chip.class_frac[k] > 0)
        extra = ""
        if labels is not None:
            extra = "\t" + labels.get(s.chip.id, "")
        # `appl` is printed only for a compiled query. A hand-written policy has
        # no applicability model, and printing 1.00 for it would present an
        # assumption as a measurement.
        appl = f"\t{s.applicability:.2f}" if compiled else ""
        lines.append(f"{s.chip.id}\t{s.chip.row}\t{s.chip.col}\t{s.score:.3f}"
                     f"{appl}\t{top}{extra}")
    if len(sel.selected) > top_k:
        lines.append(f"# NOTE: {len(sel.selected) - top_k} further selected "
                     f"{unit}s omitted")
    return "\n".join(lines)


def run(cidx: ChipIndex, policy_name: str = "vehicles",
        budget_frac: float = 0.20) -> tuple[Policy, Selection]:
    policy = BUILTIN[policy_name]
    return policy, select(score_chips(cidx, policy), budget_frac=budget_frac)


def run_blocks(raster, layer, policy_name: str = "vehicles",
               budget_frac: float = 0.20, min_area_m2: float = 200.0):
    """Same policy, same scorer, blocks instead of grid squares.

    Returns (policy, selection, missing_rules, block_labels). The third element
    is not
    optional decoration: under the block unit a policy loses every rule that
    depends on the chip index's distance fields, and a selection made by a
    silently reduced policy is not the selection anybody approved.
    """
    from ..osm import chipfeat

    policy = BUILTIN[policy_name]
    bidx = chipfeat.blocks_as_chips(raster, layer, min_area_m2=min_area_m2)
    labels = {ch.id: layer.blocks.get(ch.id).label for ch in bidx.chips}
    missing = chipfeat.missing_features(policy, chipfeat.BLOCK_FEATURES,
                                        has_dist=False, has_interfaces=False)
    sel = select(score_chips(bidx, policy), budget_frac=budget_frac)
    return policy, sel, missing, labels
