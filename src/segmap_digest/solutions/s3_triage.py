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
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..index import Chip, ChipIndex
from ..taxonomy import BY_ID, N_CLASSES, cid

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

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d["context_boosts"] = [asdict(r) for r in self.context_boosts]
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "Policy":
        d = json.loads(Path(path).read_text())
        d["context_boosts"] = [ContextRule(**r) for r in d.get("context_boosts", [])]
        d["target_size_m2"] = tuple(d.get("target_size_m2", (1.0, 1e9)))
        return Policy(**d)

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


@dataclass
class ScoredChip:
    chip: Chip
    score: float
    excluded: bool
    reason: str = ""


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


def render(sel: Selection, policy: Policy, top_k: int = 15) -> str:
    n = len(sel.selected) + len(sel.rejected)
    n_ex = sum(1 for s in sel.rejected if s.excluded)
    # The actual fraction, not the target: on small pools the floor of one
    # control chip makes them differ, and printing the target would lie.
    ctl_frac = len(sel.control) / max(len(sel.rejected), 1)
    lines = [
        f"# S3 triage -- policy {policy.policy_id} (regime {policy.regime})",
        f"# query: {policy.query}",
        f"# hard-exclude rationale: {policy.exclude_rationale}",
        "",
        f"chips total          {n}",
        f"hard-excluded        {n_ex} ({n_ex / max(n, 1):.0%})",
        f"selected (budget)    {len(sel.selected)} ({len(sel.selected) / max(n, 1):.0%})",
        f"control set          {len(sel.control)} "
        f"({ctl_frac:.0%} of rejected, target {CONTROL_SET_FRACTION:.0%} -- "
        f"MANDATORY, this is the only unbiased recall signal)",
    ]
    if 0 < len(sel.control) < CONTROL_MIN_INFORMATIVE:
        lines.append(
            f"# WARNING: a control set of {len(sel.control)} chip(s) estimates "
            f"recall with very wide confidence intervals -- treat it as a smoke "
            f"test, not a recall measurement (needs >= {CONTROL_MIN_INFORMATIVE})."
        )
    lines += [
        f"cost reduction       {sel.cost_reduction:.0%}",
        f"score captured       {sel.score_captured:.0%} of total prior mass",
        "",
        "# WARNING: 'score captured' is prior mass, NOT recall. Recall requires "
        "detector outcomes; until then this number is a plausibility check only.",
        "",
        f"## top {top_k} selected chips",
        "chip\tr\tc\tscore\ttop_classes",
    ]
    for s in sel.selected[:top_k]:
        order = np.argsort(-s.chip.class_frac)[:3]
        top = " ".join(f"{BY_ID[int(k)].name}:{s.chip.class_frac[k]:.2f}"
                       for k in order if s.chip.class_frac[k] > 0)
        lines.append(f"{s.chip.id}\t{s.chip.row}\t{s.chip.col}\t{s.score:.3f}\t{top}")
    if len(sel.selected) > top_k:
        lines.append(f"# NOTE: {len(sel.selected) - top_k} further selected chips omitted")
    return "\n".join(lines)


def run(cidx: ChipIndex, policy_name: str = "vehicles",
        budget_frac: float = 0.20) -> tuple[Policy, Selection]:
    policy = BUILTIN[policy_name]
    return policy, select(score_chips(cidx, policy), budget_frac=budget_frac)
