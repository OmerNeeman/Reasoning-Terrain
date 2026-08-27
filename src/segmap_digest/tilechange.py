"""Two dates, one grid: which tiles changed enough to deserve another look.

This is scaffolding for the change-detection work, not that work. What it fixes
in advance is the mistake every naive change detector makes: **treating all
difference as change.**

`s6_change` already makes that argument at the region level — a
`GreenGrassland -> DryGrassland` transition is the same ground in two seasons, and
a raw pixel diff calls 100% of a tile changed when ~76% of the differing area is
phenology. The tile map needs the same argument at its own altitude, because the
question a reader asks of a map is not "what differs" but *"where should I look
first"*, and those are different orderings.

So the unit here is **excess change**: how much two dates differ, minus how much
this particular ground was always going to differ. The subtrahend comes from
`s4_products.change_volatility`, which scores each tile on how volatile its class
mix is — `Shadow` and `DryGrassland` high, `House` and `PavedRoad` near zero. A
tile of dry grassland that flipped green is unremarkable. A tile of houses that
flipped to `Clutter` is not, even if fewer pixels moved.

Three things this module refuses to do, each because doing them quietly is worse
than not doing them:

  **Compare mismatched grids.** Two tile indices built at different `tile_px`, or
    over different rasters, are not comparable, and a per-tile diff across them
    would be arithmetic over unrelated ground. It raises.
  **Call a direction.** "Changed" is symmetric. Which date is the newer one is the
    caller's knowledge, not this module's, and `gained`/`lost` are named relative
    to the argument order rather than to time.
  **Confuse co-registration with change.** A shift between two dates makes every
    narrow feature differ. The OSM layer already refuses to compare unregistered
    maps (`osm.burn.align_report`); the same gate belongs here before anything is
    believed, and `diff_tile_indices` records `registered=None` to mean nobody
    checked rather than to mean it passed.
"""

from __future__ import annotations

# --- thresholds (guesses; the change work will measure them) ---------------

# Total-variation distance between two tile class histograms, above which the
# tile is worth surfacing at all. Below it, the difference is at the level of
# boundary jitter between two segmentations of the same ground.
# WHERE THIS SHOULD COME FROM: the null test -- run the detector on one date
# against itself with the segmenter re-run, and take the 95th percentile.
MIN_SHIFT = 0.08

# How strongly expected volatility discounts an observed difference.
#
#     excess = shift * (1 - VOLATILITY_CREDIT * expected)
#
# Proportional, not subtractive, and that is a correction rather than a
# preference. The subtractive form (`shift - credit * expected`) gets this
# repo's own canonical example wrong: a tile that flips wholly from
# `DryGrassland` to `GreenGrassland` has shift 0.90 and volatility 0.90 -- it is
# the same ground in two seasons, which `taxonomy` states outright -- and
# subtraction still leaves 0.225 of "excess", enough to surface it as worth a
# look. Proportionally it leaves 0.09 and reads as `expected`, which is the
# right answer. Meanwhile a house block turning to `Clutter` (shift 0.50,
# volatility 0.13) keeps 0.44 either way.
#
# 1.0 means a maximally volatile tile contributes nothing; 0.0 ignores phenology
# altogether, which is the naive detector this module exists to avoid.
VOLATILITY_CREDIT = 1.0

# Excess above which a tile is called worth-a-look.
WORTH_A_LOOK = 0.12

# A class whose volatility is below this is *structural*: if it changed,
# something happened on the ground rather than in the calendar.
STRUCTURAL_VOLATILITY = 0.15

# Structural classes that moved by more than this share of the tile make the
# verdict `structural` regardless of the excess -- a house is a house.
STRUCTURAL_SHIFT = 0.05

VERDICTS = ("quiet", "expected", "worth-a-look", "structural")


def _hist(tile: dict) -> dict[str, float]:
    """A tile's class fractions as a dict. `classes` is a truncated top-N list,
    so the histogram is partial by construction -- which is fine for a distance
    between two of them, and is why the residual mass is tracked explicitly."""
    return {name: frac for name, frac in tile.get("classes", [])}


def _total_variation(a: dict[str, float], b: dict[str, float],
                     ra: float = 0.0, rb: float = 0.0) -> float:
    """A LOWER BOUND on half the L1 distance between two partial histograms.

    `a` and `b` are top-N truncated, and treating a class missing from one side
    as zero there is wrong in the direction that invents change: a class that
    merely fell off the list still holds mass, up to that side's residual. On a
    tile with five near-equal classes, a FOUR PIXEL move that flips the rank-4/5
    tie was scored 0.200 against a true 0.00098 -- a 204x overstatement, and a
    `worth-a-look` verdict for a transition that did not happen.

    So a class listed in `a` and missing from `b` contributes at least
    `max(0, a_i - rb)`, where `rb` is `b`'s unlisted mass. That is a bound, not
    an estimate, and it can only understate change -- which is the safe
    direction for a detector whose job is to avoid crying wolf.
    """
    total = 0.0
    for k in set(a) | set(b):
        if k in a and k in b:
            total += abs(a[k] - b[k])
        elif k in a:
            total += max(0.0, a[k] - rb)
        else:
            total += max(0.0, b[k] - ra)
    return 0.5 * total


def _movers(a: dict[str, float], b: dict[str, float], limit: int = 3):
    """(gained, lost) as [(class, delta)], largest first, relative to a -> b."""
    keys = set(a) | set(b)
    deltas = [(k, b.get(k, 0.0) - a.get(k, 0.0)) for k in keys]
    gained = sorted([d for d in deltas if d[1] > 0], key=lambda d: -d[1])[:limit]
    lost = sorted([d for d in deltas if d[1] < 0], key=lambda d: d[1])[:limit]
    return ([[k, round(v, 3)] for k, v in gained],
            [[k, round(-v, 3)] for k, v in lost])


def _grids_match(a: dict, b: dict) -> str:
    for key in ("tile_px", "cols", "rows"):
        if a.get(key) != b.get(key):
            return (f"{key} differs: {a.get(key)} vs {b.get(key)}")
    if a.get("shape") != b.get("shape"):
        return f"raster shape differs: {a.get('shape')} vs {b.get('shape')}"
    return ""


def diff_tile_indices(a: dict, b: dict, registered: bool | None = None,
                      structural_classes: tuple[str, ...] = ()) -> dict:
    """Per-tile change between two tile indices over the same footprint.

    `a` and `b` come from `tilemap.build_tile_index` at two dates.
    `registered` is the caller's co-registration verdict: True if the two dates
    were checked and line up, False if they do not, None if nobody looked. False
    short-circuits — a misregistered pair produces confident nonsense.

    Returns the grid metadata plus one record per tile:

        {"i", "r", "c",
         "shift":      total-variation distance between the class histograms,
         "expected":   mean change_volatility across the two dates,
         "excess":     shift discounted by expected volatility,
         "structural": share of the shift in low-volatility classes,
         "gained": [[class, delta], ...], "lost": [...],
         "verdict":    quiet | expected | worth-a-look | structural}
    """
    mismatch = _grids_match(a, b)
    if mismatch:
        raise ValueError(
            f"these two tile indices are not comparable -- {mismatch}. Build both "
            f"with the same tile_px over the same footprint; a per-tile diff "
            f"across different grids is arithmetic over unrelated ground.")

    if registered is False:
        return {"tile_px": a["tile_px"], "cols": a["cols"], "rows": a["rows"],
                "shape": a["shape"], "n_tiles": a["n_tiles"], "registered": False,
                "tiles": [],
                "note": ("the two dates are not co-registered, so no per-tile "
                         "comparison was made: a shift between dates makes every "
                         "narrow feature differ, and the result would measure the "
                         "offset rather than the ground")}

    structural = set(structural_classes)
    tiles_a = {t["i"]: t for t in a["tiles"]}
    out = []
    for tb in b["tiles"]:
        ta = tiles_a.get(tb["i"])
        if ta is None:
            continue
        ha, hb = _hist(ta), _hist(tb)
        ra = float(ta.get("classes_other", 0.0))
        rb = float(tb.get("classes_other", 0.0))
        shift = _total_variation(ha, hb, ra, rb)

        # The volatility prior. `s4_products.compute` states that 0.0 is the
        # least safe default here -- it means "any difference is real" about
        # ground nobody has data for -- so a missing product is recorded as
        # unknown rather than silently forgiven.
        va = ta.get("s4", {}).get("change_volatility")
        vb = tb.get("s4", {}).get("change_volatility")
        known = va is not None and vb is not None
        exp = 0.5 * (float(va) + float(vb)) if known else 0.0
        excess = shift * max(1.0 - VOLATILITY_CREDIT * exp, 0.0)
        gained, lost = _movers(ha, hb)

        # The LARGER one-sided move, not half the L1. `STRUCTURAL_SHIFT` is
        # documented as "structural classes that moved by more than this share
        # of the tile"; halving an L1 turned that stated 5% into an effective
        # 10%, so 6% of a tile becoming House read as unremarkable. A one-sided
        # appearance of 6% IS a 6% move, and that is what the constant means.
        struct_share = 0.0
        if structural:
            keys = (set(ha) | set(hb)) & structural
            gained = sum(max(0.0, hb.get(k, 0.0) - ha.get(k, 0.0)) for k in keys)
            lost = sum(max(0.0, ha.get(k, 0.0) - hb.get(k, 0.0)) for k in keys)
            struct_share = max(gained, lost)

        # Structural FIRST. The rule is documented as firing "regardless of the
        # excess"; testing MIN_SHIFT before it made that false, and with
        # STRUCTURAL_SHIFT (0.05) below MIN_SHIFT (0.08) an entirely structural
        # change in [0.05, 0.08) was unreachable. On a 256 px tile at 0.5 m/px
        # that band is 1,310 m2 of new construction reported as `quiet`.
        if struct_share >= STRUCTURAL_SHIFT:
            verdict = "structural"
        elif shift < MIN_SHIFT:
            verdict = "quiet"
        elif excess >= WORTH_A_LOOK:
            verdict = "worth-a-look"
        else:
            verdict = "expected"

        out.append({"i": tb["i"], "r": tb["r"], "c": tb["c"],
                    "shift": round(shift, 3), "expected": round(exp, 3),
                    "excess": round(excess, 3),
                    "structural": round(struct_share, 3),
                    "structural_frac_of_shift": round(
                        struct_share / shift, 3) if shift > 1e-9 else 0.0,
                    "volatility_known": known,
                    "residual": round(max(ra, rb), 3),
                    "gained": gained, "lost": lost, "verdict": verdict})

    out.sort(key=lambda t: (-t["excess"], -t["shift"]))
    counts: dict[str, int] = {}
    for t in out:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    return {"tile_px": a["tile_px"], "cols": a["cols"], "rows": a["rows"],
            "shape": a["shape"], "n_tiles": len(out), "registered": registered,
            "counts": counts, "tiles": out}


def structural_classes_from_taxonomy(threshold: float = STRUCTURAL_VOLATILITY):
    """Classes whose change means something happened, from the volatility table.

    Derived rather than hand-listed, so the two places that decide what is
    volatile cannot drift apart.

    That claim used to be false: this read `s4_products.CLASS_VOLATILITY`, which
    does not exist -- the table is `VOLATILITY` -- so the `AttributeError`
    fallback fired every single time and returned a hand-written list that
    disagreed with the table it claimed to derive from. It called `Car`
    structural, whose volatility is 0.8 because a parked car moving is not a
    change of the ground, and it omitted every rock class, which genuinely is
    structural. A silent fallback that always fires is worse than no fallback,
    so there is no longer one.
    """
    from .solutions.s4_products import VOLATILITY, VOLATILITY_DEFAULT
    from .taxonomy import NAMES

    # Every class, not just the ones the table names: a class the table omits
    # takes VOLATILITY_DEFAULT, and whether that default is below the threshold
    # is a real answer, not a reason to skip it. The rock classes reach this
    # branch, and a limestone terrace turning into rubble is as structural as a
    # house doing it.
    return tuple(sorted(n for n in NAMES
                        if VOLATILITY.get(n, VOLATILITY_DEFAULT) < threshold))


def render(diff: dict, limit: int = 20) -> str:
    """The tile-level change table, in the repo's TSV idiom."""
    if diff.get("registered") is False:
        return "# " + diff.get("note", "not co-registered")
    counts = diff.get("counts", {})
    lines = [
        f"# tile change: {diff['n_tiles']} tiles of {diff['tile_px']} px — "
        + ", ".join(f"{k} {counts.get(k, 0)}" for k in VERDICTS),
        f"# excess = shift x (1 - {VOLATILITY_CREDIT} x expected volatility). A "
        f"tile that was always going to differ has to differ MORE to surface.",
    ]
    if diff.get("registered") is None:
        lines.append("# NOTE: co-registration was not checked. A shift between "
                     "the two dates would read as change on every narrow "
                     "feature.")
    lines.append("rank\ttile\tverdict\tshift\texpected\texcess\tgained\tlost")
    for rank, t in enumerate(diff["tiles"][:limit], 1):
        g = " ".join(f"{k}+{v:.2f}" for k, v in t["gained"]) or "-"
        l = " ".join(f"{k}-{v:.2f}" for k, v in t["lost"]) or "-"
        lines.append(f"{rank}\tr{t['r']}c{t['c']}\t{t['verdict']}\t"
                     f"{t['shift']:.3f}\t{t['expected']:.3f}\t{t['excess']:.3f}\t"
                     f"{g}\t{l}")
    if len(diff["tiles"]) > limit:
        lines.append(f"# NOTE: {len(diff['tiles']) - limit} further tiles omitted "
                     f"by limit={limit}. They are not resolved, only unlisted.")
    return "\n".join(lines)
