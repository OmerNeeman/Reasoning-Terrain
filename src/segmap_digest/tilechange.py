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


def _total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Half the L1 distance between two partial histograms, in [0, 1].

    Classes absent from one side count as zero there, which is the correct
    reading: the top-N truncation means a class that fell out of the list is
    below the smallest listed fraction, not necessarily gone.
    """
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


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
        shift = _total_variation(ha, hb)
        exp = 0.5 * (float(ta.get("s4", {}).get("change_volatility", 0.0))
                     + float(tb.get("s4", {}).get("change_volatility", 0.0)))
        excess = shift * max(1.0 - VOLATILITY_CREDIT * exp, 0.0)
        gained, lost = _movers(ha, hb)

        struct_share = 0.0
        if structural:
            keys = set(ha) | set(hb)
            struct_share = sum(abs(hb.get(k, 0.0) - ha.get(k, 0.0))
                               for k in keys if k in structural) * 0.5

        if shift < MIN_SHIFT:
            verdict = "quiet"
        elif struct_share >= STRUCTURAL_SHIFT:
            verdict = "structural"
        elif excess >= WORTH_A_LOOK:
            verdict = "worth-a-look"
        else:
            verdict = "expected"

        out.append({"i": tb["i"], "r": tb["r"], "c": tb["c"],
                    "shift": round(shift, 3), "expected": round(exp, 3),
                    "excess": round(excess, 3),
                    "structural": round(struct_share, 3),
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
    volatile cannot drift apart. Falls back to a named list if the product does
    not expose its table yet -- this module ships ahead of the change work.
    """
    try:
        from .solutions.s4_products import CLASS_VOLATILITY

        return tuple(sorted(n for n, v in CLASS_VOLATILITY.items()
                            if v < threshold))
    except (ImportError, AttributeError):
        return ("House", "BrickWall", "Pavement", "PavedRoad", "DirtRoad",
                "DirtRoadB", "Car")


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
