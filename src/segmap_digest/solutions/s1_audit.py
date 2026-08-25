"""S1 -- consistency audit / QA of the segmentation.

Takes the raw candidate findings from `audit.py` and turns them into something
a human can act on. Three things this layer is responsible for, each of which
was got wrong on the first real export and is now the reason the module is
shaped the way it is:

  A worklist of DECISIONS, not of rows. A class-level fact is one thing to
    decide about however many regions it touches. The sinai mosaic produced
    41,669 findings and a top-25 worklist that was the same sentence twenty-five
    times; findings are rolled up on `Finding.cause` and the per-region detail
    stays addressable through `detail()`.
  A ranking that discriminates. `audit.py` grades severity from evidence now;
    this module ranks causes by severity x area^AREA_EXPONENT, so a fact that
    covers 12% of the map outranks one that covers a field.
  An explicit statement of what was NOT examined. A short findings list on a
    map with no active checks is not a clean bill of health, and the output has
    to say so in the same breath as the finding count -- see `Coverage`.

Highest-ROI solution of the six, because its output is training-data
corrections and an error map -- not prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audit import (
    SPECK_MAX_PX,
    SPECK_MIN_PX,
    Finding,
    audit,
)
from ..index import RegionIndex
from ..taxonomy import BY_ID, N_CLASSES, PRIORS_BY_SUBJECT

# --- tunable heuristics (all guesses -- see docs/solutions/S1-audit.md) ----

# Review value = severity * area^AREA_EXPONENT. 0 = ignore area entirely,
# 1 = a 10x bigger region is 10x more worth reviewing.
AREA_EXPONENT = 0.5

# A class is called out as a *systematic* problem, not noise, when this
# fraction of its regions is flagged.
SYSTEMATIC_FLAG_RATE = 0.15

# ...and only once it has at least this many regions, so a class with 2
# instances cannot look systematic.
SYSTEMATIC_MIN_REGIONS = 8

# How many region ids to name per rolled-up cause. The rest stay addressable
# through `detail()`; the row says how many were not shown.
# WHERE THIS SHOULD COME FROM: how many examples a reviewer needs to recognise
# a pattern -- 3-5 in every review workflow anyone has described, but unmeasured
# here.
CAUSE_EXAMPLE_RIDS = 5

# --- coverage --------------------------------------------------------------
# Which check families exist, and whether each one tests the LABEL against
# evidence or merely tests the geometry of a component. The distinction matters
# because the geometry checks apply to all 47 classes and therefore make
# coverage look complete when it is not.

SUBSTANTIVE_CHECKS = (
    "context-prior", "slope-prior", "aspect-prior",
    "lithology-isolation", "nari-geometry",
)
GEOMETRY_CHECKS = ("enclosed-speck",)


@dataclass
class RootCause:
    """One class-level fact, and every region that is an instance of it."""
    key: str
    kind: str
    class_name: str
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def n_findings(self) -> int:
        return len(self.findings)

    @property
    def n_regions(self) -> int:
        return len({f.region_id for f in self.findings})

    @property
    def area_m2(self) -> float:
        seen: dict[int, float] = {f.region_id: f.area_m2 for f in self.findings}
        return float(sum(seen.values()))

    @property
    def severity(self) -> float:
        """Area-weighted mean severity: the level at which this fact is
        typically violated, not the level of its loudest instance."""
        w = np.array([max(f.area_m2, 1.0) for f in self.findings])
        s = np.array([f.severity for f in self.findings])
        return float((s * w).sum() / w.sum())

    @property
    def max_severity(self) -> float:
        return max(f.severity for f in self.findings)

    @property
    def value(self) -> float:
        return self.severity * (max(self.area_m2, 1.0) ** AREA_EXPONENT)

    def example_rids(self, n: int = CAUSE_EXAMPLE_RIDS) -> list[int]:
        ordered = sorted(self.findings, key=review_value, reverse=True)
        out, seen = [], set()
        for f in ordered:
            if f.region_id not in seen:
                seen.add(f.region_id)
                out.append(f.region_id)
            if len(out) >= n:
                break
        return out


@dataclass
class Coverage:
    """What the audit was actually able to look at.

    Silence from a check that never ran is not evidence of correctness, and
    this is the object that refuses to let the report imply otherwise."""
    has_terrain: bool
    min_area_m2: float
    px_area_m2: float
    total_area_m2: float
    class_area: dict[int, float]
    checks: dict[int, list[str]]          # class_id -> active check names
    caveats: list[str]                    # each already labelled ABSTAINED / SCOPE

    @property
    def classes_present(self) -> list[int]:
        return sorted(self.class_area, key=lambda c: -self.class_area[c])

    def substantive(self, class_id: int) -> list[str]:
        return [c for c in self.checks.get(class_id, ()) if c in SUBSTANTIVE_CHECKS]

    @property
    def n_checked(self) -> int:
        return sum(1 for c in self.classes_present if self.substantive(c))

    @property
    def area_checked(self) -> float:
        return sum(self.class_area[c] for c in self.classes_present
                   if self.substantive(c))

    @property
    def area_fraction(self) -> float:
        return self.area_checked / max(self.total_area_m2, 1e-9)

    def unchecked(self) -> list[tuple[int, float]]:
        return [(c, self.class_area[c]) for c in self.classes_present
                if not self.substantive(c)]


@dataclass
class AuditReport:
    findings: list[Finding]
    n_regions: int
    by_class: dict[int, dict] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)
    causes: list[RootCause] = field(default_factory=list)
    coverage: Coverage | None = None
    fragmentation: dict = field(default_factory=dict)

    @property
    def systematic(self) -> list[tuple[str, dict]]:
        out = [
            (BY_ID[c].name, s) for c, s in self.by_class.items()
            if s["n_regions"] >= SYSTEMATIC_MIN_REGIONS
            and s["flag_rate"] >= SYSTEMATIC_FLAG_RATE
        ]
        return sorted(out, key=lambda kv: -kv[1]["flag_rate"])

    def cause(self, key: str) -> RootCause | None:
        for c in self.causes:
            if c.key == key:
                return c
        return None


def review_value(f: Finding) -> float:
    """What a reviewer's next minute is best spent on."""
    return f.severity * (max(f.area_m2, 1.0) ** AREA_EXPONENT)


def roll_up(findings: list[Finding]) -> list[RootCause]:
    """N findings -> the distinct facts they are instances of, ranked."""
    causes: dict[str, RootCause] = {}
    for f in findings:
        rc = causes.get(f.cause)
        if rc is None:
            rc = causes[f.cause] = RootCause(f.cause, f.kind, f.class_name,
                                             f.cause_text)
        rc.findings.append(f)
    return sorted(causes.values(), key=lambda c: -c.value)


def _coverage(ridx: RegionIndex, min_area_m2: float) -> Coverage:
    class_area: dict[int, float] = {}
    for r in ridx.regions:
        class_area[r.class_id] = class_area.get(r.class_id, 0.0) + r.area_m2

    px_area = 0.0
    for r in ridx.regions:
        if r.area_px:
            px_area = r.area_m2 / r.area_px
            break

    caveats: list[str] = []
    if not ridx.has_terrain:
        caveats.append(
            "ABSTAINED -- no DEM: every mean_slope and aspect_circvar is 0.0, "
            "which is a valid-looking number meaning 'unmeasured'. The slope and "
            "aspect priors did not run rather than score against it."
        )
    if px_area:
        caveats.append(
            f"SCOPE -- the speck window is {SPECK_MIN_PX}-{SPECK_MAX_PX} resolution "
            f"cells, {SPECK_MIN_PX * px_area:.1f}-{SPECK_MAX_PX * px_area:.1f} m2 on "
            f"this raster. Nothing larger was tested for being an enclosed island."
        )
    caveats.append(
        f"SCOPE -- every check ran only on regions >= min_area {min_area_m2:g} m2 "
        f"({min_area_m2 / px_area:.0f} cells here), except enclosed-speck, which "
        f"is the check about components below that gate."
        if px_area else
        f"SCOPE -- every check ran only on regions >= min_area {min_area_m2:g} m2."
    )

    checks: dict[int, list[str]] = {}
    for c in class_area:
        active: list[str] = []
        name = BY_ID[c].name
        for p in PRIORS_BY_SUBJECT.get(name, ()):
            if p.others:
                active.append("context-prior")
            if p.slope_deg is not None and ridx.has_terrain:
                active.append("slope-prior")
            if p.aspect_variance_max is not None and ridx.has_terrain:
                active.append("aspect-prior")
        if BY_ID[c].lithology:
            active.append("lithology-isolation")
        if BY_ID[c].lithology == "Nari":
            active.append("nari-geometry")
        active.append("enclosed-speck")
        checks[c] = sorted(set(active))

    return Coverage(
        has_terrain=ridx.has_terrain, min_area_m2=min_area_m2,
        px_area_m2=px_area, total_area_m2=float(sum(class_area.values())),
        class_area=class_area, checks=checks, caveats=caveats,
    )


def _fragmentation(ridx: RegionIndex) -> dict:
    """How much of this map is salt-and-pepper.

    This is the statistic the `isolated-speck` check used to emit one row at a
    time. It is a property of the segmenter's output smoothness, not a list of
    label errors, so it is reported once."""
    if not ridx.regions:
        return {}
    px = np.array([r.area_px for r in ridx.regions])
    area = np.array([r.area_m2 for r in ridx.regions])
    small = px < SPECK_MIN_PX
    return {
        "n_indexed": int(px.size),
        "n_below_speck_floor": int(small.sum()),
        "frac_below_speck_floor": float(small.mean()),
        "area_frac_below_speck_floor": float(area[small].sum() / max(area.sum(), 1e-9)),
        "median_px": float(np.median(px)),
    }


def run(ridx: RegionIndex, min_area_m2: float = 25.0) -> AuditReport:
    findings = audit(ridx, min_area_m2=min_area_m2)
    findings.sort(key=review_value, reverse=True)

    per_class_total = np.zeros(N_CLASSES, dtype=int)
    for r in ridx.regions:
        per_class_total[r.class_id] += 1

    flagged: dict[int, set[int]] = {}
    sev: dict[int, list[float]] = {}
    for f in findings:
        c = ridx.get(f.region_id).class_id
        flagged.setdefault(c, set()).add(f.region_id)
        sev.setdefault(c, []).append(f.severity)

    by_class = {}
    for c, rids in flagged.items():
        n = int(per_class_total[c])
        by_class[c] = {
            "n_regions": n,
            "n_flagged": len(rids),
            "flag_rate": len(rids) / max(n, 1),
            "mean_severity": float(np.mean(sev[c])),
        }

    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1

    return AuditReport(
        findings, len(ridx.regions), by_class, by_kind,
        causes=roll_up(findings),
        coverage=_coverage(ridx, min_area_m2),
        fragmentation=_fragmentation(ridx),
    )


def worklist(report: AuditReport, budget: int = 25) -> str:
    """The N distinct things a reviewer should decide about first.

    Rows are ROOT CAUSES, not findings. One row can stand for thousands of
    regions; `n_regions` and `area_m2` say how many and how much, and
    `detail(report, cause)` lists them."""
    rows = report.causes[:budget]
    total_area = report.coverage.total_area_m2 if report.coverage else 0.0
    lines = [
        f"# review worklist: top {len(rows)} of {len(report.causes)} distinct root "
        f"causes covering {len(report.findings)} findings, ranked by "
        f"severity x area^{AREA_EXPONENT}",
        "# one row = one class-level decision, not one region. `sev` is the "
        "area-weighted mean severity of its members; severity is ordinal "
        "WITHIN a kind and is not a calibrated error probability.",
        "rank\tcause\tkind\tclass\tn_regions\tarea_m2\tpct_map\tsev\tmax_sev\t"
        "example_rids\tmessage",
    ]
    for i, c in enumerate(rows, 1):
        ex = c.example_rids()
        ex_txt = ",".join(str(r) for r in ex)
        if c.n_regions > len(ex):
            ex_txt += f" (+{c.n_regions - len(ex)} more)"
        pct = c.area_m2 / total_area if total_area else 0.0
        lines.append(
            f"{i}\t{c.key}\t{c.kind}\t{c.class_name}\t{c.n_regions}\t"
            f"{c.area_m2:.0f}\t{pct:.2%}\t{c.severity:.2f}\t{c.max_severity:.2f}\t"
            f"{ex_txt}\t{c.text}"
        )
    if len(report.causes) > budget:
        dropped = report.causes[budget:]
        n_f = sum(c.n_findings for c in dropped)
        a = sum(c.area_m2 for c in dropped)
        lines.append(
            f"# NOTE: {len(dropped)} lower-value root causes omitted by "
            f"budget={budget} -- {n_f} findings over {a:.0f} m2. They are not "
            f"resolved, only unlisted."
        )
    return "\n".join(lines)


def detail(report: AuditReport, cause_key: str, budget: int = 50) -> str:
    """The per-region rows behind one rolled-up cause."""
    rc = report.cause(cause_key)
    if rc is None:
        return f"# no such root cause: {cause_key}"
    rows = sorted(rc.findings, key=review_value, reverse=True)[:budget]
    lines = [
        f"# {cause_key} -- {rc.n_regions} regions, {rc.area_m2:.0f} m2",
        f"# {rc.text}",
        "rank\trid\tclass\tkind\tsev\tarea_m2\tvalue\tmessage",
    ]
    for i, f in enumerate(rows, 1):
        lines.append(
            f"{i}\t{f.region_id}\t{f.class_name}\t{f.kind}\t{f.severity:.2f}\t"
            f"{f.area_m2:.0f}\t{review_value(f):.0f}\t{f.message}"
        )
    if rc.n_findings > budget:
        lines.append(
            f"# NOTE: {rc.n_findings - budget} further findings under this cause "
            f"omitted by budget={budget}"
        )
    return "\n".join(lines)


def coverage_block(report: AuditReport) -> str:
    """What was examined, and -- louder -- what was not."""
    cov = report.coverage
    if cov is None:
        return ""
    n_present = len(cov.classes_present)
    parts = [
        "## coverage -- what was actually examined",
        f"# {cov.n_checked} of {n_present} classes present on this map have at "
        f"least one active substantive check "
        f"({len(SUBSTANTIVE_CHECKS)} check families exist; 47 classes are defined).",
        f"# those classes hold {cov.area_fraction:.1%} of classified region area.",
        "#",
        "# A CLASS WITH NO ACTIVE CHECK IS UNEXAMINED, NOT CLEAN. This audit can "
        "only contradict a label where a prior, a lithology relation or a "
        "geometry rule gives it something to contradict it with. For the "
        f"{n_present - cov.n_checked} classes below it had nothing, so no "
        "number of findings -- including zero -- says anything about them.",
    ]
    for why in cov.caveats:
        parts.append(f"# {why}")

    parts += ["", "# `substantive` tests the LABEL against evidence. `geometry_only` "
                  "applies to all 47 classes and tests the shape of a component, "
                  "not whether its class is right -- it is not coverage.",
              "class\tarea_m2\tpct_map\tsubstantive\tgeometry_only"]
    for c in cov.classes_present:
        a = cov.class_area[c]
        sub = cov.substantive(c)
        geom = [k for k in cov.checks.get(c, ()) if k not in SUBSTANTIVE_CHECKS]
        parts.append(
            f"{BY_ID[c].name}\t{a:.0f}\t{a / max(cov.total_area_m2, 1e-9):.2%}\t"
            f"{','.join(sub) if sub else 'NONE -- UNEXAMINED'}\t"
            f"{','.join(geom) or '-'}"
        )

    unchecked = cov.unchecked()
    if unchecked:
        top = unchecked[:5]
        share = sum(a for _, a in unchecked) / max(cov.total_area_m2, 1e-9)
        parts.append(
            f"# NOTE: {share:.1%} of classified area is in classes with no "
            f"substantive check. Largest: "
            + ", ".join(f"{BY_ID[c].name} ({a / max(cov.total_area_m2, 1e-9):.1%})"
                        for c, a in top)
        )
    return "\n".join(parts)


def fragmentation_block(report: AuditReport) -> str:
    fr = report.fragmentation
    if not fr:
        return ""
    lines = [
        "## fragmentation (a property of the raster, not a list of errors)",
        f"# {fr['n_below_speck_floor']} of {fr['n_indexed']} indexed regions "
        f"({fr['frac_below_speck_floor']:.1%}) are below the {SPECK_MIN_PX}-cell "
        f"speck floor, carrying {fr['area_frac_below_speck_floor']:.2%} of "
        f"indexed area; median region is {fr['median_px']:.0f} cells.",
        "# This is salt-and-pepper, and it is stated once here instead of "
        "emitted as one finding per component. Components below the index's own "
        "min_area_px are not counted at all -- the true fragmentation rate is "
        "higher than this line.",
    ]
    n_speck = report.by_kind.get("isolated-speck", 0)
    if n_speck and report.findings:
        lines.append(
            f"# {n_speck / len(report.findings):.0%} of findings are "
            f"isolated-speck, i.e. the band just above that floor. Read them as "
            f"a measurement of boundary smoothness per class pair, not as a list "
            f"of label errors -- which is what the roll-up turns them into."
        )
    return "\n".join(lines)


def render(report: AuditReport, budget: int = 25) -> str:
    cov = report.coverage
    head = (f"# S1 consistency audit -- {report.n_regions} regions, "
            f"{len(report.causes)} root causes, "
            f"{len(report.findings)} candidate findings")
    if cov is not None:
        head += (f"\n# coverage: {cov.n_checked}/{len(cov.classes_present)} "
                 f"classes checked, {cov.area_fraction:.1%} of area. Read the "
                 f"coverage block before reading the findings.")
    parts = [head, "", coverage_block(report), "", fragmentation_block(report),
             "", "## by kind", "kind\tcount"]
    for k, v in sorted(report.by_kind.items(), key=lambda kv: -kv[1]):
        parts.append(f"{k}\t{v}")
    if not report.by_kind:
        parts.append("# none -- see the coverage block for what that does and "
                     "does not mean")

    parts += ["", "## systematic (a class-level problem, not noise)"]
    if report.systematic:
        parts.append("class\tn_regions\tn_flagged\tflag_rate\tmean_sev")
        for name, s in report.systematic:
            parts.append(
                f"{name}\t{s['n_regions']}\t{s['n_flagged']}\t"
                f"{s['flag_rate']:.0%}\t{s['mean_severity']:.2f}"
            )
    else:
        parts.append(
            f"none -- no class has >={SYSTEMATIC_FLAG_RATE:.0%} of its "
            f">={SYSTEMATIC_MIN_REGIONS} regions flagged. This is a statement "
            f"about the classes that were checked only."
        )

    parts += ["", worklist(report, budget)]
    return "\n".join(parts)
