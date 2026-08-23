"""S1 -- consistency audit / QA of the segmentation.

Takes the raw candidate findings from `audit.py` and turns them into something
a human can act on: ranked by review value, grouped by class so systematic
failures separate from one-off noise, and with an explicit review budget so the
output is a worklist rather than a wall.

Highest-ROI solution of the six, because its output is training-data
corrections and an error map -- not prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audit import Finding, audit
from ..index import RegionIndex
from ..taxonomy import BY_ID, N_CLASSES

# --- tunable heuristics (all guesses -- see docs/solutions/S1.md) ----------

# Review value = severity * area^AREA_EXPONENT. 0 = ignore area entirely,
# 1 = a 10x bigger region is 10x more worth reviewing.
AREA_EXPONENT = 0.5

# A class is called out as a *systematic* problem, not noise, when this
# fraction of its regions is flagged.
SYSTEMATIC_FLAG_RATE = 0.15

# ...and only once it has at least this many regions, so a class with 2
# instances cannot look systematic.
SYSTEMATIC_MIN_REGIONS = 8


@dataclass
class AuditReport:
    findings: list[Finding]
    n_regions: int
    by_class: dict[int, dict] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def systematic(self) -> list[tuple[str, dict]]:
        out = [
            (BY_ID[c].name, s) for c, s in self.by_class.items()
            if s["n_regions"] >= SYSTEMATIC_MIN_REGIONS
            and s["flag_rate"] >= SYSTEMATIC_FLAG_RATE
        ]
        return sorted(out, key=lambda kv: -kv[1]["flag_rate"])


def review_value(f: Finding) -> float:
    """What a reviewer's next minute is best spent on."""
    return f.severity * (max(f.area_m2, 1.0) ** AREA_EXPONENT)


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

    return AuditReport(findings, len(ridx.regions), by_class, by_kind)


def worklist(report: AuditReport, budget: int = 25) -> str:
    """The N findings a reviewer should look at first."""
    rows = report.findings[:budget]
    lines = [
        f"# review worklist: top {len(rows)} of {len(report.findings)} findings, "
        f"ranked by severity x area^{AREA_EXPONENT}",
        "rank\trid\tclass\tkind\tsev\tarea_m2\tvalue\tmessage",
    ]
    for i, f in enumerate(rows, 1):
        lines.append(
            f"{i}\t{f.region_id}\t{f.class_name}\t{f.kind}\t{f.severity:.2f}\t"
            f"{f.area_m2:.0f}\t{review_value(f):.0f}\t{f.message}"
        )
    if len(report.findings) > budget:
        lines.append(
            f"# NOTE: {len(report.findings) - budget} lower-value findings omitted "
            f"by budget={budget}"
        )
    return "\n".join(lines)


def render(report: AuditReport, budget: int = 25) -> str:
    parts = [
        f"# S1 consistency audit -- {report.n_regions} regions, "
        f"{len(report.findings)} candidate findings",
        "",
        "## by kind",
        "kind\tcount",
    ]
    for k, v in sorted(report.by_kind.items(), key=lambda kv: -kv[1]):
        parts.append(f"{k}\t{v}")

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
            f">={SYSTEMATIC_MIN_REGIONS} regions flagged"
        )

    parts += ["", "## " + worklist(report, budget).split("\n", 1)[0].lstrip("# "),
              worklist(report, budget).split("\n", 1)[1]]
    return "\n".join(parts)
