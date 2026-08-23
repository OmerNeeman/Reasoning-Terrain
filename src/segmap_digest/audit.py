"""Deterministic consistency audit over the region index.

This is the cheap half of the reasoning layer. It applies the co-occurrence and
morphology priors in `taxonomy.PRIORS` -- world knowledge the segmenter never
had, because it learned "TerraRosa" as integer 15 and knows nothing about
decalcification clay.

The output is a *candidate* list. Deciding which candidates are real errors is
the LLM's job; enumerating them is not, and paying an LLM to enumerate them
would defeat the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import Region, RegionIndex
from .taxonomy import (
    BY_ID,
    PRIORS_BY_SUBJECT,
    SUPERCLASS_OF,
    cid,
    class_distance,
)


@dataclass
class Finding:
    region_id: int
    class_name: str
    kind: str
    severity: float            # 0..1
    area_m2: float
    message: str

    def as_row(self) -> str:
        return (f"{self.region_id}\t{self.class_name}\t{self.kind}\t"
                f"{self.severity:.2f}\t{self.area_m2:.0f}\t{self.message}")


def audit(ridx: RegionIndex, min_area_m2: float = 25.0) -> list[Finding]:
    findings: list[Finding] = []
    for r in ridx.regions:
        if r.area_m2 < min_area_m2:
            continue
        findings += _check_priors(ridx, r)
        findings += _check_lithology_context(ridx, r)
        findings += _check_nari_geometry(ridx, r)
        findings += _check_inclusion(ridx, r)
    findings.sort(key=lambda f: (-f.severity, -f.area_m2))
    return findings


def _check_priors(ridx: RegionIndex, r: Region) -> list[Finding]:
    out: list[Finding] = []
    for p in PRIORS_BY_SUBJECT.get(r.class_name, ()):
        nh = ridx.neighbor_class_hist(r)
        total = sum(nh.values()) or 1.0

        if p.kind == "contradicts" and p.others:
            bad = sum(nh.get(cid(n), 0.0) for n in p.others)
            if bad / total > 0.4:
                out.append(Finding(
                    r.id, r.class_name, "contradicted-context",
                    min(1.0, 0.5 + bad / total / 2), r.area_m2,
                    f"{bad / total:.0%} of boundary is with "
                    f"{{{', '.join(p.others)}}} -- {p.why}",
                ))

        if p.kind == "expects" and p.others:
            good = sum(nh.get(cid(n), 0.0) for n in p.others)
            if good / total < 0.05:
                out.append(Finding(
                    r.id, r.class_name, "missing-expected-context", 0.45, r.area_m2,
                    f"no boundary with any of {{{', '.join(p.others)}}} -- {p.why}",
                ))

        if p.slope_deg is not None:
            lo, hi = p.slope_deg
            if not (lo <= r.mean_slope <= hi):
                out.append(Finding(
                    r.id, r.class_name, "slope-violation", 0.8, r.area_m2,
                    f"mean slope {r.mean_slope:.1f} deg outside required "
                    f"{lo}-{hi} deg -- {p.why}",
                ))

        if p.aspect_variance_max is not None and r.aspect_circvar > p.aspect_variance_max:
            out.append(Finding(
                r.id, r.class_name, "aspect-incoherent", 0.85, r.area_m2,
                f"aspect circular variance {r.aspect_circvar:.2f} exceeds "
                f"{p.aspect_variance_max} -- {p.why}",
            ))
    return out


def _check_lithology_context(ridx: RegionIndex, r: Region) -> list[Finding]:
    """A rock unit whose neighbours are all a *different* lithology is the
    classic RGB-blind confusion: limestone/dolomite/nari are near-identical in
    colour, and only the geological map separates them."""
    d = BY_ID[r.class_id]
    if not d.lithology:
        return []
    nh = ridx.neighbor_class_hist(r)
    total = sum(nh.values()) or 1.0
    other = 0.0
    litho_share: dict[str, float] = {}
    for k, v in nh.items():
        kl = BY_ID[k].lithology
        if kl is None:
            continue
        litho_share[kl] = litho_share.get(kl, 0.0) + v
        if kl != d.lithology:
            other += v
    if other / total > 0.85 and litho_share:
        dom = max(litho_share, key=litho_share.get)
        if dom != d.lithology:
            return [Finding(
                r.id, r.class_name, "isolated-lithology", 0.6, r.area_m2,
                f"{other / total:.0%} of boundary is with {dom} units; a "
                f"{d.lithology} island inside {dom} is usually a lithology "
                f"confusion, not a real contact (unresolvable in RGB -- needs the "
                f"geological map)",
            )]
    return []


def _check_nari_geometry(ridx: RegionIndex, r: Region) -> list[Finding]:
    """Nari is a calcrete crust: it caps units. Expect thin high bands, not
    large low-lying blobs."""
    if BY_ID[r.class_id].lithology != "Nari":
        return []
    if r.area_m2 > 20000 and r.compactness > 0.35:
        return [Finding(
            r.id, r.class_name, "nari-morphology", 0.55, r.area_m2,
            f"large ({r.area_m2:.0f} m2) and blocky (compactness "
            f"{r.compactness:.2f}); nari caps units and should appear as thin "
            f"plateau-edge bands",
        )]
    return []


def _check_inclusion(ridx: RegionIndex, r: Region) -> list[Finding]:
    """A small speck of one class fully enclosed by a semantically distant class
    -- salt-and-pepper noise, or a real out-of-vocabulary object."""
    if r.area_m2 > 400 or not r.neighbors:
        return []
    nh = ridx.neighbor_class_hist(r)
    total = sum(nh.values()) or 1.0
    host, share = max(nh.items(), key=lambda kv: kv[1])
    if share / total < 0.9:
        return []
    dist = class_distance(r.class_id, host)
    if dist < 0.7:
        return []
    if SUPERCLASS_OF[r.class_id] == "artifact":
        return [Finding(
            r.id, r.class_name, "oov-candidate", 0.5, r.area_m2,
            f"small {r.class_name} speck enclosed by {BY_ID[host].name} -- by "
            f"definition outside the 45-class vocabulary; prime target for a "
            f"detector pass",
        )]
    return [Finding(
        r.id, r.class_name, "isolated-speck", 0.35 + 0.3 * dist, r.area_m2,
        f"{r.area_m2:.0f} m2 island fully enclosed by {BY_ID[host].name} "
        f"(class distance {dist:.2f})",
    )]


def to_tsv(findings: list[Finding], limit: int | None = None) -> str:
    rows = findings[:limit] if limit else findings
    lines = [
        "# consistency audit -- CANDIDATES, not confirmed errors",
        "rid\tclass\tkind\tseverity\tarea_m2\tmessage",
    ]
    lines += [f.as_row() for f in rows]
    if limit and len(findings) > limit:
        lines.append(f"# NOTE: {len(findings) - limit} lower-severity findings omitted")
    return "\n".join(lines)


def summary(findings: list[Finding]) -> str:
    if not findings:
        return "no findings"
    kinds: dict[str, int] = {}
    for f in findings:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    return f"{len(findings)} findings ({parts})"
