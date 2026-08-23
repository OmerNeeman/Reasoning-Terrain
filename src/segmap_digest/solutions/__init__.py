"""The six things a reasoning layer can do with a Smart Terrain segmentation map.

Shared infrastructure lives one level up (taxonomy, index, digests). Each
solution here is a thin, runnable naive implementation on top of it, so you can
see the shape of the answer before arguing with the heuristics.

    S1  audit        consistency / QA of the segmentation itself
    S2  adjudicate   pick among a short candidate list for a suspect region
    S3  triage       choose which chips get sent to an expensive detector
    S4  products     trafficability, concealment, drainage, fire fuel
    S5  query        answer questions over the index; hand the rest to an LLM
    S6  change       semantic diff between two dates

EVERY HEURISTIC CONSTANT IN THIS PACKAGE IS A GUESS. They are grouped at the top
of each module, named, and documented in docs/solutions/S*.md alongside the open
questions. Replace them with measured values; do not treat them as findings.
"""

from . import s1_audit, s2_adjudicate, s3_triage, s4_products, s5_query, s6_change  # noqa: F401

SOLUTIONS = {
    "s1": ("audit", "consistency / QA of the segmentation", s1_audit),
    "s2": ("adjudicate", "candidate shortlist for a suspect region", s2_adjudicate),
    "s3": ("triage", "which chips to send to an expensive detector", s3_triage),
    "s4": ("products", "trafficability / concealment / drainage / fire", s4_products),
    "s5": ("query", "structured queries over the index", s5_query),
    "s6": ("change", "semantic diff between two dates", s6_change),
}
