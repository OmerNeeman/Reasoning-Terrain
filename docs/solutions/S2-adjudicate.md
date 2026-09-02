# S2 — Confusion adjudication

`segmap solve s2 [--region N]` · [`s2_adjudicate.py`](../../src/segmap_digest/solutions/s2_adjudicate.py)

**What it is.** S1 says "region 245 looks wrong." S2 answers "then what is it?" —
by scoring a short candidate list on three evidence terms and refusing to answer
when the evidence can't separate the candidates.

**The refusal is the feature.** Limestone / Dolomite / Nari rocky terrain are
not separable in RGB even for a human expert. A system that confidently picks
one is manufacturing corruptions. `UNDECIDABLE-NEEDS-hardness-class` is a
first-class verdict here, and the design goal is that it fires *often*.

---

## Naive algorithm (implemented)

```
candidates = {incumbent} ∪ {c : class_distance(incumbent, c) ≤ 0.5}
                        ∪ {3 dominant neighbour classes}          # never all 45

for each candidate:
    context    = prior fit (expects/contradicts) + lithology continuity
    morphology = slope band fit × aspect-coherence gate
    geometry   = area band + linearity
    total      = 0.45·context + 0.35·morphology + 0.20·geometry

if top two differ ONLY by lithology and are within 0.08  → UNDECIDABLE
elif challenger − incumbent ≥ 0.15                       → SWITCH
else                                                     → KEEP
```

The asymmetry is deliberate: `SWITCH_MARGIN` (0.15) is large because the cost of
a corruption exceeds the value of a correction. **Net gain = corrections −
corruptions**, and that quantity can easily be negative.

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `W_CONTEXT / W_MORPHOLOGY / W_GEOMETRY` | .45 / .35 / .20 | Fit on adjudicated regions; or drop to logistic regression on the three terms |
| `CANDIDATE_MAX_DISTANCE` | 0.5 | Recall check: how often is the true class outside the shortlist? |
| `SWITCH_MARGIN` | 0.15 | Set from the corruption:correction cost ratio you'll accept |
| `UNDECIDABLE_MARGIN` | 0.08 | Should be *higher* than it looks right — err toward abstention |
| `MORPHOLOGY_SLOPE` | 8 bands | **A geomorphologist.** Biggest single error source in the module |
| `AREA_BAND` | 3 classes | Measurable directly from labelled data — do this first, it's free |
| `LINEAR_MIN_ELONGATION` | 2.0 | Measure the elongation distribution of true road regions |

---

## Open questions

1. **What replaces the hand-weighted score?** Three terms with hand-set weights
   is a placeholder for a calibrated model. With ~500 adjudicated regions, a
   logistic regression over the same features would be better *and* give
   calibrated probabilities, which is what the SWITCH margin actually needs.
2. **Where does the segmenter's own confidence go?** Absent. Per-pixel logits or
   a validation confusion matrix would make the candidate list evidence-based
   instead of taxonomy-derived. **Highest-value missing input.**
3. **Does the shortlist contain the truth?** If the true class is outside
   `CANDIDATE_MAX_DISTANCE`, no weighting saves you. Measure shortlist recall
   before tuning anything else.
4. **Should UNDECIDABLE be per-evidence-type?** Currently only lithology triggers
   it. Terra rossa vs Clayeysoil is arguably just as unresolvable without a soil
   map; Batha vs Garigue may be unresolvable without canopy height.
5. **Is a region the right granularity?** A single component may span a real
   contact between two units — the right answer could be "split this region",
   which the current verdict vocabulary can't express.
6. **How does an accepted SWITCH propagate?** Nothing writes back today. Does it
   edit the raster, emit a correction layer, or queue a human confirmation?
7. **Should S2 ever run on unflagged regions?** Running it everywhere gives a
   whole-map second opinion — and multiplies the corruption risk by the number
   of correct regions you touch.

## Options and extensions

- **Ancillary-data joins, in value order:** geological map (settles the
  lithology cases outright — the largest single lever), soil map, canopy height
  (settles the vegetation series), acquisition date (settles irrigated vs
  unirrigated).
- **Abstain-by-default mode.** Only emit SWITCH when a named ancillary layer
  supports it; otherwise UNDECIDABLE. Lower yield, near-zero corruption.
- **LLM as the adjudicator, code as the evidence-gatherer.** Hand the model the
  region row, the neighbour histogram, the candidate list and the per-term
  scores, and let it write the verdict and rationale. Keep the arithmetic here.
- **Multi-lens verification.** For each proposed SWITCH, run independent checks
  (context / morphology / geometry) as separate verifiers and require a
  majority. Catches the case where one strong term drags a bad candidate up.
- **Split/merge verdicts.** Extend the vocabulary beyond KEEP/SWITCH/UNDECIDABLE.

## How to evaluate

The **override ledger** (`s2_adjudicate.ledger`), not accuracy:

| Outcome | Meaning |
|---|---|
| correction | wrong → right |
| corruption | right → wrong |
| **net gain** | corrections − corruptions. **If negative, S2 is hurting you.** |
| abstention rate | share of UNDECIDABLE — should be high on lithology cases |
| shortlist recall | how often the truth is in the candidate list at all |

Report net gain **per class group** — it will be strongly positive for
geometry-constrained classes (Car, House, roads) and near zero or negative for
the lithology grid. That split, not the aggregate, is the decision.
