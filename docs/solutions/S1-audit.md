# S1 — Consistency audit / QA of the segmentation

`segmap solve s1` · [`s1_audit.py`](../../src/segmap_digest/solutions/s1_audit.py) · engine in [`audit.py`](../../src/segmap_digest/audit.py)

**What it is.** Flag regions whose label contradicts their context, their slope,
or their geometry, and rank them by what a reviewer's next minute is best spent
on. Output is a worklist, not prose.

**Why it's first.** It's the only solution whose output feeds back into the
segmenter: confirmed findings become training-data corrections, and the flag
rate per class is an error map you can act on without any detection ground
truth existing.

---

## Naive algorithm (implemented)

```
for each region ≥ min_area:
    check co-occurrence priors     (expects / contradicts, boundary-share weighted)
    check slope band               (badlands need relief, hydromorphic needs flat)
    check aspect coherence         (a dip slope is one facet or it isn't)
    check lithology isolation      (a Limestone island inside Dolomite)
    check nari geometry            (calcrete caps: thin high bands, not blobs)
    check enclosed specks          (salt-and-pepper, or an OOV object)
rank by severity × area^0.5
group by class → flag_rate → "systematic vs noise"
```

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `AREA_EXPONENT` | 0.5 | Reviewer time-motion: is a 100× bigger error worth 10× the attention, or 100×? |
| `SYSTEMATIC_FLAG_RATE` | 0.15 | The per-class flag rate on a *known-good* tile — that's the noise floor |
| `SYSTEMATIC_MIN_REGIONS` | 8 | Small-sample guard; set from the binomial CI you're willing to act on |
| boundary-share thresholds (0.4 contradicts / 0.05 expects) in `audit.py` | 0.4 / 0.05 | Sweep against reviewed regions |
| prior slope bands | see `taxonomy.PRIORS` | **A geologist, not a sweep** |

---

## Open questions

1. **What's the false-positive rate?** Right now: unknown. Every number this
   emits is a candidate. Until a reviewer labels a sample, "2 findings" and "200
   findings" are equally uninterpretable. **This is the blocking question.**
2. **Is a "region" the right unit?** Connected components fragment badly — 1,583
   of them in the 1024px fixture. A boundary wobble creates two findings where a
   human sees one problem. Alternatives: merge adjacent same-class components
   across thin gaps; work on superpixels; work at the chip level.
3. **Should severity be calibrated or ordinal?** Currently a made-up 0–1 float
   that gets multiplied by area. If severities aren't comparable across check
   types, the ranking is arbitrary.
4. **Which checks actually earn their place?** Six check types ship here. Some
   probably fire constantly and mean nothing. Precision per check type is the
   number to measure first.
5. **How do priors get maintained?** They're Python dataclasses today. If a
   geologist owns them they need to be a data file with review history.
6. **Does the audit run per tile or per AOI?** Priors like "nari caps units" are
   regional statements; a tile crop can make a legitimate unit look isolated.

## Options and extensions

- **Cheap win — a class-confusion prior from the segmenter itself.** If you have
  the model's per-pixel logits or a validation confusion matrix, "what else could
  this be" stops being a guess. This is the single largest available improvement
  and needs no new data collection.
- **Boundary-quality check.** Flag regions whose boundary is unusually ragged
  relative to others of the same class — a proxy for low-confidence pixels
  without needing logits.
- **Temporal consistency as a free labeller.** A region whose class flips between
  dates but shouldn't (see S6 `impossible`) is a confirmed error with no
  reviewer involved.
- **Feed findings to an LLM in batches, not one at a time.** The model is good
  at spotting that 40 findings share one root cause; it is not needed to
  enumerate them.
- **Sample-based error estimate.** Rather than reviewing the worklist top-down,
  review a *random* sample stratified by class to get an unbiased per-class
  accuracy number. The worklist is for fixing; the sample is for measuring. Do
  both — they answer different questions.

## How to evaluate

Have a reviewer label ~150 flagged regions plus ~150 random unflagged ones
(the control — without it you only measure precision, never recall):

| Metric | Definition |
|---|---|
| precision @ severity ≥ *t* | fraction of flagged regions that are real errors |
| recall | fraction of real errors that got flagged (needs the unflagged control) |
| precision per check type | which of the six checks to keep |
| systematic detection | did flag-rate-by-class find the classes with real problems |
