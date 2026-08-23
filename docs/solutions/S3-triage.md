# S3 — Cost-aware detection triage

`segmap solve s3 --policy vehicles|structures|oov` · [`s3_triage.py`](../../src/segmap_digest/solutions/s3_triage.py)

**What it is.** Use the cheap segmentation as a prior to decide which image
chips get sent to an expensive detector. This is the solution with a direct
cost line attached.

**The load-bearing claim.** The LLM is *not* in the per-chip loop. It compiles a
query into a small cacheable policy, once per query type; a deterministic scorer
applies it to millions of chips. An LLM call per chip replaces one expensive
call with another and saves nothing.

---

## Regimes — the economics differ by an order of magnitude

| Regime | Example | What the map gives you | Savings |
|---|---|---|---|
| **R1** target *is* a class | cars, houses, water | the mask is nearly the answer | 90–99% |
| **R2** target is context-bound | fences, culverts, tents | strong prior from co-occurring classes | 70–95% |
| **R3** target invisible to taxonomy | camouflage, disturbed earth | only `Clutter`/`Unclassified`/`Shadow` + entropy | 30–60% |

Classify the query into a regime *first*; it determines whether you're ranking
or hunting, and R3 is where over-pruning does real damage.

## Naive algorithm (implemented)

```
policy = BUILTIN[name]                    # hand-written stand-in for the LLM step
for each chip:
    ex_frac = Σ frac[hard_exclude]
    if ex_frac > 0.80:  score = 0, excluded
    else:
        score  = Σ weight[c]·frac[c]
        score *= (1 − ex_frac)            # partial exclusion counts too
        score *= Π context_boost factors  # dist_to.X < N, interface.A_B > N, entropy > N
        score *= (1 + entropy_weight·H/4)
select top ⌈budget_frac·N⌉ + a 3% random sample of the REJECTED chips
```

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `EXCLUDE_DOMINANCE` | 0.80 | Measured hit rate inside excluded mass — should be ~0 |
| `ENTROPY_WEIGHT` per regime | 0/0.15/0.5 | Sweep against detector outcomes |
| `CONTROL_SET_FRACTION` | 0.03 | Set from the CI width you need on the recall estimate |
| all `class_weights` | hand-written | **Replace with measured p(hit\|class) — the whole point of the calibration loop** |
| `budget_frac` | 0.20 | Should not exist. Replace with a *recall target* (below) |

---

## Open questions

1. **What is the detector's cost model?** Per call, per megapixel, or per
   second? This changes the optimal chip packing completely — per-call pricing
   rewards merging adjacent selected chips aggressively; per-megapixel makes
   merging neutral or harmful because you pay for the padding. **Answer this
   before optimising anything.**
2. **`budget_frac` is the wrong knob.** Top-K is a placeholder. The real
   selector fits `p(hit | score)` on outcome data (isotonic regression — monotone
   by construction) and takes the smallest set reaching a *recall target*:
   "cover 95% of expected targets at 11% of cost." That's a knob a user can sign
   off on; K=5000 isn't. **Blocked on outcome data, not on design.**
3. **Where does the control set live in the budget?** 3% of rejected chips is a
   real cost. It buys the only evidence that the other 90% of savings is safe.
   Who signs off on paying it?
4. **Is chip-level the right granularity?** A 128 px chip at 0.3 m is 38 m
   across and routinely contains six classes. Region-aware chips (crop to the
   trafficable component) may score far better than a fixed grid.
5. **Who approves a hard-exclude list?** Excluding 85% of an AOI is a decision
   with consequences. `exclude_rationale` exists so a human can review it in 30
   seconds — but there's no approval workflow, and no test that the rationale
   matches the list.
6. **How are policies versioned and invalidated?** Keyed on (query, taxonomy
   version, detector version, GSD band) in the design; nothing enforces it. A
   stale policy against a retrained segmenter fails silently.
7. **Does the LLM policy-generator beat the hand-written one?** Unknown — and
   testable today, because the hand-written policies are checked in as the
   baseline.
8. **Spatial diversity.** Nothing stops the whole budget collapsing onto one
   dense settlement. Needed for R3 especially.

## Options and extensions

- **Cascade.** Cheap/small detector over the top ~40%, expensive detector on its
  hits plus the top slice. Often another 3–5× on top of the base saving.
- **Negative filtering before positive ranking.** For most queries 60–90% of a
  terrain tile is trivially excludable at near-zero risk; ranking the remainder
  is the risky part. Two knobs, two risk profiles, two sign-offs.
- **Region-derived chips** instead of a fixed grid (see Q4).
- **Scene-aware policy adjustment.** One extra LLM call per AOI with the `l0` +
  `l1q` digest (~2–4k tokens), letting the model tune the policy to *this* scene
  — "this AOI is 70% badlands with one wadi corridor." Trivially cheap next to a
  detection sweep.
- **R1 shortcut.** When the target *is* a class, skip detection entirely for
  counting and area questions; dispatch only for instance separation and
  verification.
- **Outcome log → calibration.** Log every dispatched chip with its score,
  policy id, class histogram, and result. That log is what turns the guessed
  weights into measured ones and makes the LLM prior a cold-start device only.

## How to evaluate

| Metric | Definition |
|---|---|
| cost reduction | chips dispatched / chips in a full sweep |
| **recall retention** | detections found / detections a full sweep would find — **estimated via the control set** |
| yield | detections per chip dispatched, vs. the full-sweep baseline |
| excluded-mass safety | hit rate inside hard-excluded chips; should be ~0 |
| class-conditional recall | **per class** — aggregate 95% can hide 0% on Garigue |

Target shape for R1/R2: **10–20% of cost at ≥95% recall.** R3 is more like 50%
of cost at 90%. You need a full-sweep baseline on one or two tiles to report any
of this — build that first.

> The synthetic tile has no detector and no targets, so `score captured` in the
> CLI output is **prior mass, not recall**. It is a plausibility check on the
> scorer, nothing more.
