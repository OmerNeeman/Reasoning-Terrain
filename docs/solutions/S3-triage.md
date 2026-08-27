# S3 — Cost-aware detection triage

`segmap solve s3 --policy vehicles|structures|oov` · [`s3_triage.py`](../../src/segmap_digest/solutions/s3_triage.py)

**What it is.** Use the cheap segmentation as a prior to decide which image
chips get sent to an expensive detector. This is the solution with a direct
cost line attached.

**The load-bearing claim.** The LLM is *not* in the per-chip loop. It compiles a
query into a small cacheable policy, once per query type; a deterministic scorer
applies it to millions of chips. An LLM call per chip replaces one expensive
call with another and saves nothing.

**What is new.** The answer is now given *to a query* rather than to a fixed
policy name — `run_query(cidx, "find trees")` — and the answer knows when the
query is **meaningless on this ground**.

---

## Two questions, not one

A query carries two separable things per unit of ground, and collapsing them is
how you get a confident wrong answer:

| | question | who answers it |
|---|---|---|
| **applicability** | could the thing asked about plausibly be here **at all**? | `applicability(chip, policy)` — new |
| **priority** | given that it could, how much does this unit deserve the spend? | `score_chips` — unchanged |

```
worth_a_look = applicability × priority
```

> Searching for a missing tree is super relevant in a forest but not on a road
> in the desert of Sinai. — the owner

A unit with zero applicability scores zero however interesting it looks, and
the renderer reports it separately:

```
not applicable       16 (100%) -- the query does not apply to this ground
hard-excluded         0 (0%)
```

Those are three different sentences and the system must never merge them:

| what happened | what it means | how it is reported |
|---|---|---|
| **not applicable** | the question is a category error on this ground | `applicability < 0.02`, counted and named |
| **hard-excluded** | the target could exist here in principle but physically cannot | `hard_exclude` + `exclude_rationale` |
| **scored low** | applicable, searchable, just not worth the budget | ranked below the cut, sampled by the control set |

"The query does not apply here" is **not** "we searched and found nothing", and
reporting the second when the first is true is exactly the silent wrong answer
this repo exists to prevent. When *every* unit is inapplicable, `render` says so
and disowns its own cost figure — 94% cost reduction on a search that never
happened is not a saving.

---

## The query surface

```python
compile_query(text, gsd=None) -> Policy          # deterministic, no LLM
applicability(chip, policy)   -> float           # 0..1
score_query(cidx, policy)     -> [ScoredChip]    # applicability × priority
run_query(cidx, query, budget_frac=0.20, gsd=None) -> (Policy, Selection, diagnostics)
```

Nothing above disturbs the old surface: `BUILTIN`, `Policy`, `ContextRule`,
`score_chips`, `select`, `render`, `run`, `run_blocks` behave exactly as before,
and a hand-written `BUILTIN` policy renders byte-identical output — it has no
applicability model, so `render` prints no applicability column for it rather
than printing `1.00` and passing an assumption off as a measurement.

Not wired into the CLI or web UI yet (both are owned elsewhere this session);
`run_query` is a library call today.

### The compiler is deterministic, and that is temporary

`compile_query` is hand-written parsing over the taxonomy. It is the same kind
of stand-in that `BUILTIN` is: it exists so the scorer, the applicability model
and the audit surface can be validated **with no model in the loop at all**.
The architecture says a model compiles the query into a small cacheable policy
once per query type; replacing this function with that model changes what
*fills* a `Policy`, not what *consumes* one. Everything downstream — the
weights, the rationale, the rules, the audit line — is already the interface.

### What it understands

| input | example | becomes |
|---|---|---|
| exact class names | `Maquis`, `PavedRoad`, `DirtRoadB` | that class |
| superclasses (`taxonomy.SUPERCLASS`) | `vegetation`, `road`, `built`, `rock`, `water` | the whole group |
| ordinal series | `degradation`, `succession`, `road series` | `DEGRADATION_SERIES` / `ROAD_SERIES` |
| synonyms (hand-written, `SYNONYMS`) | `tree`, `building`, `car`, `track`, `wall`, `orchard`, `rubble` | class tuples |
| absence | `missing tree`, `no trees`, `where trees were removed` | `absence=True` |
| proximity | `find vehicles **near** roads` | context, not target → `dist_to.*` rules |
| transition | `House -> Clutter`, `where was a road paved` | `transition=("House","Clutter")` |

`tree` is **derived, not listed**: it is every class whose taxonomy `canopy`
exceeds `TREE_MIN_CANOPY`, so adding a canopy-bearing class to `taxonomy.py`
extends the compiler for free. `Batha` (canopy 0.2) is a dwarf shrub and is
deliberately *habitat*, not a tree.

### What it records, so a human can approve it in 30 seconds

```
# S3 triage -- policy q-missing-tree@e10038b8 (regime R2)
# query: missing tree
# understood: missing -> absence sense; tree -> UnirrigatedOrchard+IrrigatedOrchard+Garigue+Maquis
#             | ABSENCE sense (looking for where the target is NOT)
# hard-exclude rationale: woody vegetation does not root on open water, boulder
#             fields, badlands or bare rock faces
```

`Policy` gained `query_terms`, `applies_to`, `context_classes`, `absence`,
`unmatched`, `notes`, `transition` — all defaulted, so every existing
construction site still works. `unmatched` is load-bearing: **if nothing in the
query matched the vocabulary the policy says so loudly and applies nowhere**,
rather than quietly matching everything and returning the top 20% of noise.

```
# WARNING: NOTHING IN THIS QUERY MATCHED THE TAXONOMY. No target class was
identified, so this policy applies NOWHERE and selects nothing. Rephrase using
class names, a superclass (...), or a known synonym -- do not treat an empty
result as 'no targets found'.
```

---

## How applicability is computed

Everything it uses is a statement the taxonomy already makes — no new
world-knowledge file, no second source of truth:

| source | what it contributes |
|---|---|
| the target classes | target mass in the unit |
| `taxonomy.PRIORS` (`expects`) | co-occurrence context: *a vehicle sits on something trafficable, near built-up ground* |
| `taxonomy.class_distance` | near-miss credit: Garigue is partial evidence for a Maquis query, a boulder field is not |
| `ANCHOR_CLASSES` distance fields | proximity: bare ground 30 m inside a maquis edge is still tree country |
| `ClassDef.traffic` | trafficable ground counts as vehicle habitat, so an off-road vehicle query is not pruned onto the road network |

**Presence** — `min(1, target + 0.6·habitat)`, floored by proximity to habitat.
**Absence** — *is this the kind of ground the target belongs on*: `max(target,
habitat, proximity)`. Note what absence does **not** do here: it does not ask
where the gap is. That is the priority half, where an absence policy weights the
**context** and gives the target class itself zero, then discounts by the
fraction of the unit the target already fills. Splitting it that way is what
keeps "a fully wooded chip" (applicable, nothing to find) distinct from "a
desert road" (not applicable at all).

> This was got wrong twice while being built, in both directions, and both
> failures are recorded in the constants: killing applicability off as the
> target filled the unit made a missing tree score 0 in a *forest*; moving the
> same cliff into the priority half zeroed every unit on a forested tile, so
> the flagship query returned a confident, arbitrary ranking. A 38 m chip in
> maquis is always mostly canopy and a missing tree is 5 m across. See
> `ABSENCE_GAP_EXPONENT`.

---

## Worked examples

Two 154 m tiles at 0.3 m, 16 chips each. Desert: basalt rocky and stoney ground
with one dirt track. Forest: maquis and garigue with one 24 m clearing of batha
and the same track.

| query | tile | not applicable | selected | what it means |
|---|---|---|---|---|
| `missing tree` | desert road | **16/16** | 0 | *the question does not apply here* — and the renderer disowns the cost figure |
| `find trees` | desert road | **16/16** | 0 | same |
| `find vehicles` | desert road | 0/16 | 3 | the same ground **is** meaningful for a vehicle query — a track in the desert is exactly where a vehicle is |
| `missing tree` | forest | 0/16 | 3 | applicable everywhere; the ranking does the work |
| `find trees` | forest | 0/16 | 3 | mean applicability 0.98 |

The forest ranking for `missing tree` — the clearing wins, the closed canopy
does not:

```
chip	r	c	score	appl	top_classes
4	1	0	0.078	0.70	Maquis:0.70 Batha:0.25 DirtRoad:0.05
0	0	0	0.023	0.86	Maquis:0.86 Batha:0.14
1	0	1	0.000	1.00	Maquis:1.00        <- applicable, but nothing missing here
```

Compiler output for the query set:

| query | targets | absence | transition | unmatched |
|---|---|---|---|---|
| `find trees` | Unirrigated/IrrigatedOrchard, Garigue, Maquis | no | — | — |
| `missing tree` | same | **yes** | — | — |
| `find vehicles near roads` | Car | no | — | — (roads → context) |
| `where was a house destroyed` | House | yes | `House -> Clutter` | — |
| `where was a road paved` | PavedRoad, DirtRoad, DirtRoadB | no | `DirtRoad -> PavedRoad` | — |
| `House -> Clutter` | Clutter | no | `House -> Clutter` | — |
| `find cars in the desert of Sinai` | Car | no | — | `desert`, `sinai` (reported) |
| `asdf zzz` | — | — | — | `asdf`, `zzz` → applies nowhere |

---

## Looking ahead: change detection

A change query is the same shape as a detection query — a query scored per unit
of ground — so it compiles into the same `Policy`. `transition` is parsed today
and **consumed by nothing**: `X -> Y`, `became`, `destroyed`, `paved`, `planted`
all land in `Policy.transition`, with `"*"` for an endpoint the query left open.
On a single map the compiler falls back to the present-day half of the
transition (the `to` side is what should be visible now, the `from` side is
context) and says so in `notes`. Change detection itself is not implemented here.

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

The query path wraps that, it does not replace it:

```
policy = compile_query(text, gsd)         # deterministic stand-in for the LLM step
scored = score_chips(cidx, policy)        # ← unchanged, the priority half
for each chip:
    a = applicability(chip, policy)
    if a < APPLICABILITY_MIN: score = 0, dropped as NOT APPLICABLE (a distinct reason)
    else:                     score *= a
        if absence:           score *= (1 − target_fraction)   # the gap term
select as above — the control set is unchanged and still mandatory
```

The control-set discipline survives untouched: rejected units are still sampled,
including units rejected for inapplicability, so an over-aggressive
applicability model is falsifiable by the same mechanism that falsifies an
over-aggressive hard-exclude list. It is still the only unbiased recall signal.

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `EXCLUDE_DOMINANCE` | 0.80 | Measured hit rate inside excluded mass — should be ~0 |
| `ENTROPY_WEIGHT` per regime | 0/0.15/0.5 | Sweep against detector outcomes |
| `CONTROL_SET_FRACTION` | 0.03 | Set from the CI width you need on the recall estimate |
| all `class_weights` | hand-written | **Replace with measured p(hit\|class) — the whole point of the calibration loop** |
| `budget_frac` | 0.20 | Should not exist. Replace with a *recall target* (below) |
| `APPLICABILITY_MIN` | 0.02 | Hit rate **inside the dropped units**, via the control set. If it isn't ~0, this is too high |
| `TREE_MIN_CANOPY` | 0.3 | The class owner's answer to "what does a user mean by *tree*" |
| `TRAFFICABLE_MIN` | 0.45 | Vehicle mobility tables. Deliberately the same value as `s5_query.CORRIDOR_MIN_TRAFFIC` — one physical claim, one threshold |
| `HABITAT_CONTEXT_WEIGHT` | 0.6 | Measured p(target \| context present, target class absent) |
| `NEAR_MISS_WEIGHT` | 0.6 | The segmenter's confusion matrix; `class_distance` is standing in for it |
| `APPLICABILITY_NEAR_M` | 120 m | The spatial scale of the target's context — differs per query family |
| `PROXIMITY_APPLICABILITY` | 0.5 | How much weaker "next to habitat" is than "is habitat" |
| `ABSENCE_GAP_EXPONENT` | 1.0 | Minimum interpretable clearing size ÷ unit area — i.e. a function of chip size and GSD, not a constant |
| `CONTEXT_WEIGHT` | 0.45 | Same calibration loop as `class_weights` |
| `COMPILED_NEAR_M` / `_FACTOR` | 100 m / 1.4 | Same sweep as the hand-written `context_boosts` |
| `SYNONYMS`, `TRANSITION_VERBS` | hand-written | Replaced wholesale when a model compiles the query |

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
9. **Who signs off on "not applicable"?** It is a stronger claim than
   hard-exclude — it says the *question* was wrong, not the ground — and right
   now one threshold (`APPLICABILITY_MIN`) makes it with no approval step. The
   control set is the only thing that can catch it being wrong, and on a small
   AOI the control set is too small to catch anything (see the render warning).
10. **Applicability is unit-sized, and the unit is arbitrary.** A 38 m chip in
    maquis is always mostly canopy; a missing tree is 5 m across. Every absence
    threshold is therefore really a statement about chip size ÷ target size, and
    none of them are written that way yet. The block unit (`run_blocks`) makes
    this worse: a block has no distance fields at all, so the proximity term
    silently contributes 0 — reported as "no evidence", never as "far away".
11. **Compiled weights are coarser than the hand-written ones.**
    `compile_query` gives every context class the same `CONTEXT_WEIGHT`, where
    `BUILTIN["vehicles"]` grades PavedRoad 0.9 / DirtRoad 0.7 / Batha 0.2 by
    hand. The hand-written policies remain the baseline the compiler has to
    beat — which is testable today, because both are checked in.
12. **The synonym table is a liability with no owner.** `tree → canopy ≥ 0.3`
    is derived and defensible; `rubble → Clutter+Unclassified` is somebody's
    guess about what a user meant. Every compile prints which entry fired, which
    makes it auditable, not correct.
13. **Absence over open country is under-discriminated.** "Where were trees
    removed" scores high across an entire batha plain, because batha is
    maquis context everywhere. Proximity to real canopy only raises a floor; it
    does not rank. This wants a real *deviation-from-expected* model, not a
    context mass.

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
- **Model-compiled policies.** Swap `compile_query` for one LLM call per query
  *type*, keeping the same `Policy` and the same audit line. The deterministic
  compiler stays as the fallback for a model outage and as the regression
  baseline: any model-compiled policy that selects a very different set from
  the deterministic one for `find vehicles` is a finding, in one direction or
  the other.
- **Applicability as a user-facing answer.** "Your query does not apply to 84%
  of this AOI" is a useful thing to say *before* spending anything, and it is
  already computed.

## How to evaluate

| Metric | Definition |
|---|---|
| cost reduction | chips dispatched / chips in a full sweep |
| **recall retention** | detections found / detections a full sweep would find — **estimated via the control set** |
| yield | detections per chip dispatched, vs. the full-sweep baseline |
| excluded-mass safety | hit rate inside hard-excluded chips; should be ~0 |
| **inapplicable-mass safety** | hit rate inside units dropped as *not applicable* — should be ~0, and it is a **worse** failure than an excluded-mass hit, because the system told the user their question did not apply |
| compiler agreement | does `compile_query("find vehicles")` select what `BUILTIN["vehicles"]` selects? Divergence is measurable today, with no detector |
| class-conditional recall | **per class** — aggregate 95% can hide 0% on Garigue |

Target shape for R1/R2: **10–20% of cost at ≥95% recall.** R3 is more like 50%
of cost at 90%. You need a full-sweep baseline on one or two tiles to report any
of this — build that first.

> The synthetic tile has no detector and no targets, so `score captured` in the
> CLI output is **prior mass, not recall**. It is a plausibility check on the
> scorer, nothing more.
