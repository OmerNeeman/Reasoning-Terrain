# S1 — Consistency audit / QA of the segmentation

`segmap solve s1` · [`s1_audit.py`](../../src/segmap_digest/solutions/s1_audit.py) · engine in [`audit.py`](../../src/segmap_digest/audit.py)

**What it is.** Flag regions whose label contradicts their context, their slope,
or their geometry; roll those flags up into the *distinct facts* they are
instances of; and rank the facts by what a reviewer's next minute is best spent
on. Output is a worklist of decisions, not prose and not a wall of rows.

**Why it's first.** It's the only solution whose output feeds back into the
segmenter: confirmed findings become training-data corrections, and the flag
rate per class is an error map you can act on without any detection ground
truth existing.

**Read the coverage block before the findings.** S1 can only contradict a label
where something gives it grounds to. On the sinai mosaic that is 15 of the 33
classes present, covering 3.0% of classified area. The other 97% is
**unexamined, not clean**, and the report says so above the finding count.

---

## Naive algorithm (implemented)

```
for each region:
    if area ≥ min_area:
        check co-occurrence priors     (expects / contradicts, boundary-share weighted)
        check slope band               (badlands need relief, hydromorphic needs flat)
        check aspect coherence         (a dip slope is one facet or it isn't)
        check lithology isolation      (a Limestone island inside Dolomite)
        check nari geometry            (calcrete caps: thin high bands, not blobs)
        check enclosed OOV object      (artifact class, object scale, ground area)
    if SPECK_MIN_PX ≤ area_px ≤ SPECK_MAX_PX:
        check enclosed speck           (noise scale, resolution cells)

grade severity from the evidence, per check      → SEVERITY_BAND
roll findings up on Finding.cause                → N decisions, not N rows
rank causes by severity × area^0.5
group by class → flag_rate → "systematic vs noise"
report coverage: which classes had an active check, and over what area
```

Every check runs behind `--min-area` **except** `enclosed-speck`, which is the
check *about* components below that gate. Gating it inverts it — that is how
300 m² buildings ended up ranked as isolated specks on aza.

---

## What was wrong, and what the fix measured

Three failures, all found on real exports, all reproduced here before being
changed. Numbers below come from runs on `data/incoming/sinai/` (20 tiles,
0.496 m/px, 31822×37535, 133,976 regions) and `data/incoming/aza/` (4 tiles,
0.122 m/px, 8509×8405, 95,170 regions), at default `--min-area 25`.

### 1. A prior that was wrong about this model

The co-occurrence priors encoded textbook soil genesis: terra rossa as a
decalcification residue of hard carbonate, rendzina as the shallow soil of
chalk and marl. **The class owner has confirmed these classes are SOIL TYPES,
not genetic units** — the segmenter assigns them from what the soil surface is
like, so "Rendzina must border chalk or marl" is not a claim the labels ever
made.

Four priors were retired: `TerraRosa expects` hard carbonate, `TerraRosa
contradicts` soft carbonate and basalt, `Rendzina expects` chalk and marl,
`Rendzina contradicts` basalt. On the sinai mosaic they had produced **10,788
of 41,669 findings** (Rendzina 9,793, TerraRosa 991, contradicted-context 4) —
every one a false positive manufactured by our own assumption. The
`missing-expected-context` count fell from 10,842 to 58.

The **class definitions** were corrected too. `TerraRosa` and `Rendzina` now
describe the soil and say the label carries no parent-rock claim;
`ChalkSmoothRockSlopes` no longer asserts a rendzina association. That text is
what an LLM reasons over, so deleting the check while leaving the genetic story
in the definition would have kept the bad inference alive.

The retired priors survive as `taxonomy.RETIRED_PRIORS` — text, not checks —
with the reason attached, so the next person to read a soil-genesis chapter
finds the decision instead of re-deriving it. `tests/test_s1_rollup.py` fails if
a soil class regains a parent-rock neighbour requirement.

**Untouched, deliberately:** priors about geometry and physics. Badlands need
relief, a dip slope needs a coherent aspect, a hydromorphic soil sits low, a
car sits on something trafficable. Those are properties of the mapped object,
not of its history.

### 2. A worklist that showed one finding twenty-five times

41,669 findings over 133,976 regions, and a top-25 that was one Rendzina
sentence repeated twenty-five times, because a single prior with a fixed
severity of 0.45 swept the whole budget.

**Roll-up.** Every `Finding` now carries a `cause` — the class-level fact it is
an instance of (`isolated-speck/DryGrassland/in-Rendzina`) — and a `cause_text`
stating that fact once without per-region parameters. `worklist()` emits one row
per cause with its region count, total area, share of map, area-weighted
severity and up to five example region ids. The per-region rows stay addressable
through `detail(report, cause_key)`.

| AOI | regions | findings before | findings after | distinct root causes |
|---|---:|---:|---:|---:|
| sinai mosaic | 133,976 | 41,669 | 26,592 | **178** |
| sinai tile `x802_y847` | 11,058 | 4,136 | 2,484 | **34** |
| aza mosaic | 95,170 | 347 | 5,980 | **120** |

The sinai top 10 is now three kinds and eight classes, headed by
`DryGrassland islands inside Rendzina` (15,469 regions, 0.15% of map) and
`Clutter objects enclosed by Rendzina` (207 regions — detector targets).

aza's count *rises* because the speck window moved down to the scale where aza
actually fragments (see 3); the 347 it had before included 159 buildings
misfiled as specks. 5,980 findings, 120 decisions.

**Graded severity.** `SEVERITY_BAND` gives each kind a `(floor, span)`; each
check computes a violation fraction from its own evidence and maps it onto that
band, discounted by how much evidence there was to violate (a context claim from
5 m of shared boundary is weaker than the same claim from 5 km). Severity is an
**ordinal ranking signal within a kind** — nothing has calibrated it as a
probability, or shown the bands are comparable *across* kinds.

### 3. A speck threshold set against a synthetic fixture

`isolated-speck` was 69% of sinai-mosaic findings on data where 20–46% of
connected components are under 10 px. The old window was a fixed ground area,
25–400 m², set against a 0.3 m/px fixture. Measured on the real exports:

| | aza, 0.122 m/px | sinai tile, 0.496 m/px |
|---|---|---|
| raw connected components | 192,155 | 14,257 |
| median component | 11 cells (0.16 m²) | 55 cells (13.5 m²) |
| modal log₂ bin | **8–16 cells** | **32–128 cells** (28.5%) |
| below 10 cells | 46.3% of comps, 0.45% of area | 19.8%, 0.02% |
| below 32 cells | 70.5%, 1.62% of area | 38.9%, 0.09% |
| old 25–400 m² window, in cells | **1,667–26,675** | **102–1,625** |

A 16× difference in what "small" meant. **What was chosen:**

- **`SPECK_MIN_PX = 32`, `SPECK_MAX_PX = 128` — resolution cells, not m².** What
  makes something a speck is that the segmenter had too few cells to assert a
  textural class over it. 32–128 cells is the *modal* component size measured on
  the sinai tile; below 32 is the sub-resolution tail, above 128 the segmenter
  is drawing objects. In ground units: 7.9–31.5 m² on sinai, 0.48–1.9 m² on aza
  — small in both, which is the point.
- **The check is exempt from `--min-area`**, because it is the check about
  components below that gate.
- **The sub-32-cell tail is reported once, as a fragmentation rate**, not
  enumerated: 20.4% of indexed sinai-mosaic regions and 40.4% of aza regions sit
  below the floor, carrying 0.08% and 1.06% of area. That is a property of the
  segmenter's output smoothness, not a list of label errors.
- **`oov-candidate` was split out and keeps a ground-area window**
  (`OOV_MAX_M2 = 400`). "Is there an unnamed object here worth a detector pass?"
  is a question about object size, not cells — a car is a car at any GSD.

The aza distribution has **no mode above 1 cell**: it decays monotonically from
the floor, the signature of a map with no object scale at all. That is itself a
finding about aza, and no threshold fixes it.

---

## Heuristic constants — all guesses unless noted

| Constant | Value | Where it should come from |
|---|---|---|
| `AREA_EXPONENT` | 0.5 | Reviewer time-motion: is a 100× bigger error worth 10× the attention, or 100×? |
| `SYSTEMATIC_FLAG_RATE` | 0.15 | The per-class flag rate on a *known-good* tile — that's the noise floor |
| `SYSTEMATIC_MIN_REGIONS` | 8 | Small-sample guard; set from the binomial CI you're willing to act on |
| `CAUSE_EXAMPLE_RIDS` | 5 | How many examples a reviewer needs to recognise a pattern |
| `CONTRADICTS_SHARE` / `EXPECTS_SHARE` | 0.40 / 0.05 | Sweep against reviewed regions |
| `LITHOLOGY_ISOLATION_SHARE` | 0.85 | The geological map, as ground truth |
| `NARI_BLOB_AREA_M2` / `_COMPACTNESS` | 20,000 / 0.35 | A geologist, or the measured size/shape distribution of mapped nari |
| `SPECK_MIN_PX` / `SPECK_MAX_PX` | 32 / 128 | **Measured** (table above) — but properly, the segmenter's receptive field and its minimum mapping unit, which its owner knows |
| `OOV_MAX_M2` | 400 | The size of the objects the downstream detector is paid to find |
| `ENCLOSURE_SHARE` | 0.90 | Sweep; 1.0 is too brittle for a raster boundary |
| `SEVERITY_BAND` | see `audit.py` | A reviewed sample. Ordinal today, uncalibrated, and untested across kinds |
| `EVIDENCE_FULL_PERIMETER_M` | 200 | The boundary length at which neighbour-class share stops being noisy — measurable today from the region index |
| `SLOPE_VIOLATION_SIGMA` | 2.0 | **Unexercised**: no AOI here ships a DEM |
| prior slope bands | see `taxonomy.PRIORS` | **A geologist, not a sweep** |
| `index.build_regions(min_area_px=12)` | 12 | Not S1's to set, but measured: it drops 50.5% of aza components (0.56% of area) and 22.4% of sinai-tile components (0.02%) — 0.18 m² vs 2.95 m², 16× apart in ground terms. Whoever owns `index.py` should say whether that floor is cells or metres. |

---

## Open questions

1. **What's the false-positive rate?** Still unknown, and now the *only* thing
   standing between this and a usable precision number. Every count above is a
   candidate count. **This is the blocking question**, and the roll-up makes it
   cheaper to answer: 178 causes on the sinai mosaic is a reviewable sample in a
   way 41,669 findings never was. Review the causes, not the rows.
2. **`class_distance` is doing no work in `isolated-speck`.** It returns 1.0 for
   *every* cross-superclass pair, so "semantically distant host" filters almost
   nothing. The top two sinai causes are `DryGrassland` inside `Rendzina` and
   `Rendzina` inside `DryGrassland` — 21,719 findings between them about two
   classes that co-occur on the same ground by definition, grass growing on
   soil. The check is effectively "small + enclosed". Either `class_distance`
   needs a notion of *compatible* classes (vegetation over soil is not a
   contradiction) or the speck check needs a different distance. **Do not fix
   this by adding a prior** — that is exactly the mistake that produced the
   retired ones. It needs the class owner, or a confusion matrix.
3. **Is a "region" the right unit?** Connected components fragment badly —
   133,976 on the sinai mosaic, 95,170 on aza, and 20–40% of them below the
   speck floor. A boundary wobble creates two findings where a human sees one
   problem. The roll-up hides this at the worklist level; it does not fix the
   unit. Alternatives: merge adjacent same-class components across thin gaps;
   superpixels; work at the chip level.
4. **Is severity comparable across kinds?** Within a kind it now tracks the
   evidence. Across kinds the `SEVERITY_BAND` floors are an ordering someone
   asserted, and `review_value` multiplies them by area^0.5 as if they were on
   one scale. Same reviewed sample answers this.
5. **Coverage is 3.0% of the sinai mosaic and 33.7% of aza.** The audit has
   nothing to say about Rendzina (65% of sinai) or Shadow (29% of aza). Closing
   that gap is not a matter of tuning — it needs either priors for the dominant
   classes (from the class owner) or the DEM, which alone would activate the
   slope and aspect checks that currently abstain everywhere.
6. **~~How do priors get maintained?~~ Partly answered, badly.** They are still
   Python dataclasses, but the retired set is now recorded in code with reasons.
   If a class owner owns them they need to be a data file with review history —
   and the soil-genesis episode is the argument for it: a wrong prior cost
   10,788 false findings and was invisible until someone read the worklist.
7. **Does the audit run per tile or per AOI?** Priors like "nari caps units" are
   regional statements; a tile crop can make a legitimate unit look isolated.
   Measured: the same 20 tiles give 4,136 findings on one tile and 41,669 across
   the mosaic — regions cut at seams get wrong areas and wrong neighbour lists,
   so the mosaic is the honest unit and the tile is the fast one.

## Options and extensions

- **Cheap win — a class-confusion prior from the segmenter itself.** If you have
  the model's per-pixel logits or a validation confusion matrix, "what else could
  this be" stops being a guess. This is the single largest available improvement,
  needs no new data collection, and would settle open question 2.
- **Boundary-quality check.** Flag regions whose boundary is unusually ragged
  relative to others of the same class — a proxy for low-confidence pixels
  without needing logits.
- **Temporal consistency as a free labeller.** A region whose class flips between
  dates but shouldn't (see S6 `impossible`) is a confirmed error with no
  reviewer involved.
- **Feed the *causes* to an LLM, not the findings.** 178 rolled-up facts fit in a
  prompt; 26,592 findings do not, and the model's value is judging whether a
  fact is real — not enumerating its instances.
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
| precision per **root cause** | which facts are real; one verdict retires thousands of rows |
| recall | fraction of real errors that got flagged (needs the unflagged control) |
| precision per check type | which of the checks to keep |
| systematic detection | did flag-rate-by-class find the classes with real problems |
| coverage-weighted recall | recall × fraction of area under an active check — the number that stops a clean-looking report from meaning "clean" |
