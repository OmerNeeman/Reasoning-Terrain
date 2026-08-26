# Weakness review — the three features built this session

Three adversarial reviews, run in parallel, one per feature. 24 findings, all
marked CONFIRMED by the reviewer; the ones flagged **[re-verified]** below were
independently reproduced a second time before being written down.

Nothing here is fixed. This is the record.

---

## `segmap ui` — 8 findings

| # | Severity | Defect |
|---|---|---|
| 1 | **high** | One POST wedges the whole server, permanently |
| 2 | **high** | A negative region id adjudicates a *different* region and labels it with the id you passed |
| 3 | med | S6 shape mismatch raises outside the try block → 500 + traceback |
| 4 | med | S6's `except` misses what rasterio actually raises |
| 5 | **high** | No `Origin` / `Host` / `Content-Type` validation |
| 6 | med | The error handler ships `traceback.format_exc()` to the browser |
| 7 | low | Two paths return zero bytes instead of a status code |
| 8 | low | Legend click does nothing on the Solutions tab |

**1. A single request can wedge the server.** `_LOCK` is held across the entire
dispatch, and the S6 branch calls `loader.load()` on a raw path from the POST
body. Reproduced with a FIFO: every later request hangs forever while
`/api/meta` still answers in 0.9 ms — so the page looks healthy and every button
spins with no timeout and no recovery short of Ctrl-C. The same shape means one
in-flight `/api/ask` blocks a trivial Verbs query for the length of the model
call.

**2. Negative region id → confident wrong answer. [re-verified]**
`RegionIndex.get` is `regions[rid - 1]`, so `-1` indexes `regions[-2]`, and the
`Adjudication` stores the id you *passed*, not the region's own. The S2 input has
no `min`. Live on the demo tile: `region = -1` returns a full four-candidate
evidence table and an `UNDECIDABLE-NEEDS-geological-map` verdict headed
`# region -1: incumbent ChalkTerrace` — which is really region **15542** of
15543. In a demo whose entire premise is that every number is checkable against
the map, this is the worst possible failure mode: not a crash, a confident wrong
answer. `999999` gives `IndexError` → 500. `if rid:` also treats a typed `0` as
blank.

**5. Any page you visit can drive this API.** A `text/plain` POST is a CORS
simple request, so no preflight — a malicious page can wedge the server, spend
Anthropic credits through `/api/ask`, and probe the filesystem through S6.
`Host: attacker.example` is accepted, so DNS rebinding makes the responses
readable, at which point S6's distinct error strings (`unsupported label raster
format` vs `No such file or directory`) are a file-existence oracle over the
whole filesystem. `--host` is exposed with no warning.

**7. Zero-byte responses. [re-verified]** `Content-Length: abc` →
`http=000`, connection dropped. `log_message` reads `self.path` before
`http.server` has assigned it on a malformed request line, so it raises *inside*
`send_error` and aborts the 400 mid-flight.

**Dead code left behind:** `_args()` is defined and never called; `legend` is
threaded from the CLI through `serve` into `_STATE.legend` and never read
(`run_ask` recomputes it).

**Cleared:** every call signature against the six solution modules is correct;
all dropdown values match module keys; **no XSS** (every dynamic value goes
through `textContent`, `innerHTML` only ever receives `""`); no request path
mutates shared state; `s3_triage.run`'s tuple is unpacked in the right order;
`num()` casting is sound.

---

## `segmap showcase` — 8 findings

| # | Severity | Defect |
|---|---|---|
| 1 | **high** | S4/S1/S2 have no error guard — one raster kills the page, nothing is written |
| 2 | **high** | `densest_window` fallback captions an empty crop with a confident fraction |
| 3 | **high** | The `count` merge heuristic is structurally unsound at small n |
| 4 | med | "Plausibly yes" asserted when every assessable class was skipped |
| 5 | med | The S5 callout hard-codes "no DEM", contradicting the page's own DEM pill |
| 6 | med | Same raster computed 6×; ~5.5 GB peak on a 67 Mpx tile |
| 7 | med | `have()` tests at full res, `isolate()` renders decimated → blank figure, confident caption |
| 8 | low | Season-cost mean divides by full extent, not classified pixels |

**1. No guard on three of six solution blocks.** S5, the distance figure,
corridor and S3 are wrapped; **S4, S1 and S2 are bare**. A raster whose `valid`
mask is all-False — a fully-nodata mosaic tile, or a `--max-mpx` crop landing in
a nodata gap — reaches `np.percentile([], 10)` → `IndexError`. The file is only
opened at the very end, so six completed sections and the entire index build are
discarded and the CLI prints a raw numpy traceback.

**2. The small-raster fallback lies.** The guard trips on any raster with one
dimension ≤ ~851 px and returns the global centroid. Reproduced on an 800×12000
swath with settlements at both ends: it returns a point in empty grassland
midway between them, and captions it *"centred on the densest window at (399,
6728). 5.8% of this window is House."* The rendered crop's actual House fraction
is **0.0**. (The non-fallback path was verified correct against brute force.)

**3. The merge heuristic fires by construction at small n. [re-verified]**
`top_share` is bounded below by `1/n` and the gate is only `n >= 3`, so at n=3
almost any inequality clears 0.35. Reproduced with **five physically separate**
car blobs, none touching: the page declares them fused and replaces the correct
count of 5 with *"~7 Car"* derived by dividing total area by a guessed 12 m².

> **Does this invalidate the finding reported from the sinai tile? No —
> re-verified.** House there is n=81 with a 64.5% top share, **53× the
> structural floor**, and Car was flagged by the *median* test (87 m² vs a 12 m²
> car), not the share test. Both signals are real. What is not defensible is the
> derived **"~1,159 houses"**: that divides total area by a hard-coded 60 m²
> "typical house" and is only sound for a tight size distribution. Treat the
> fusion as established and the implied count as an order-of-magnitude guess.

**4. A clean bill of health from zero evidence.** `suspect` requires
`typical is not None`, so a class with no reference size can never be flagged;
`bad` is then empty and the `else` branch fires. On a BrickWall-only raster the
table row reads *"no reference size — not assessed"* and the verdict directly
beneath it reads *"On this map, plausibly yes."*

**6. Six identical recomputations, five rasters pinned.** Instrumented: **10
calls** to `s4_products.compute` per page, 6 byte-identical. `arr`, `dry`, `wet`,
`delta` and `traf` all stay alive in `build()`'s frame through S1, S2 and S3
after last use. Measured peak ~83 B/px → **~5.5 GB at 67 Mpx**, ~100 GB at the
1.2 Gpx mosaic the docs target.

**7. Presence tested at full res, figure rendered decimated.** On 8192×8192
(step 7) with 40 cars of 4×8 px, `have("Car")` passes on 1280 px but only **23
survive** `[::7, ::7]` — an effectively blank figure under a confident caption,
with nothing saying the class was decimated away.

**Cleared:** no HTML injection path (`_SHELL` doubles every CSS brace, `format`
does not rescan substitutions, `_pre` truncates before escaping);
`ridx.get(rid)` is sound here because ids are renumbered `1..N`; `_box`/`_fill`
cannot invert or go out of bounds; `zoom` handles a raster smaller than
`ZOOM_PX`; `colourise`/`_grey` always decimate first.

---

## S5 eval harness — 8 findings

**The headline: "every number traceable to a tool result" cannot be tested by
this harness even with a working key.** Three defects each break it alone.

| # | Severity | Defect |
|---|---|---|
| 1 | **critical** | The model can never emit the number the ground truth records |
| 2 | **high** | An iteration-capped run is indistinguishable from a clean one |
| 3 | **high** | `audit()` prints summaries containing no numbers; payloads are never persisted |
| 4 | **high** | No check that quoted ids are returned ids; model sees 25 of 37 |
| 5 | med | `truth/describe.txt` truncated at 25 rows; the model sees 40 |
| 6 | med | `describe` and `area` disagree; SUMMARY.md records one and never flags it |
| 7 | med | Q7/Q8's abstention test is short-circuited by the system prompt |
| 8 | med | Q5's rubric mandates an assumption that never applied |

**1. Formatting makes the ground truth unreachable. [re-verified]**
`tools.render()` formats the headline scalar with `%g` — 6 significant figures.

| `truth/` records | The model actually sees |
|---|---|
| `203,550.80` | `= 203,551 m2` |
| `16,086,883.54` | `= 1.60869e+07 m2` |

A grader marks a correct model wrong. Worse, inverted: **any model that printed
`16,086,883.54` could not have gotten it from a tool** — so the harness awards
"correct" for precisely the fabrication it exists to catch. The corridor payload
shows the same quantity three disagreeing ways at once (`1.60869e+07`,
`16086884`, `97.3%`).

**2. Silent truncation reads as success.** `MAX_ITERATIONS = 12` exits the
runner quietly; `stop_reason` is `tool_use`, not `refusal`, so `ask_tools`
returns normally, `run.sh` records `exit=0`, and `audit()` still prints *"every
number above was computed by one of them"*. Q9 is the likely trigger: no
all-class verb exists, so the model iterates per class across 30 classes.
`run.sh` also has no `timeout` (a stall hangs all ten questions) and never
checks the `### exit=` trailer it writes.

**3. The audit trail contains no numbers.** `audit()` prints `c['summary']`,
which is only `"<scalar> <unit>, <n> rows, <n> ids"`. The `render(payload)` the
model actually consumed is discarded and exists nowhere in `out/live_ask/`. For
`describe` — which has no scalar — the audit line is literally
`ok  describe  ->  30 rows, 30 class id(s)`. A brief saying "Rendzina dominates
at 65.3%" is indistinguishable from one saying 61%: neither figure appears
anywhere in the captured artifact.

**4. Ids are counted, never listed.** `EVIDENCE_MAX_IDS = 25` means the model
received 25 of 37 House ids. A fabricated 26th id is indistinguishable from one
it was legitimately withheld, and `audit()` prints only the count. S5's own docs
list this as missing; SUMMARY.md does not.

**5. Ground truth captured through a tighter cap than the tool path.**
`truth/describe.txt` stops at 25 rows (`--budget` default) where
`TOOL_MAX_ROWS = 40`. The five missing classes include a single 10 m²
`UnirrigatedOrchard` region — the only agriculture pixel on the tile, and the
single best misclassification observation available for Q10. A model that spots
it is graded as hallucinating.

**6. Two verbs, two answers, one recorded.** `describe` counts pixels; `area`
sums indexed regions post-min-area-cut. DirtRoad 203,626 vs 203,550.80; PavedRoad
112,522 vs 112,506.51. The system prompt says *"start broad with `describe`"* —
so a compliant model reads the number the ground truth does not have. (The Q3
ratio survives: 1.8097 vs 1.8092, both round to 1.81×.)

**7. The refusal test measures prompt recall, not the code contract.**
`TOOL_ROLE` already states *"vehicle type, building function … no tool call will
get at them."* A compliant model answers Q7/Q8 with **zero tool calls**, so
`UNANSWERABLE_CONCEPTS` never runs. SUMMARY.md claims the opposite.

**8. A mandated assumption that constrained nothing.** The Q5 rubric requires the
answer to state the 25° slope limit. With no DEM every slope is 0.0, so the limit
is vacuous and 97.3% is an unconstrained upper bound — yet a model that says so
is *penalised* for omitting a listed assumption, and one that parrots it scores
full marks. Related and unchecked: a model may answer Q9 with `corridor` (whose
description advertises slope handling), producing a real 97.3% for a steepness
question that never touched slope.

**Also:** no `truth/` file was written for Q7–Q9 at all, though SUMMARY.md's
file list claims one per query.

**Cleared:** all 20 captured runs really do record `exit=1` and an
`AuthenticationError`; `questions.txt` is read completely; `rc=$?` is captured
correctly; stdout/stderr interleaving does not corrupt the answer block; the
corridor, distance and count truth files are internally consistent.

---

## What this changes

1. **The eval harness needs repair before the key does.** Fixing the key and
   re-running today would produce a graded table that measures formatting
   collisions and prompt recall rather than the planner. Findings 1, 3 and 4 are
   the blocking set: persist the rendered payloads, print ids not counts, and
   settle a numeric-tolerance rule.
2. **`segmap ui` is a localhost demo and should say so.** Findings 1, 2 and 5 are
   all reachable from the page's own controls or from any tab the operator has
   open.
3. **`showcase` fails loudly on rasters it should survive.** Three of six
   solution blocks are unguarded, and two captions state things the image does
   not show.
4. **The sinai fusion finding stands; the "~1,159 houses" figure does not.**
   Fusion is established at 53× the structural floor. The implied count divides
   by a guessed 60 m² and should be read as an order of magnitude.
