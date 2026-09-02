# Open issues — reasoning-terrain

One page, so that "what is still open in RT" is a file rather than an
archaeology exercise across six documents, three HTML reports, five solution
docs, a git log and an agent's memory.

**What this is for.** Nearly everything below is already known somewhere. What
did not exist was a single list you can make decisions against — most of these
items are not waiting on engineering, they are waiting on an answer, and the
answers are not the author's to give. §7 is the question set; everything above it
is the evidence for it.

## How to read this

| Mark | Means |
|---|---|
| **✔ re-verified 2026-08-31** | I ran it today. The number or the behaviour is current. |
| **↩ inherited** | Carried from the source named, not re-run. Trust it the way you trust its date. |
| **● decision** | Nothing is broken. Someone has to choose, and until they do the code is holding a placeholder. |

## Decisions taken, 2026-08-31

The owner answered the §7 questions on the day this file was written. What was
decided, and where it got to:

| Question | Answer | State |
|---|---|---|
| Q3 — fix the four commands that lie? | **yes, one pass with tests** | **done** — 2.1, 2.2, 2.3 and 1.3's code half all fixed; 16 tests in [test_fixes_open_issues.py](tests/test_fixes_open_issues.py), 9 of which fail against the code as it was (the other 3 are deliberate controls) |
| Q4 — `class_distance` | **leave the distance, make S1 say so** | **done** — `s1_audit.speck_distance_caveat()` enumerates the taxonomy at call time and prints the result wherever speck findings are reported, and names the fix that is *not* allowed |
| Q1 — which input first | **the DEM** | open — the next piece of work, not started |
| Q8a — rename the 12.5 cm weights | **yes** | **done** — `ST_12_5cm_model.onnx`; the old name still resolves |
| Q8b — segment both full orthos | **yes** | running, ~35 min each |
| Q2, Q5, Q6, Q7 | not yet answered | open |

A second round of answers came back on **2 September**:

| Question | Answer | State |
|---|---|---|
| Q2 — the two-date disagreement | **run the discriminators** | **done — and the verdict is bad news**, see 5.2 |
| Q7 — where engineering goes next | **georeference the answers** (2.8) | accepted; not started |
| §4 cross-cutting — rename the nari abstain | **yes** | **done** — `UNDECIDABLE-NEEDS-hardness-class` |
| §2.11 — the conflicting finding count | **settle it** | re-running `solve s1` on the sinai mosaic |
| Q5 — soft-pred export · Q6 — S4 score-or-classes | owner is getting both settled | awaiting the answers |

Running the discriminators also turned up a new defect in our own S6 — see 2.12.

Deliberately **not** done: everything else in this file. In particular 2.4's
underlying defect is unchanged — S1 now states that its speck check is "small and
enclosed" rather than claiming a semantic test it does not perform. That is an
honesty fix, not a correctness fix, and it is what was asked for.

---

Sources folded in: [HANDOFF.md](HANDOFF.md) §8/§9/§12, [README.md](README.md)
"Next steps", the six `docs/solutions/*.md` "Open questions" sections,
`docs/handover.html` (the sixteen-defect adversarial review + its five owner
questions), `docs/five-fixes.html` ("found along the way · not fixed"),
`out/status/WEAKNESSES.md` (24 findings over `ui` / `showcase` / the S5 eval
harness), the commit bodies of `f7ed72e`, `943c2f8`, `9dea10d`, `6b10cef`, and
this session's `segmap segment` work.

**Already closed, so not repeated below:** the five owner questions of the last
review round (S2 scoring scale, playground wedging, the `scale` class-note,
NaN-means-unmeasured, packaging) are fixed and demonstrated in
`docs/five-fixes.html`; the sixteen adversarial defects of `943c2f8` are fixed
with tests. One of the five review answers has since gone further than its page
says: `tools.py:310` now **lists** evidence ids with an `ids_omitted` count,
where `out/status/WEAKNESSES.md` finding 4 recorded them as counted-only.

---

## 1. Blocked on an input, not on code

These are the expensive ones. No amount of engineering closes any of them.

**1.1 No DEM anywhere. Still the single biggest blocker.** ↩ HANDOFF §9.1
23 of 47 classes are lithology×geomorphology, and morphology is what
slope/aspect/curvature resolve and colour does not. Without it S1's slope and
aspect priors abstain, S2's morphology term returns a neutral 0.5, S5's slope
filter refuses, and `corridor` reports 93.9% of the sinai mosaic reachable
because every slope is 0.0. One 29 MB Copernicus GLO-30 tile covers all three
AOIs, no auth.

**1.2 No reviewed regions, so no false-positive rate anywhere.** ↩ HANDOFF §9.3
~150 reviewed regions is the blocking question of S1, and the same labelled set
answers S2's shortlist-recall question and calibrates every severity number in
the repo. Today every S1 count is a *candidate* count.

**1.3 The aza class mapping is unconfirmed and nothing says so.** ✔ re-verified
The aza tiles carry no `ID_TO_LABEL_MAPPING` tag, so the sinai mapping is
borrowed via `--classes`. No provenance line is printed anywhere. Worse, without
`--classes` the mosaic path does not validate at all:
[mosaic.py:84](src/segmap_digest/mosaic.py#L84) is
`block = remap(raw, h["mapping"]) if h["mapping"] else raw.astype(np.uint8)` —
raw wire ids straight through as if dense, surfacing later as a bare `KeyError`
from `digests.py`. The single-tile path calls `_check` and gives an actionable
error; the mosaic path skips it. Under the borrowed mapping, "marl badlands" is
the dominant class inside dense urban Gaza.

> **CODE HALF FIXED 2026-08-31.** The mosaic path now calls `_check` (through
> `_check_tile`, which names the offending tile), so an untranslated
> sparse-wire-id tile fails at the file that caused it instead of as a `KeyError`
> three modules downstream. And a borrowed mapping announces itself on stderr:
> *"N of M tiles carry no ID_TO_LABEL_MAPPING tag and are read with the mapping
> from &lt;path&gt;. That mapping is from another export — if it is the wrong one,
> every class name on this AOI is wrong and nothing downstream can tell."* A tile
> carrying its own tag is not reported as borrowed. **The aza mapping itself is
> still unconfirmed** — this makes the borrowing visible, it does not make it
> right.

**1.4 The class notes are a programmer's paraphrase.** ↩ HANDOFF, `segmap notes`
The highest-leverage artifact in the repo is prose. `confused_with` already feeds
S2's shortlist and `scale` now overrides S2's guessed area bands
([class_notes.py:84](src/segmap_digest/class_notes.py#L84)); `season` and `never`
are recorded and read by nothing, and the template now says so honestly.

**1.5 No geological map, and the one that matters cannot be had.** ↩ `docs/story-per-tile.html`
Limestone/Dolomite/Nari is the largest single accuracy lever and S2 abstains on
exactly those cases. Three geological maps agree neither aza nor sinai contains
the rocks half the taxonomy is built around; leb is the first AOI that does
(Limestone 3.25%, Dolomite 1.01%, Nari 0.44%), and the Geological Survey of
Israel's service returns empty north of the border.

**1.6 The live LLM path has never made a successful request.** ✔ re-verified
`ANTHROPIC_API_KEY` is not set in this shell, and the key in `.env.rt` is **20
characters** — a stub. All 20 recorded runs returned HTTP 401; nothing was
billed. `segmap tools` works offline, so the surface is inspectable, but nothing
about the model's actual verb selection has been observed. **Do not just re-run
it — see 2.9 first.**

**1.7 A second real date — no longer blocked.** ✔ new today
`data/raw/2022-10-29.tif` and `data/raw/2025-06-06.tif` are the same footprint,
2.5 years apart, 197 Mpx each at 0.125 m/px. `segmap segment` turned 24 Mpx of
each into label rasters and `segmap solve s6` ran across them for the first time
on real ground. **The result is itself an open issue — see 5.2.**

---

## 2. Reproducible defects

Ranked by what a wrong answer costs.

**2.1 `segmap report --osm` is accepted and does nothing.** ✔ re-verified
[report.py](src/segmap_digest/report.py) contains **zero** occurrences of `osm`,
and `cmd_report` never wires the flag into S2 or S4 — so every
`Evidence.reference` in a `report` run is `None`, silently, and the page looks
identical with and without the flag. `segmap solve` and `segmap playground` both
wire it correctly; the gap is specific to `report`. This is also the command the
README and the run-rt skill lead with.
*(First recorded in `docs/five-fixes.html` "found along the way · not fixed".)*

> **FIXED 2026-08-31.** `report.build` takes `osm=` and hands it to
> `s1_audit.run`, `s2_adjudicate.adjudicate` and `s4_products.compute`; the CLI
> does one fetch over the union of layers the four sections need; and the page
> states which of the two cases it is in, because an S2 row reading
> `reference: none` because OSM disagrees and one reading it because no map was
> loaded mean opposite things. Verified on `data/incoming/leb --osm --max-mpx 8`:
> **313 ways burned**, and the S4 overlay now moves 8.64% of classified pixels
> (road 0.75 → 0.89, building 0.20 → 0.00). Found while fixing it: the report was
> also dropping S1's whole fragmentation block, which `render` had always
> carried. That is on the page now too.

**2.2 `s4_products.summarise` crashes on an all-nodata window.** ✔ reproduced today
[s4_products.py:604](src/segmap_digest/solutions/s4_products.py#L604) calls
`np.percentile(vals, 10)` where `vals = arr[raster.valid]` — empty when the AOI
window has no classified pixels, giving
`IndexError: index -1 is out of bounds for axis 0 with size 0` at 96% of a
playground job. Reproduced today from
`data/incoming/leb/tile_cropped_x3307_y3673_z0.125.tif`, which is 94% nodata and
whose centred 6 Mpx crop is **100%** nodata (0 valid pixels). Three preceding
`RuntimeWarning: Mean of empty slice` come from the same cause at lines 579, 580,
603. The correct behaviour is a stated "no classified pixels in this window", not
a traceback.

> **FIXED 2026-08-31.** `summarise` and `osm_delta_from` both state the empty
> window and return; the three `Mean of empty slice` warnings go with it. A
> window that *does* have data is unaffected, pinned as a control rather than
> assumed.

**2.3 The run-rt skill's own step 0 picks the tile that triggers 2.2.** ✔ new today
`.claude/skills/run-rt/SKILL.md` step 0 does
`ls data/incoming/leb/*.tif | head -1`, which lands on `x3307_y3673` — one of the
two 94%-nodata edge tiles. Measured over all six leb tiles: `x3307_*` give **0%**
classified after a 6 Mpx centre crop; the four `x3308_*`/`x3309_*` give 50–56%.
So the checklist fails on its own first choice every time it is run.

> **FIXED 2026-08-31.** Step 0 now ranks the tiles by classified fraction *after*
> the crop and takes the best, and says why. It also records that an all-nodata
> crop is a legitimate and now-stated outcome — just a useless thing to demo
> with.

**2.4 `class_distance` does not discriminate, and almost every finding depends on it.**
✔ re-verified by enumeration today: all **1,592** cross-superclass ordered pairs
return exactly **1.0**, and **91.0% of all 2,162 ordered class pairs** are at or
above the 0.70 `isolated-speck` threshold. So the "semantically distant host"
filter admits nearly everything, and the strangeness term it feeds into severity
is a constant for every finding it produces.
↩ inherited (not re-run — the sinai mosaic index is a ~24 min / 47 GB cold
rebuild): 97% of sinai's findings are `isolated-speck`, and the top two causes
are `DryGrassland` inside `Rendzina` and its inverse, 21,719 findings between
them about grass growing on soil.
**Do not fix this with a new prior**; that is the mistake `RETIRED_PRIORS` exists
to remember. It needs the class owner or a confusion matrix.

> **STATED, NOT FIXED, 2026-08-31** — the owner's choice of the four options.
> `s1_audit.speck_distance_caveat()` prints: *"this check is 'small and enclosed',
> not 'small, enclosed and semantically odd'. Its distant-host term admits 91% of
> all 2162 ordered class pairs at the 0.7 threshold, and all 1592
> cross-superclass pairs score exactly 1.0 … Closing this needs a measured
> confusion matrix or a compatibility relation from the class owner; it must NOT
> be closed by adding a prior."* It appears in `solve s1` and on the report page.
> The ranking is unchanged, and the numbers are computed at call time so they
> cannot drift from the taxonomy.

**2.5 `near` uses centroid distance; `distance` does it properly.** ✔ re-verified
[s5_query.py:226](src/segmap_digest/solutions/s5_query.py#L226) filters on
`dt[centroid]`, wrong for a long sinuous region whose centroid can lie outside
it. It emits a note saying so, which is honest and not a fix. `distance` is
already pixel-to-pixel and correct, so `near` should be reimplemented on it.

**2.6 The corridor noise floor is absolute over AOIs spanning four orders of magnitude.** ✔ re-verified
`CORRIDOR_MIN_AREA_M2 = 500.0` at
[s5_query.py:45](src/segmap_digest/solutions/s5_query.py#L45), running over
everything from a 200 m² fixture to a 175 km² mosaic. `f7ed72e` stopped it
reporting a confident `0.0% reachable` when it would drop *everything*, and
recorded the deeper fix — scaling it to the AOI — as an open question, because it
would change every corridor answer on real data.

**2.7 `min_area_px = 12` is in cells, so it means different things per AOI.** ✔ re-verified
[index.py:75](src/segmap_digest/index.py#L75). 12 cells is **0.180 m² on aza** and
**2.954 m² on sinai** — a 16× swing in what gets indexed at all. The same class
of mistake produced the aza worklist that ranked 300 m² buildings as isolated
specks, which is why `audit.py:59-96` now carries two separate windows (cells for
specks, m² for OOV candidates) with the reasoning written out.

**2.8 No output is georeferenced.** ↩ HANDOFF §9.10, S5 open question 4
`LabelRaster` carries the transform and CRS and **nothing emits lat/lon** —
every digest and query answer is pixel `cy`/`cx` plus a region id. A region id is
checkable against `segmap digest l2`, not against anyone's map. For someone who
has to act on a finding this is the gap between an answer and a usable answer.

**2.9 The S5 eval harness cannot test its own central claim.** ✔ re-verified
The claim is "every number traceable to a tool result".
[tools.py:356,378](src/segmap_digest/tools.py#L378) format the headline scalar
with `:,g` — 6 significant figures — so the model sees `203,551` where
`out/live_ask/truth/` stores `203,550.80`. A grader marks a correct model wrong;
inverted, **any model printing the exact ground-truth figure cannot have got it
from a tool**, so the harness awards full marks for the fabrication it exists to
catch. Also open from the same review: `ToolAnswer.audit()` persists summaries
containing no numbers (the payload the model actually consumed is discarded), and
`MAX_ITERATIONS = 12` exits silently in a way that reads as success. Fix these
before spending a valid key (1.6).

**2.10 `segmap ui` and `segmap showcase` findings are unaddressed.** ↩ `out/status/WEAKNESSES.md`, 2026-08-27
The sixteen-defect fix pass covered `tilemap_ui`, `playground`, `osm/burn`,
`tilechange`, `s3_triage`, `cli` and `class_notes` — it did not touch `webui.py`
or `showcase.py`, whose 16 findings include: `_LOCK` held across an entire
request; a negative region id producing a confident wrong answer;
`Content-Length: abc` → zero-byte response; any page on the internet being able
to drive the API via a `text/plain` POST; a clean bill of health rendered from
zero evidence; presence tested at full resolution but the figure rendered
decimated; and six identical recomputations per page. ✔ I re-verified one marker
of staleness: `_args` at [webui.py:90](src/segmap_digest/webui.py#L90) is defined
and called exactly nowhere.

**2.11 Documentation that is wrong in the same way twice.** ✔ re-verified
- [loader.py:407](src/segmap_digest/loader.py#L407) — `colormap()`'s docstring
  still says `"""(45, 3) uint8 RGB palette."""`; it builds `(47, 3)`.
- [README.md:30](README.md#L30) **and** [QUICKSTART.md:42](QUICKSTART.md#L42)
  both still praise the synthetic fixture for putting "terra rossa on hard
  carbonate" — the *retired* hypothesis, presented as a virtue, in the two files
  a newcomer reads first.
- `docs/solutions/S4-options.md` uses rendzina/terra rossa parent-rock inference
  as product signal in **C3 — Dust potential** and **C5 — Erosion
  susceptibility** (search by heading; the line numbers move every time the
  catalogue grows).
- S3's module and doc still use detector-dispatch/cost framing for a detector
  that does not exist.
- `report.py` has carried five ruff findings since before this session (four
  unused imports, one f-string with no placeholders). Confirmed present on `HEAD`
  as well as in the working tree, so they are not new — but it does mean nobody
  can use "ruff is clean" as a gate on this repo.
- **Two documents disagree about the same measurement.** HANDOFF §11's table —
  the one that says "all three verified by running `segmap solve s1` today"
  (2026-08-27) — gives the sinai mosaic **26,592 findings over 178 root causes**.
  `docs/solutions/S1-audit.md` open question 7 gives **41,669 across the mosaic**
  against 4,136 on one tile. Both are quoted as measurements of the same 20
  tiles. I did not re-run it (cold rebuild, ~24 min / 47 GB), so I cannot say
  which is current — but a repo whose whole argument is "numbers come from code"
  cannot carry two of them for one quantity. **Being settled 2026-09-02:**
  `segmap solve s1 -i data/incoming/sinai/` is re-running (cold rebuild over
  1,194 Mpx at 59.3% classified); the loser gets corrected.

**2.12 S6 has no guard against a pair that came from two different mappings.** ✔ new 2026-09-02
Run over the full 197 Mpx pair that 5.2 rejects, `s6_change` reports **84.1% of
co-valid pixels differ** and then adjudicates **61% of the differing area as
`real-change`**, printing *"71% of the differing area is plausibly REAL
change"*. It calls `House → Clutter` **demolition** over 48,255 m² and 2,579
components, and `PavedRoad → DirtRoad` **infrastructure** over 1,623 components.
Every one of those categories is defensible *given* two dates of one mapping —
and the whole point of 5.2 is that this is not that. The categoriser has no
precondition: nothing anywhere asks whether the two rasters are comparable
before their difference is described.

The check is cheap and the discriminators already are it: **built-class
retention** (a pair where 18% of `House` stays `House` and asphalt becomes dirt
is not two dates of one map) and **overall pixel agreement** (15.89% here).
Either belongs in front of `compare()` as a stated refusal, exactly like the
`registered=False` short-circuit that reviewers called exemplary. Without it, S6
is at its most confident precisely where it is most wrong — the failure shape
this repo names in its own design rules.

---

## 3. Scale walls

Not bugs. Places where the design stops working as the AOI grows, which is a
design question, not a tuning one.

**3.1 Only `l0` and `l1` survive an AOI.** ↩ HANDOFF §9.7, measured 2026-08-27
Sinai mosaic token estimates: `l0` **398**, `l1` **1,541**, `chips` **425,882**,
`l3` **1,701,818**, `l1q` **1,735,351**, `l2` **3,047,077**. On aza's single km²,
`l2` is **2,259,149**. `l2`/`l3` scale with region *count*, not extent, so the
lever is fragmentation. 38.8% of sinai's 18,104 chips are entirely nodata and are
emitted anyway.

**3.2 "Region" is not a meaningful unit on this data.** ↩ HANDOFF §9.8
The sinai mosaic's largest region is one `Rendzina` component of **86.23 km² —
49.4% of all region area — with 51,810 neighbours**. Top 10 regions hold 65.9%;
the median region is **26 m²**; 20.4% of indexed regions are under the 32-cell
speck floor. Half the map is one percolating blob whose compactness (0.00) says
nothing, and the other half is salt-and-pepper. Every per-region number inherits
this. Candidate fixes: a morphological opening upstream, merging same-class
components across thin gaps, superpixels, or working at chip level.

**3.3 `count` counts blobs, not objects.** ✔ verified, `data/incoming/sinai`
On `tile_cropped_x801_y846_z0.5.tif` the largest `House` component is 44,829 m²
and holds **64.5% of all House area** — the settlement is fused into one
"building", across 81 components. `Car` is fused too (median component 87 m²
where a car is ~10 m²). The same check on aza comes back clean at 6%. Same verb,
opposite verdict, and whether `count` counts objects is a property of the
segmentation rather than of the verb. Any answer phrased "there are 37 buildings"
is faithfully relaying a number that does not mean what was asked.

---

## 4. Decisions only the owner can make

● Each of these is a placeholder holding the shape of an answer nobody has given.
The full argument for each is in the named doc's "Open questions".

**S1 — audit.** Is a region the right unit (3.2)? Is severity comparable across
kinds, or is `SEVERITY_BAND` an ordering someone asserted? Does the audit run per
tile or per AOI — regions cut at a tile seam get wrong areas and wrong neighbour
lists, so the mosaic is the honest unit and the tile is the fast one (the
finding counts quoted for this are the ones that conflict; see 2.11)? Should priors become a reviewed data file rather than Python
dataclasses — the soil-genesis episode cost 10,788 false findings?

**S2 — adjudicate.** What replaces three hand-weighted terms (≈500 adjudicated
regions buys a calibrated logistic regression)? Where does the segmenter's own
confidence go — **flagged in the doc as the highest-value missing input**, and
newly relevant now that `segmap segment` runs the model in-repo (5.4). Should
`UNDECIDABLE` be per-evidence-type rather than lithology-only? How does an
accepted SWITCH propagate — edit the raster, emit a correction layer, or queue a
human confirmation? Should S2 ever run on *unflagged* regions?

**S3 — triage.** What is the detector's cost model: per call, per megapixel, or
per second — **answer this before optimising anything**, it inverts the chip
packing. Who pays for the 3% control set that is the only evidence the other 90%
of savings is safe? Who approves a hard-exclude list that removes 85% of an AOI,
or a "not applicable" verdict that says the *question* was wrong? Does the LLM
policy-generator beat the hand-written baseline (testable today — both are
checked in)?

**S4 — products.** Is a 0–1 score the right output at all, or GO/SLOW-GO/NO-GO,
or km/h, or a routing cost surface — **answer before tuning any constant**. Whose
doctrine owns the trafficability numbers? Which three of the 39 catalogued
products get built (the doc recommends GO/SLOW-GO/NO-GO, diggability, obstacle
inventory and states plainly that nothing is implemented)? Concealment from
*whom* — it is implicitly nadir EO. How do products compose?

**S5 — query.** What is the actual query distribution — ten real analyst
questions would redesign the module, and the six verbs are the ceiling on what
can be asked. Should answers be georeferenced (2.8)? Does the planner actually
plan (1.6)?

**S6 — change.** Co-registration is assumed and won't hold: a one-pixel shift
makes a boundary-following halo of fake transitions, and `MIN_EVENT_AREA_M2` does
not catch it because misregistration noise is numerous, small, and *aggregates*.
**Two dates or two model versions?** — if the segmenter was retrained between
runs, most "change" is model drift, and nothing distinguishes them today. This
one just stopped being hypothetical (5.2). Should transitions be localised to a
polygon layer instead of an aggregate table?

**Cross-cutting.** ✔ **DONE 2026-09-02** for the nari abstain: the verdict is now
`UNDECIDABLE-NEEDS-hardness-class`, and the rationale names which side of the
hard/soft carbonate split each candidate sits on and says outright that a nari
layer is not the thing to go looking for. `s2_adjudicate.HARDNESS` groups nari
with the hard carbonates, because that is what the crust behaves as at the
surface, which is the only place this taxonomy looks. The original reasoning:
retire `UNDECIDABLE-NEEDS-geological-map` for nari and rename it
`NEEDS-hardness-class`: nari is a surface calcrete crust over mapped bedrock
and no geological service answers it at any scale, while the map *does* split hard
from soft carbonate, which is what actually controls those classes. And: the
distribution is `reasoning-terrain` while the import package is `segmap_digest`
and the command is `segmap` — deliberately not renamed, worth one deliberate pass
if the name stops making sense, never as a side effect.

---

## 5. New surface: `segmap segment` (added 2026-08-31)

RGB ortho → wire-id label raster, via the Smart Terrain ONNX models. See
[segment.py](src/segmap_digest/segment.py). Open items it brings with it:

**5.1 The weights are named for a resolution they do not have.** ✔ verified
`models/ST_15cm_model.onnx` is byte-identical (md5 `88ba82f8…`) to the product's
`Merged_2025-02-13…` weights, whose own registry calls them
`RESOLUTION_12_5` — i.e. the "15cm" model is the **12.5 cm** model.
`ST_50cm_model.onnx` matches `Merged_2026-05-25…` and is genuinely 50 cm. The
code keys off `MODELS = {0.125: ..., 0.5: ...}` and ignores the filename, so
nothing is wrong today; the filenames are a trap for the next person. Rename, or
leave and keep the comment?

> **RENAMED 2026-08-31** to `ST_12_5cm_model.onnx`. `LEGACY_NAMES` keeps the old
> filename resolving — somebody else's `models/` directory should not break on
> our rename — and the canonical name wins when both are present.

**5.2 The two dates disagree about the ground far more than the ground changed.** ✔ new today
Same footprint, same crop, similar radiometry (mean RGB 131/103/85 vs
118/95/87), 24 Mpx each:

| | 2022-10-29 | 2025-06-06 |
|---|---|---|
| top classes | Rendzina 32%, Maquis 10%, Shadow 10%, UnirrigatedOrchard 9%, House 8% | DryGrassland 33%, Batha 17%, Clutter 11%, Garigue 8%, Rendzina 7% |

`segmap solve s6` reports **87.4% of co-valid pixels differ**, adjudicated 58%
real-change / 17% noise / 12% demolition. `Rendzina → DryGrassland` alone is
63,706 m² over 5,728 components, and `House → Clutter` is called demolition
across **630 components**. Either the ground genuinely did that, or the segmenter
reads the same ground differently across dates — which is exactly S6 open
question 2, now with real numbers attached.

> **DISCRIMINATORS RUN 2026-09-02 — the pair is not differenceable.** Three
> tests, all at the full 197 Mpx, none needing an opinion from outside the repo:
>
> 1. **Null test: all-quiet.** `s6_change.compare(A, A)` returns
>    `changed_frac 0.000000%`, 0 events, no categories. So the machinery is not
>    inventing change, and the disagreement is in the inputs.
> 2. **Built surface, which cannot flip with a season.** Of 2022's `House` area
>    only **18.2%** is `House` in 2025 — **45.5% becomes `Clutter`**. Of
>    `PavedRoad`, **18.0%** survives and **53.0% becomes `DirtRoad`**.
>    `BrickWall` retains **2.2%**, `Car` **0.0%**. Buildings do not vanish, and
>    asphalt does not become dirt.
> 3. **The whole map slides by superclass:** soil **−25.6 points** (34.7% →
>    9.1%), agriculture **−11.8** (12.0% → 0.2% — the orchards are gone),
>    vegetation **+40.6** (30.2% → 70.8%). Overall pixel agreement between the
>    two dates: **15.89%**.
>
> A whole-scene orchard-to-vegetation and asphalt-to-dirt slide is not phenology.
> **This is S6's open question 2 answered in the affirmative: these are not two
> dates of one map, they are two mappings.** Either the segmenter was retrained
> between the acquisitions or the domain shift moved the decision boundaries
> wholesale, and *which* cannot be settled from inside this repo — it needs the
> model version for each acquisition, which is exactly what S6 asked to have
> logged. **Consequences:** no S6 number on this pair is reportable; the next S6
> feature is model-version logging, not co-registration hardening; and 1.7 is
> only half an unblock — we have a second date, not a comparable one.
> Full output: `out/summary/discriminators.log`.

**5.3 `roads` and `dsem_landcover` are dropped.** ✔ by design, today
Both come free in the same forward pass — `roads` is a binary mask,
`dsem_landcover` one float band. Nothing in RT consumes either, so they are
discarded rather than written out to rot. `roads` is an obvious independent check
against the OSM road layer (2.1's `Evidence.reference` has no second opinion
today); `dsem_landcover` is unidentified — nobody here knows what its float
means.

**5.4 The model's per-pixel confidence is thrown away at source.** ✔ new today
The `preds` output is already arg-maxed inside the graph, so there is no
probability to keep — which is the same gap S2 names as its *highest-value
missing input*. The product has other exports (`soft_preds`, 45–48 channels) in
the sibling Dynamic-Terrain drop. Getting a soft-pred export of these two
resolutions would feed S2 directly.

**5.5 Operational facts worth writing down.** ✔ measured today
0.09 Mpx/s at the default `--tile 1024`, peak **18 GB RSS**; `--tile 2048` is
~25% faster per useful pixel and peaked at **67 GB**. So ≈35 min per 197 Mpx
ortho on 28 CPU cores. `--gpu` asks for `CUDAExecutionProvider` and falls back
with a warning — the plain `onnxruntime` wheel has none, and this box has a
Quadro RTX 6000 sitting idle. Neither full ortho has been segmented end to end;
only the centred 24 Mpx of each.

**5.6 Environment drift found while making this run.** ✔ today
`onnxruntime` was absent from the env `segmap` is installed in (I installed
1.29.0). `python -m build`, which run-rt step 5 calls for, is also absent — I
used `pip wheel --no-deps --no-build-isolation`, same backend. Both belong in a
pinned dev environment rather than in whatever a session happens to install.

---

## 6. What is *not* open

So that a reader does not go looking. Cleared by reviewers who ran the code:
`WayFootprint.local_mask` reproduces its bbox exactly across 441 cases at every
raster edge; there is no row/col swap anywhere; every empty-input path is clean;
`page_results` escapes every interpolation it controls; S2 is bit-identical
across 60 adjudications when OSM is silent about everything. The S1 coverage
block, `osm_delta`, the `registered=False` short-circuit and S3's three-way
rejection vocabulary were called exemplary. Suite: **257 passed, 1 skipped** as
of today.

---

## 7. The questions

Ranked by how much they unblock. Each is a question because the code cannot
answer it.

**Q1 — Which input do you get first: the DEM, or ~150 reviewed regions?**
→ **ANSWERED: the DEM.** Not started; it is the next piece of work.

The DEM unblocks 23 classes, four abstaining checks, two solutions and the
corridor verb, and is one 29 MB download with no auth (1.1). The labelled set
unblocks S1's precision number, S2's shortlist recall and every severity ordering
in the repo (1.2), and needs a person's time. They are the top two items on every
existing list and neither has moved. *Recommendation: DEM first — it is an
afternoon, and it changes what the reviewed set would even be reviewing.*

**Q2 — Is the two-date disagreement real change or model drift, and who settles it?**
→ **ANSWERED: run the discriminators. Done — and they came back against the
pair.** Null test all-quiet, built surface retaining 18%, asphalt becoming dirt,
15.89% pixel agreement (5.2). What remains is not a code question: it needs the
**model version for each acquisition** from whoever produced the imagery.
87.4% of pixels differ between your two orthos (5.2). Until that is answered, S6
on real data cannot be reported, and the answer decides whether S6's next feature
is co-registration hardening or model-version logging. *Cheapest path: I run the
null test and a known-unchanged crop, which is code and costs you nothing.*

**Q3 — Do I fix the four defects that make a command lie, right now?**
→ **ANSWERED: yes. Done** — see the decisions table above.
`report --osm` silently ignored (2.1), `s4_products.summarise` crashing on an
all-nodata window (2.2), the run-rt skill picking the tile that triggers it
(2.3), and the mosaic path skipping validation so a borrowed mapping crashes
three modules downstream instead of warning (1.3). All four are small, all four
are already diagnosed, and three of them are on the paths the README leads with.
*Recommendation: yes, as one commit with tests.*

**Q4 — `class_distance`: confusion matrix, or a compatibility relation?**
→ **ANSWERED: neither yet — state the limitation instead. Done** (2.4).
Nearly every sinai finding is an `isolated-speck`, and the filter behind them
admits 91% of all class pairs — measured today (2.4). Two ways out: measure a confusion matrix
(needs 1.2) or have the class owner declare which pairs are *compatible*
(vegetation over soil is not a contradiction). Adding a prior is explicitly the
wrong answer. *This one is yours or the class owner's; I cannot invent it.*

**Q5 — Can you get a soft-pred export of the 12.5 cm and 50 cm models?**
→ open.
S2 names the segmenter's own confidence as its highest-value missing input, and
`preds` arg-maxes it away inside the graph (5.4). The sibling Dynamic-Terrain
drop has 45- and 48-channel `soft_preds` models, so the export path exists —
just not for these two resolutions. *If yes, S2 stops guessing its candidate
list from the taxonomy.*

**Q6 — Which three of the 39 S4 products, and as a score or as classes?**
→ open.
The catalogue is written, the recommendation is on the page (GO/SLOW-GO/NO-GO,
diggability, obstacle inventory), and the doc says outright that nothing is
implemented because someone still has to choose. The score-vs-classes question
must be answered first: it changes the output contract, not the tuning.

**Q7 — Where should engineering effort go once Q3 is done?**
→ **ANSWERED: georeference the answers (2.8).** Accepted, not started — it is
the next piece of work. Four candidates were offered,
in the order I would take them: **(a)** georeference the answers (2.8) — it is
the difference between an answer and a usable one, and it is contained; **(b)**
the `ui`/`showcase` findings (2.10), one of which lets any page on the internet
drive the local API; **(c)** the S5 eval harness (2.9), which must precede
spending a real key; **(d)** the region-unit problem (3.2), which is the largest
and the least contained. *Or say "none of these, do X" — that is what this
question is for.*

**Q8 — Two small ones, for completeness.**
→ **ANSWERED: both yes.** The rename is done (5.1). Both full orthos are being
segmented as of 2026-08-31; the 24 Mpx crops stay under `out/labels/` and the
full pair lands in `out/labels/full/`.
