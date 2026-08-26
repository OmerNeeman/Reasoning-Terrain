# S5 round-trip check — round 1

**Date:** 2026-08-26 · **Branch:** `claude/review-findings` @ `ed64a5d`
**Tile:** `data/incoming/sinai/tile_cropped_x801_y846_z0.5.tif`
8192×8192 px @ 0.496 m/px · 67.1 Mpx · 4065 × 4065 m · **100.0% classified** ·
15543 regions · 1024 chips

## Outcome: BLOCKED — the model half of the check did not run

Twenty `segmap ask` invocations (10 questions × {opus-5/high, sonnet-5/medium})
all terminated on the first API request with:

```
anthropic.AuthenticationError: Error code: 401 -
{'type': 'error', 'error': {'type': 'authentication_error',
 'message': 'API key is invalid.'}}
```

20 of 20 runs, identical failure, ~2 s each. No tokens were billed.

**Cause: the key in `.env.rt` is not a valid key.** It loads correctly — the
variable is populated and begins `sk-ant-` — but it is **20 characters** long.
A real Anthropic API key is ~100+ characters. The `${VAR:+set}` pre-flight in
the protocol passed because it only tests non-emptiness, not validity.

Consequently **none** of the five graded columns can be filled in: no verbs were
called, no numbers were produced, no refusal was relayed, no verdict is
available. Grading is deferred, not failed. Recording this faithfully rather
than simulating it, per instruction.

## What did run, and passed

Everything on the offline side. The three pre-flight gates and the entire
ground-truth pass completed against the real raster.

| Step | Result |
|---|---|
| `pip install -e '.[geo,llm]'` | ok |
| `python -m pytest tests -q` | **151 passed, 1 failed, 1 skipped** — see *Caveats* |
| Key loads into the shell | yes (but see above — loading ≠ valid) |
| Tile has `ID_TO_LABEL_MAPPING` | **yes** — no `--classes` needed |
| `segmap index` | 15543 regions in 14.2 s, 1024 chips in 34.5 s, cached |
| Ground truth for Q1–Q6 | computed, saved under `truth/` |
| Abstention paths Q7–Q9 | computed offline, all three refuse correctly |

## Ground truth (offline, `segmap solve s5 --query`)

These are the numbers the model's answers must be graded against when the key
is fixed. Saved verbatim in `truth/`.

| Q | Query | Result | Evidence |
|---|---|---|---|
| 1 | `describe` | 30 classes. Rendzina 65.30%, DryGrassland 24.98%, Batha 3.64%, DirtRoad 1.23%, MaralBadlands 1.01%, PavedRoad 0.68%, House 0.42%, Car 0.02% | per-class region counts |
| 2 | `count House minarea 40` | **37** | region ids 1216, 1201, 1227, … (+12) |
| 3 | `area DirtRoad` | **203,550.80 m²** | 220 regions |
| 3 | `area PavedRoad` | **112,506.51 m²** | 85 regions |
| 3 | ratio (derived) | dirt ≈ **1.81×** paved | — |
| 4 | `distance MaralBadlands DirtRoad` | **0.50 m** — one pixel; they touch | rid 1817 @ (3320,3977) ↔ rid 1586 @ (3320,3976) |
| 5 | `corridor DirtRoad vehicle wheeled` | **97.3%**, 16,086,883.54 m², 1 component | 3 pockets totalling 152 m² dropped as noise |
| 6 | `count Car` | **24** | region ids 1312, 1292, 1300, … (all 24 listed) |

**Q5 assumptions the answer must state** (all three are in the tool note, so a
model that omits them is dropping information it was handed):
traffic threshold **0.45** · slope limit **25.0°** · vehicle width **2.5 m
(5 px opening)**, passages narrower than that removed.

### Abstention paths — the code half is confirmed correct

| Q | Probe | Response |
|---|---|---|
| 7 | `count Truck` | *"'Truck' is outside the 47-class vocabulary: the taxonomy has exactly one vehicle class, Car. Vehicle **type** was never predicted"* |
| 8 | `count School` | *"'School' is outside the 47-class vocabulary: the taxonomy has one building class, House. Building **function** is not predicted and is not visible from nadir anyway"* |
| 9 | `find Rendzina slope > 25` | *"no DEM is attached to this tile, so every region's slope is unmeasured (stored as 0.0). A slope filter cannot be applied"* |

The code-side half of the refusal contract holds. **The model-side half — whether
it relays the refusal instead of substituting a near-miss class or answering
zero — remains untested**, which is precisely the thing this round trip existed
to measure.

## Partial answer to S5 open question 2 — is the six-verb surface enough?

The planner was never exercised, so this is answered from verb coverage only,
not from model behaviour. Of the ten questions:

- **Six are cleanly covered** by a single verb or a pair (Q1 `describe`,
  Q2 `count`+minarea, Q3 two `area` calls, Q4 `distance`, Q5 `corridor`,
  Q6 `count`).
- **Q3 needs arithmetic the verbs do not do.** "How does it compare" requires a
  ratio across two tool results. Nothing computes it, so the model must divide
  203,550.80 by 112,506.51 itself — **a number in the prose that no tool
  returned.** This is a real gap in the "no un-traceable numbers" guarantee, and
  it is visible without a single API call. Either a `compare`/`ratio` verb, or
  the grading rule has to permit arithmetic over quoted tool outputs.
- **Q7–Q9 are covered by refusal**, which is a designed answer, not a gap.
- **Q10 has no verb at all.** "Which parts do you suspect are misclassified" is
  what S1 computes and what `--digest` was built for. Whether the model stays
  inside the verbs, points at `--digest`, or invents findings is untestable here
  — and it is the single most informative of the ten.

Verdict on the surface: adequate for five of ten, silently arithmetic-dependent
for one, deliberately refusing on three, and uncovered for one.

## Overall verdict — zero un-traceable numbers?

**Unproven, per model.** Not "yes" and not "no": no model produced any number.
The property is untested for both `claude-opus-5` and `claude-sonnet-5`.

## Caveats

1. **No DEM.** Nothing on this tile has a measured slope; every `mean_slope` and
   `aspect_circvar` is 0.0 meaning *unmeasured*. S1 abstains on the slope and
   aspect priors, S2 scores morphology as unscored, and Q9 is unanswerable by
   construction. Any trafficability or corridor figure inherits this.
2. **Borrowed class mapping — NOT used.** The chosen sinai tile carries its own
   `ID_TO_LABEL_MAPPING` GeoTIFF tag (47 entries), so class names come from the
   file itself. `examples/smart_terrain_class_ids.json` was not passed. This
   caveat, anticipated in the run request, does not apply. It *would* have
   applied to the `aza` tiles, which carry no tag.
3. **One pre-existing test failure**, present before any change in this run and
   not fixed here: `tests/test_real_raster.py::test_nodata_is_not_trafficable`
   expects `100.0% of the classified area` and gets `0.0%`. The fixture's whole
   reachable area is 200 m², below `CORRIDOR_MIN_AREA_M2 = 500`, so the noise
   floor swallows the entire answer while the note still reports the pocket.
   It touches Q5's verb directly. Recorded, not repaired.

## Files

```
questions.txt        the ten questions, tab-separated
run.sh               the runner (tag, model, effort)
truth/               offline ground truth, one file per query
opus/q01..q10.txt    20 captured runs — all 401, kept as the failure record
sonnet/q01..q10.txt
```

## To resume

Replace the key in `.env.rt`, confirm it is ~100 chars, then:

```bash
(set -a; source .env.rt; set +a; bash out/live_ask/run.sh opus   claude-opus-5   high)
(set -a; source .env.rt; set +a; bash out/live_ask/run.sh sonnet claude-sonnet-5 medium)
```

The index is cached, so each question costs seconds plus model latency.
