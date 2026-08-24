# S5 — Natural-language querying

`segmap solve s5 --query "..."` · `segmap ask "<question>"` · `segmap tools`
[`s5_query.py`](../../src/segmap_digest/solutions/s5_query.py) ·
[`tools.py`](../../src/segmap_digest/tools.py) ·
[`ask.py`](../../src/segmap_digest/ask.py)

**What it is.** Answer questions about the map. Split by construction:

- **Quantitative** (areas, counts, distances, adjacency, connectivity) → answered
  by code. In-context arithmetic over a large table is exactly where LLMs go
  wrong, and exactly what's easy to compute.
- **Interpretive** → handed to the LLM, with the *narrowest* digest that can
  support the answer.

The tiny query language here is **not the user interface**. It's the tool
surface an LLM would call. Having it exist first is what keeps the model out of
the arithmetic.

---

## Naive algorithm (implemented)

```
find <class> [minarea <m2>] [slope < N | slope > N] [near <class> <m>]
count <class>          # same filters as find
area  <class>          # same filters as find
distance <class> <class>
describe
corridor <seed-class> [vehicle wheeled|tracked|foot]
```

`corridor` is the interesting one: threshold the S4 trafficability raster,
connected-component it, and report which components touch the seed class. That
answers "where can a wheeled vehicle actually get to from the road network"
without any LLM involvement.

Every `QueryResult` carries `ids` — the region ids (or component / class ids)
the answer was computed from — and `render()` prints them. `distance` is
pixel-to-pixel closest approach and returns both endpoints, so it can be checked
on the map; `near` is still centroid-based and says so in its own note.

`llm_handoff()` routes interpretive questions to a digest by keyword —
deliberately crude. The point is that routing *exists*, not that these keywords
are right.

## The tool layer (implemented)

[`tools.py`](../../src/segmap_digest/tools.py) exposes the verbs as typed
Anthropic tool definitions. `segmap ask "<question>"` runs the SDK's tool runner
(`claude-opus-5`, adaptive thinking) over them; `segmap ask --digest` keeps the
old interpretive path for questions no verb covers.

- **One implementation.** A tool call is compiled into a query *string* and
  handed to `s5_query.query`. The compiled string is in the payload, so any
  answer can be reproduced with `segmap solve s5 --query '...'`.
- **Evidence in every payload**: `evidence.ids` (capped at `EVIDENCE_MAX_IDS`),
  `ids_total`, `ids_omitted`, and a line saying where to look them up. Rows are
  capped at `TOOL_MAX_ROWS` and the payload states how many were dropped.
- **Four statuses**, because these are four different answers: `ok`, `empty`
  (a measured zero — a finding), `unanswerable` (the label raster cannot answer
  this at all), `error` (malformed call, returned as a payload so the model can
  correct itself).
- **Abstention.** A slope filter with no DEM attached is `unanswerable`, not
  "0 degrees". A class outside the vocabulary is `unanswerable` with the reason
  from `UNANSWERABLE_CONCEPTS` — vehicle type, building function, fence
  presence, anything about colour.
- **Prompt cache.** Tool definitions render before the system prompt and the
  legend sits behind the existing cache breakpoint, so `SPECS` is a fixed tuple
  and the definitions are emitted in a stable order.

The tool definitions, the dispatch, the evidence payload and the refusals are
unit-tested with no SDK installed
([`tests/test_tools.py`](../../tests/test_tools.py)). **The live API path is
unexercised** — there is no key in this environment.

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `CORRIDOR_MIN_TRAFFIC` | 0.45 | Inherits every uncertainty in S4's `traffic` scale |
| `CORRIDOR_MIN_AREA_M2` | 500 | What size of isolated pocket is worth reporting |
| `EVIDENCE_MAX_IDS` | 25 | How many ids a reviewer will actually spot-check |
| `TOOL_MAX_ROWS` | 40 | Token budget per tool result vs. how much the model needs |
| `MAX_ITERATIONS` | 12 | Observed plan length across real questions — none observed yet |
| `UNANSWERABLE_CONCEPTS` | ad hoc | The refusal list is hand-written; it should come from the questions users actually ask |
| `llm_handoff` keyword lists | ad hoc | Should be a classifier or the model's own tool choice |

---

## Open questions

1. ~~**Who parses the natural language?**~~ **Decided: the model emits tool
   calls against these verbs — safe and auditable.** There is no
   code-execution path, and adding one is a separate decision, not an
   implementation detail of this one. What that buys: the model cannot produce
   a number the code did not compute, every call is logged with its query
   string, and the refusal cases are enforced in code rather than trusted to
   the prompt. What it costs: the verb set is the ceiling on what can be asked,
   and question 2 below is now the binding constraint.
2. **What's the actual query distribution?** The six verbs are guesses at what
   people ask. Ten real analyst questions would redesign this module.
3. ~~**Are the answers verifiable?**~~ **Done, partly.** Every result carries the
   region ids it was computed from and the tool payload caps the sample with the
   total stated. Still missing: a mask or GeoJSON export, and any check that the
   ids the model *quotes* are the ids the tool *returned*.
4. **Should results be georeferenced?** Currently pixel coordinates. Real use
   needs lat/lon or a GeoJSON export — `LabelRaster` carries the transform but
   nothing uses it. This is now the largest gap in the evidence story: region
   ids are checkable against `segmap digest l2`, but not against anyone's map.
5. **Multi-tile / AOI queries.** Everything is single-tile. "Where in this
   50 km² AOI" needs an index across tiles and a different cost profile.
6. **`near` uses centroid distance.** For a long sinuous region the centroid can
   be far from every part of it. Should be minimum distance from the region's
   pixels — `distance` now does exactly that, so `near` should be reimplemented
   on top of it. Currently `find near ...` states the flaw in its own note
   rather than fixing it.
7. **How does an interpretive answer get grounded?** `llm_handoff` returns a
   digest and a question, and then nothing checks the answer against the index.
   `--digest` is now the explicitly-labelled un-grounded path.
8. **Does the planner actually plan?** Unknown. The tool surface is tested; the
   model's use of it is not — no key in this environment, so not one live call
   has been made. The round-trip check below is the first thing to run.
9. **Is `count` counting the right thing?** It counts connected components, so
   two houses sharing a wall are one, and anything under the index's min-area
   cut is zero. The tool description says so, which is not the same as being
   right.

## Options and extensions

- ~~**Make the verbs tools, and let the model compose them.**~~ Done: `find`,
  `count`, `area`, `distance`, `describe`, `corridor`. `between` and `along` are
  the obvious next two, and both need question 4 (georeferencing) first.
- **Programmatic tool calling / code execution.** Let the model write Python
  against the index inside a sandbox, so intermediate results never enter the
  context window. Much more expressive than a fixed verb set, and the right
  answer once the query distribution turns out to be long-tailed (it will).
  Deliberately not built: it trades the audit trail for expressiveness, and the
  audit trail is what makes an answer usable here.
- ~~**Answer + evidence, always.**~~ Done for region / component / class ids.
  A mask or GeoJSON feature collection is still open (question 4).
- **Query provenance in the output.** Partly done — every payload carries the
  compiled query string and the thresholds a verb applied. Not done: which
  *version* of the heuristics produced it.
- **Saved queries as named products.** The queries people actually repeat become
  S4-style products with a stable definition.
- ~~**Negative answers matter.**~~ Done: `status: empty` says it in words and is
  a different payload from `status: unanswerable`.

## How to evaluate

- **Golden set.** 30–50 questions with hand-computed answers, run as tests. The
  quantitative verbs are exactly testable — do this, it's cheap. Currently the
  tests check consistency with `s5_query` and the shape of the payload, not
  correctness against hand-computed truth (except `distance`, which is checked
  against brute force).
- **Round-trip check.** Ask the LLM a question, have it emit tool calls, compare
  against the golden answer. Measures the planner, not the arithmetic. **Not yet
  run — no API key in this environment.**
- **Refusal rate.** How often does the system correctly say "the map can't answer
  that" (e.g. anything about vehicle *type*, building *function*, or fence
  presence — none are in the taxonomy)? Silent plausible answers to
  unanswerable questions are the main failure mode. The code-side half is
  enforced (`UNANSWERABLE_CONCEPTS`, and abstention on slope without a DEM); the
  model-side half — whether it relays the refusal instead of substituting a
  near-miss class — is untested.
