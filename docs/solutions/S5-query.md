# S5 — Natural-language querying

`segmap solve s5 --query "..."` · [`s5_query.py`](../../src/segmap_digest/solutions/s5_query.py)

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
count <class>
area  <class>
corridor <seed-class> [vehicle wheeled|tracked|foot]
```

`corridor` is the interesting one: threshold the S4 trafficability raster,
connected-component it, and report which components touch the seed class. That
answers "where can a wheeled vehicle actually get to from the road network"
without any LLM involvement.

`llm_handoff()` routes interpretive questions to a digest by keyword —
deliberately crude. The point is that routing *exists*, not that these keywords
are right.

## Heuristic constants — all guesses

| Constant | Value | Where it should come from |
|---|---|---|
| `CORRIDOR_MIN_TRAFFIC` | 0.45 | Inherits every uncertainty in S4's `traffic` scale |
| `CORRIDOR_MIN_AREA_M2` | 500 | What size of isolated pocket is worth reporting |
| `llm_handoff` keyword lists | ad hoc | Should be a classifier or the model's own tool choice |

---

## Open questions

1. **Who parses the natural language?** Right now: nobody — you type the
   structured query yourself. The two options are (a) the LLM emits these
   queries as tool calls, or (b) the LLM writes Python against the region/chip
   index directly. **(a) is safer and more auditable; (b) is far more
   expressive.** This is the main design decision and it isn't made.
2. **What's the actual query distribution?** The four verbs are guesses at what
   people ask. Ten real analyst questions would redesign this module.
3. **Are the answers verifiable?** A query returns numbers with no provenance. A
   user can't tell a correct answer from a wrong one. Every result should be able
   to emit the region ids or a mask it was computed from.
4. **Should results be georeferenced?** Currently pixel coordinates. Real use
   needs lat/lon or a GeoJSON export — `LabelRaster` carries the transform but
   nothing uses it.
5. **Multi-tile / AOI queries.** Everything is single-tile. "Where in this
   50 km² AOI" needs an index across tiles and a different cost profile.
6. **`near` uses centroid distance.** For a long sinuous region the centroid can
   be far from every part of it. Should be minimum distance from the region's
   pixels — cheap to fix, currently wrong.
7. **How does an interpretive answer get grounded?** `llm_handoff` returns a
   digest and a question, and then nothing checks the answer against the index.

## Options and extensions

- **Make the verbs tools, and let the model compose them.** `find`, `count`,
  `area`, `corridor`, plus `distance`, `between`, `along`. The model plans; the
  code computes. This is the recommended next step.
- **Programmatic tool calling / code execution.** Let the model write Python
  against the index inside a sandbox, so intermediate results never enter the
  context window. Much more expressive than a fixed verb set, and the right
  answer once the query distribution turns out to be long-tailed (it will).
- **Answer + evidence, always.** Every response carries the region ids, a mask,
  or a GeoJSON feature collection, so a human can check it on the map.
- **Query provenance in the output.** Which digest was used, which heuristics
  and thresholds were involved. A corridor answer that doesn't mention
  `CORRIDOR_MIN_TRAFFIC = 0.45` is hiding its main assumption.
- **Saved queries as named products.** The queries people actually repeat become
  S4-style products with a stable definition.
- **Negative answers matter.** "No Houses near a paved road in this tile" is a
  real answer and currently renders as an empty table. Say it in words.

## How to evaluate

- **Golden set.** 30–50 questions with hand-computed answers, run as tests. The
  quantitative verbs are exactly testable — do this, it's cheap.
- **Round-trip check.** Ask the LLM a question, have it emit tool calls, compare
  against the golden answer. Measures the planner, not the arithmetic.
- **Refusal rate.** How often does the system correctly say "the map can't answer
  that" (e.g. anything about vehicle *type*, building *function*, or fence
  presence — none are in the taxonomy)? Silent plausible answers to
  unanswerable questions are the main failure mode.
