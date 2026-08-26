# `segmap ui` — the demo UI

```bash
segmap ui -i data/incoming/sinai/tile_cropped_x801_y846_z0.5.tif
# -> http://127.0.0.1:8765
```

The raster and both indices load once at startup, so the first question costs a
second rather than the ~50 s an index build costs. With a warm
`.segmap_cache/`, startup is a few seconds.

Three tabs, and the split between them is the point of the repo:

| Tab | Needs a key? | What it is |
|---|---|---|
| **Verbs** | no | `s5_query` driven by hand — `describe`, `count`, `area`, `find`, `distance`, `corridor`. Every answer carries the region ids it was computed from. This is the surface a model would be given. |
| **Solutions** | no | S1–S6, each rendered exactly as its CLI renders it. **None of them calls a model at runtime.** |
| **Ask** | **yes** | The one path in the repo that calls a model. It plans; the verbs compute. The tool-call log renders beneath the prose, so a claim and its evidence are on screen together. |

Two of the three tabs are fully offline. `Ask` is the only thing a missing or
invalid key takes away, and it degrades to a message rather than an error.

The left panel is the colourised label raster (decimated for display; every
number is computed at full resolution) and the class legend. Clicking a class
drops its name into whichever box is open.

## The Ask tab needs the key in the server's environment

The key is read by the server process, not the browser. Start it with the key
already loaded:

```bash
(set -a; source .env.rt; set +a; segmap ui -i <tile>)
```

Without it the tab returns the same actionable message `segmap ask` prints,
rather than a traceback — and the other two tabs are unaffected.

## Notes

- Stdlib `http.server` only; no new dependencies. It binds to `127.0.0.1` and is
  a demo server, not a deployment.
- Requests are serialised behind one lock. There is one user.
- **S6 needs a second-date raster** of the same footprint. There is only one date
  in `data/incoming` for any tile, so that panel explains itself and does
  nothing until you give it a path.
- **S1 and S2 abstain on slope and aspect without a DEM**, and say so in their
  own output. That is the honest answer, not a gap in the UI.

---

# `segmap showcase` — one HTML page about your map

```bash
segmap showcase -i /path/to/your-raster.tif -o out/showcase.html
# add --classes ids.json if the GeoTIFF has no ID_TO_LABEL_MAPPING tag
```

Writes a single self-contained `.html` — images base64-inlined, no sidecar
directory — explaining all six solutions and illustrating each from the raster
you point it at. Roughly a minute on a 67 Mpx tile once the index is cached,
3–6 MB out.

What it renders, all computed from your file:

- the colourised tile, and the class legend with real fractions
- **class isolations** — roads, built environment, vehicles, rock morphology and
  the vegetation series, each highlighted on a ghosted basemap
- **full-resolution detail crops**, auto-located on the densest window for each
  group, so the settlement is found without being told where it is
- **all four S4 products** plus a dry-vs-wet difference map
- the `corridor` passable-ground raster next to the verb's own output
- **S1's top-ranked flagged regions boxed on the map**, beside the cause table
- **S3's triage decision drawn as chips** — red dispatched, blue control set
- worked S5 examples including the two refusals
- a measured answer to *"does `count` count the right thing?"* for your raster

That last one is worth knowing about. `count` counts connected components, so
whether it counts *objects* depends on whether your segmentation fuses them. The
page measures it rather than asserting it, and the answer differs by map: on
`sinai/x801_y846` the largest House component holds 64% of all House area — the
whole settlement is one "building" — while on `aza/x3237_y3495` the largest holds
6% and the counts are trustworthy. Same verb, opposite verdict.

## Nothing on either page calls a model

`segmap ui`'s Ask tab is the only path in the repo that does. `showcase` never
does, and all six solutions are deterministic.
