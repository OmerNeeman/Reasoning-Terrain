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
