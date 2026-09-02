---
name: run-rt
description: Run the Reasoning Terrain pipeline end-to-end as a live smoke test -- baseline tests, a real segmap report run, a live playground job, the per-fix regression tests, and the packaging install check. Use this whenever the user asks to "run RT", "run the RT demo", "verify RT works", or wants to see the pipeline actually run rather than just read about it. This is the exact "Run it yourself" checklist from docs/five-fixes.html, kept here so it stays runnable instead of going stale as prose.
---

# Run RT

Five steps, each one a real command against real code -- no synthetic
narration. Run them from the repo root, in order. After each step, report
a one-line pass/fail before moving to the next; if a step fails, stop and
report it rather than pushing on and burying the failure under later
output. At the end, give one short summary (pass/fail per step), not a
transcript dump.

## 0. Pick an input raster

Steps 2 and 3 need a label raster. `data/incoming/leb/*.tif` is real,
gitignored customer data (`data/incoming/*` is excluded except its
README) -- it may not exist on this checkout.

```bash
ls data/incoming/leb/*.tif 2>/dev/null | head -1
```

- If that lists a file, use `data/incoming/leb` as `-i`/upload input for
  steps 2 and 3.
- **For step 3's single-tile upload, do not just take the first one.**
  `data/incoming/leb` has two 94%-nodata edge tiles (`x3307_*`) whose
  centred 6 Mpx crop is *entirely* nodata, and `head -1` lands on one of
  them every time. Pick a tile with data:
  ```bash
  for f in data/incoming/leb/*.tif; do
    python -c "
import sys
from segmap_digest import loader
r = loader.load(sys.argv[1], classes='src/segmap_digest/data/smart_terrain_class_ids.json')
c = loader.crop_to_max_mpx(r, 6.0)
print(f'{1 - c.nodata_frac:6.1%}  {sys.argv[1]}')" "$f"
  done | sort -r | head -1
  ```
  On this drop the four `x3308_*`/`x3309_*` tiles come back 50-56%
  classified and the two `x3307_*` come back 0.0%. An all-nodata crop is
  a legitimate input and `s4_products.summarise` now says so rather than
  raising -- but it exercises none of the pipeline, so it is the wrong
  tile to demo with.
- If it's empty, generate a synthetic tile instead and say so plainly
  (don't silently substitute):
  ```bash
  segmap synth -o /tmp/rt-synth-tile
  ```
  then use `/tmp/rt-synth-tile.tif` (check the actual extension `synth`
  wrote) as the input for steps 2 and 3 instead of `data/incoming/leb`.

## 1. Confirm the baseline

```bash
pytest tests -q
```
Expect: all green, 0 failed. The exact pass count drifts as the suite
grows -- don't hardcode "223 passed" as a pass/fail gate, just require
`0 failed`.

## 2. Run the real pipeline

```bash
segmap report -i <INPUT> --osm --max-mpx 8 -o out/demo-fixes/report --chip 128
```
(`<INPUT>` = `data/incoming/leb` or the synthetic fallback from step 0.)
Expect: `wrote out/demo-fixes/report/index.html` with no traceback.
Note for whoever reads the output: `--osm` is accepted here but
`report.py` does not currently wire it into S2/S4 (a known gap, not a
bug in this skill) -- don't be surprised the report looks identical
with or without the flag.

## 3. Watch a real job through the playground

Start the server in the background, submit one real upload, poll it to
completion, then stop the server -- don't leave it running.

```bash
segmap playground --port 8711 &
SERVER_PID=$!
sleep 1
curl -s -D - -o /dev/null -X POST http://127.0.0.1:8711/run \
    -F "file=@<ONE_TIF_FROM_INPUT>;type=image/tiff" -F "max_mpx=6"
# read the Location header for the job id, then:
curl -s http://127.0.0.1:8711/api/job/<job-id>
# poll every couple seconds until "status":"done" (or "failed" -- report either)
kill $SERVER_PID
```
If port 8711 is already taken, try 8712/8713 rather than failing outright.
Expect: the job reaches `"status": "done"` with `"pct": 100`, or an honest
`"status": "failed"` with a clear `error` -- either is a valid outcome to
report, a hang with no poll ever returning is not.

## 4. Re-run each fix's own regression tests

```bash
pytest tests/test_osm.py -k "s2_reference_penalty or s2_renormalizes or s2_is_bit_identical" -v   # fix 1: S2 ranking scale
pytest tests/test_playground.py -v                                                              # fix 2: job timeout + LRU
pytest tests/test_osm.py -k scale_note -v                                                        # fix 3: scale-note area override
pytest tests/test_tilemap.py -k "nan or unmeasured or builtin_scores_nodata" -v                  # fix 4: NaN-means-unmeasured
```
Expect: every test in each of these four runs passes.

## 5. Verify the package installs non-editably

Build a wheel, install it in a throwaway venv, and confirm the bundled
data resolves without the repo checkout in sight. Clean up afterward --
don't leave the venv, dist dir, or build/ artifacts behind.

```bash
python -m build --wheel --outdir /tmp/rt-dist .
python -m venv /tmp/rt-venv && /tmp/rt-venv/bin/pip install /tmp/rt-dist/*.whl
cd /tmp && /tmp/rt-venv/bin/python -c \
    "from segmap_digest import class_notes, playground; \
     print(class_notes.DEFAULT_PATH.is_file(), playground.DEFAULT_CLASSES.is_file())"
# expect: True True -- and the printed paths should be under
# .../site-packages/segmap_digest/data/, not this repo checkout
rm -rf /tmp/rt-dist /tmp/rt-venv
rm -rf <repo-root>/build /tmp/build  # python -m build can drop build/ in the repo root; remove it if so
```

## When done

One short summary, e.g.:

```
1. baseline tests       -- PASS (0 failed)
2. real pipeline run    -- PASS (out/demo-fixes/report/index.html written)
3. live playground job  -- PASS (status: done, pct 100)
4. per-fix regressions  -- PASS (4/4 groups green)
5. packaging install    -- PASS (both bundled files resolved, no repo path)
```

If step 0 fell back to a synthetic tile, say so in the summary -- it
changes what steps 2 and 3 actually exercised.
