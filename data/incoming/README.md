# Drop real data here

Contents of this directory are gitignored (except this file) — nothing you put
here gets committed.

## Minimum to get off synthetic data

| File | Format | Notes |
|---|---|---|
| `classes.*` | anything readable — JSON, CSV, YAML, a Python dict, a screenshot of the config | **The class-id → name mapping the model actually emits.** Send this even if you send nothing else |
| `tile_01.tif` | single-band GeoTIFF | the label raster. Ids may be the sparse "wire" ids the model emits (0..241) — if the file carries an `ID_TO_LABEL_MAPPING` tag we read it automatically, otherwise pass `--classes`. `.npy` or single-band PNG also fine. **Not** a colourised RGB export |
| — | metres/pixel | if it's not in the GeoTIFF header, just tell me |

## Next most valuable

| File | Notes |
|---|---|
| `tile_01_dem.tif` | elevation, any resolution — say which. Worth more than RGB for this taxonomy: dip slope / terrace / badlands / boulder are all morphology |
| acquisition date | settles irrigated-vs-unirrigated and green-vs-dry grassland |
| `tile_02.tif`, `tile_03.tif` | different terrain (carbonate / basalt / with a settlement) — shows which priors fire everywhere and mean nothing |

## Big multipliers, only if they already exist

| File | Unlocks |
|---|---|
| geological / soil map | the Limestone/Dolomite/Nari separation — the largest single accuracy lever. S2 abstains on exactly these cases today |
| second date, same footprint | makes S6 real; every `impossible` transition it finds is free ground truth |
| validation confusion matrix | replaces guessed candidate lists in S2 with measured ones |
| any reviewed/corrected regions | even 50 gives S1 a false-positive rate |

## Then

```bash
segmap report -i data/incoming/tile_01.tif --gsd 0.3 -o out/
open out/index.html
```
