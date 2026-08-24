"""Synthetic Smart Terrain label rasters, so the POC runs with zero data.

Not a simulator -- a fixture. It produces spatially coherent regions with
plausible co-occurrence (terra rossa on hard carbonate, rendzina on chalk,
badlands only where it is steep, maquis on shaded slopes, a village with roads
and cars) so the digest and audit code has something structured to chew on.

Swap it for a real tile via `loader.load()` the moment you have one.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .loader import LabelRaster
from .taxonomy import cid


def _noise(rng: np.random.Generator, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Smooth [0,1] field with features roughly `scale` pixels across."""
    field = rng.standard_normal(shape)
    field = ndi.gaussian_filter(field, sigma=scale, mode="wrap")
    lo, hi = field.min(), field.max()
    return (field - lo) / (hi - lo + 1e-9)


def _fractal(rng: np.random.Generator, shape: tuple[int, int], scales, weights) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    for s, w in zip(scales, weights):
        out += w * _noise(rng, shape, s)
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-9)


def _path_mask(rng, shape, width_px, axis=1, wiggle=40.0):
    """A smooth meandering band across the tile -- roads, tracks."""
    h, w = shape
    n = w if axis == 1 else h
    walk = np.cumsum(rng.standard_normal(n))
    walk = ndi.gaussian_filter1d(walk, sigma=n / 12.0, mode="nearest")
    walk = walk / (np.abs(walk).max() + 1e-9) * wiggle
    centre = (h if axis == 1 else w) / 2 + walk

    mask = np.zeros(shape, dtype=bool)
    coord = np.arange(h if axis == 1 else w)
    for i in range(n):
        band = np.abs(coord - centre[i]) <= width_px / 2
        if axis == 1:
            mask[band, i] = True
        else:
            mask[i, band] = True
    return mask


def generate(
    size: int = 1024,
    gsd: float = 0.3,
    seed: int = 7,
    with_dem: bool = True,
    median_slope_deg: float = 7.0,
) -> LabelRaster:
    rng = np.random.default_rng(seed)
    shape = (size, size)
    s = size / 1024.0  # scale factor so features look the same at any size

    # --- terrain -----------------------------------------------------------
    # Scale relief so the median slope lands on a target, rather than picking a
    # relief in metres -- otherwise slope depends on tile size and gsd and the
    # whole scene comes out as cliff.
    elev = _fractal(rng, shape, (110 * s, 45 * s, 16 * s), (1.0, 0.45, 0.18))
    gy0, gx0 = np.gradient(elev, gsd)
    median_tan = float(np.median(np.hypot(gx0, gy0)))
    target_tan = np.tan(np.radians(median_slope_deg))
    elev *= target_tan / max(median_tan, 1e-9)

    gy, gx = np.gradient(elev, gsd)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)          # radians, 0 = north-ish
    northness = np.cos(aspect)            # +1 north-facing (shaded, wetter)
    wetness = 1.0 - (elev - elev.min()) / (np.ptp(elev) + 1e-9)

    # --- lithology zones ---------------------------------------------------
    lz = _fractal(rng, shape, (200 * s, 70 * s), (1.0, 0.3))
    litho = np.full(shape, "Limestone", dtype=object)
    litho[lz < 0.22] = "Basalt"
    litho[(lz >= 0.22) & (lz < 0.38)] = "Marl"
    litho[(lz >= 0.38) & (lz < 0.50)] = "Chalk"
    litho[(lz >= 0.50) & (lz < 0.68)] = "Dolomite"
    # Nari caps: thin bands at slope breaks near local highs.
    crest = (elev > np.percentile(elev, 72)) & (slope > 4) & (slope < 12)
    litho[crest & (lz >= 0.68)] = "Nari"

    veg = _fractal(rng, shape, (30 * s, 10 * s), (1.0, 0.4))
    stone = _fractal(rng, shape, (22 * s, 8 * s), (1.0, 0.5))

    out = np.zeros(shape, dtype=np.uint8)

    # --- rock / soil / vegetation by rule ---------------------------------
    def put(mask, name):
        out[mask & (out == 0)] = cid(name)

    steep = slope > 22
    mid = (slope > 9) & (slope <= 22)
    gentle = slope <= 9

    # Marl: badlands where steep and dissected, smooth slopes mid, terrace flat.
    marl = litho == "Marl"
    put(marl & steep, "MaralBadlands")
    put(marl & mid, "MaralSmoothRockSlopes")
    put(marl & gentle & (stone < 0.55), "MaralTerrace")
    put(marl & gentle, "Rendzina")

    # Chalk: only ever smooth slopes; rendzina below.
    chalk = litho == "Chalk"
    put(chalk & (slope > 7), "ChalkSmoothRockSlopes")
    put(chalk, "Rendzina")

    # Basalt.
    bas = litho == "Basalt"
    put(bas & steep & (stone > 0.62), "BasaltBoulder")
    put(bas & (slope > 12), "BasaltRockyTerrain")
    put(bas & (stone > 0.5), "BasaltStoneyTerrain")
    put(bas, "ClayeySoil")

    # Nari caps.
    nari = litho == "Nari"
    put(nari & (stone > 0.62), "NariRockyTerrain")
    put(nari & (slope < 5), "NariTerrace")
    put(nari & (slope > 16) & (np.abs(northness) > 0.7), "NariRockDipSlope")
    put(nari, "NariStoneyTerrain")

    # Hard carbonate: limestone gets the full morphology set, dolomite fewer.
    for rock, has_full in (("Limestone", True), ("Dolomite", False)):
        m = litho == rock
        put(m & steep & (stone > 0.68), f"{rock}Boulder")
        put(m & steep, f"{rock}RockyTerrain")
        if has_full:
            planar = mid & (np.abs(northness) > 0.75) & (stone < 0.45)
            put(m & planar, "LimestoneRockDipSlope")
            put(m & mid & (stone > 0.6), "LimestoneBeddedRock")
        put(m & mid, f"{rock}StoneyTerrain")
        put(m & gentle & (stone < 0.35), f"{rock}Terrace")
        # Terra rossa: residual clay of hard carbonate, on gentle ground.
        put(m & gentle & (veg < 0.55), "TerraRosa")
        put(m & gentle, f"{rock}StoneyTerrain")

    # Vegetation overprints gentle-to-mid slopes; the degradation series tracks
    # moisture and aspect.
    veg_ok = (slope < 26) & np.isin(out, [cid(n) for n in (
        "TerraRosa", "ClayeySoil", "Rendzina", "LimestoneStoneyTerrain",
        "DolomiteStoneyTerrain", "NariStoneyTerrain", "BasaltStoneyTerrain",
        "LimestoneTerrace", "DolomiteTerrace", "NariTerrace", "MaralTerrace")])
    moisture = 0.55 * veg + 0.30 * np.clip(northness, 0, 1) + 0.15 * wetness
    out[veg_ok & (moisture > 0.68)] = cid("Maquis")
    out[veg_ok & (moisture > 0.58) & (moisture <= 0.68)] = cid("Garigue")
    out[veg_ok & (moisture > 0.47) & (moisture <= 0.58)] = cid("Batha")
    out[veg_ok & (moisture > 0.40) & (moisture <= 0.47)] = cid("DryGrassland")

    # --- hydrology ---------------------------------------------------------
    low = wetness > 0.93
    low = ndi.binary_opening(low, np.ones((7, 7)))
    out[low] = cid("Water")
    fringe = ndi.binary_dilation(low, np.ones((15, 15))) & ~low
    out[fringe] = cid("HydromorpicSoil")
    valley = (wetness > 0.80) & (slope < 4) & (out != cid("Water")) & ~fringe
    out[valley] = cid("ClayeyDeepSoil")

    # --- agriculture on flat deep soil ------------------------------------
    flat = (slope < 8) & np.isin(out, [
        cid("ClayeyDeepSoil"), cid("ClayeySoil"), cid("TerraRosa"),
        cid("DryGrassland"), cid("Batha"), cid("Rendzina")])
    for k, name in enumerate(("IrrigatedField", "UnirrigatedOrchard", "IrrigatedOrchard")):
        for _ in range(2):
            fh = int(rng.integers(60 * s, 150 * s))
            fw = int(rng.integers(80 * s, 190 * s))
            r0 = int(rng.integers(0, max(size - fh, 1)))
            c0 = int(rng.integers(0, max(size - fw, 1)))
            box = np.zeros(shape, dtype=bool)
            box[r0:r0 + fh, c0:c0 + fw] = True
            sel = box & flat
            if sel.sum() > (fh * fw) * 0.2:       # only if it mostly lands on flat ground
                out[sel] = cid(name)
                if k > 0:                          # orchards: leave inter-row gaps
                    rows = np.zeros(shape, dtype=bool)
                    rows[r0:r0 + fh:max(int(9 * s), 2), c0:c0 + fw] = True
                    out[sel & ~rows] = cid("DryGrassland")

    # --- roads -------------------------------------------------------------
    paved = _path_mask(rng, shape, max(7 * s, 3), axis=1, wiggle=90 * s)
    out[paved] = cid("PavedRoad")
    for _ in range(2):
        track = _path_mask(rng, shape, max(4 * s, 2), axis=0, wiggle=140 * s)
        out[track] = cid("DirtRoad")
    out[_path_mask(rng, shape, max(3 * s, 2), axis=1, wiggle=200 * s)] = cid("DirtRoadB")

    # --- village -----------------------------------------------------------
    vr = int(rng.integers(size * 0.15, size * 0.75))
    vc = int(rng.integers(size * 0.15, size * 0.75))
    n_houses = int(22 * max(s, 0.5))
    for _ in range(n_houses):
        hr = vr + int(rng.normal(0, 70 * s))
        hc = vc + int(rng.normal(0, 70 * s))
        hh = int(rng.integers(10 * s, 26 * s)) or 4
        hw = int(rng.integers(10 * s, 30 * s)) or 4
        if not (0 <= hr < size - hh and 0 <= hc < size - hw):
            continue
        out[hr - 3:hr + hh + 3, hc - 3:hc + hw + 3] = cid("Pavement")
        out[hr:hr + hh, hc:hc + hw] = cid("House")
        # Cast shadow to the north-west, as a nadir scene with a southern sun.
        sh = max(int(4 * s), 2)
        out[hr - sh:hr, hc - sh:hc + hw] = cid("Shadow")
        if rng.random() < 0.45:                    # a wall on some plots
            out[hr + hh + 2:hr + hh + 3, hc - 3:hc + hw + 3] = cid("BrickWall")

    # cars on trafficable surfaces near the village
    near = np.zeros(shape, dtype=bool)
    r0, r1 = max(vr - int(160 * s), 0), min(vr + int(160 * s), size)
    c0, c1 = max(vc - int(160 * s), 0), min(vc + int(160 * s), size)
    near[r0:r1, c0:c1] = True
    parkable = near & np.isin(out, [cid("Pavement"), cid("PavedRoad"), cid("DirtRoad")])
    ys, xs = np.nonzero(parkable)
    if len(ys):
        for i in rng.choice(len(ys), size=min(30, len(ys)), replace=False):
            y, x = ys[i], xs[i]
            out[y:y + max(int(5 * s), 2), x:x + max(int(9 * s), 3)] = cid("Car")

    # --- out-of-vocabulary things land in Clutter / Unclassified ----------
    for _ in range(6):
        r = int(rng.integers(0, size)); c = int(rng.integers(0, size))
        rr = int(rng.integers(8 * s, 30 * s)) or 5
        cc = int(rng.integers(8 * s, 30 * s)) or 5
        out[r:r + rr, c:c + cc] = cid("Clutter" if rng.random() < 0.7 else "Unclassified")

    return LabelRaster(
        labels=out, gsd=gsd,
        dem=elev.astype(np.float32) if with_dem else None,
    )


def second_date(base: LabelRaster, seed: int = 99) -> LabelRaster:
    """A plausible 'date 2' for the same tile, for S6 change reasoning.

    Deliberately mixes the four things a change detector has to tell apart:
    seasonal phenology, vegetation succession, genuine construction, and label
    noise -- including a lithology flip, which cannot physically happen and is
    always a labelling error.
    """
    rng = np.random.default_rng(seed)
    out = base.labels.copy()
    h, w = out.shape
    s = max(h / 1024.0, 0.25)

    # 1. Season: the dry season has arrived.
    out[out == cid("GreenGrassland")] = cid("DryGrassland")

    # 2. Succession: a burnt-then-regrowing patch shifts along the series.
    patch = np.zeros(out.shape, dtype=bool)
    pr, pc = int(rng.integers(0, h * 0.7)), int(rng.integers(0, w * 0.7))
    patch[pr:pr + int(180 * s), pc:pc + int(180 * s)] = True
    out[patch & (out == cid("Maquis"))] = cid("Garigue")
    out[patch & (out == cid("Garigue"))] = cid("Batha")

    # 3. Construction: a new building cluster on open ground.
    br, bc = int(rng.integers(0, h * 0.8)), int(rng.integers(0, w * 0.8))
    for _ in range(6):
        r = br + int(rng.normal(0, 40 * s)); c = bc + int(rng.normal(0, 40 * s))
        hh, hw = int(14 * s) or 5, int(18 * s) or 6
        if 0 <= r < h - hh and 0 <= c < w - hw:
            out[r - 2:r + hh + 2, c - 2:c + hw + 2] = cid("Pavement")
            out[r:r + hh, c:c + hw] = cid("House")

    # 4. Infrastructure: one dirt road gets sealed.
    dirt = out == cid("DirtRoad")
    comp, n = ndi.label(dirt, structure=np.ones((3, 3)))
    if n:
        biggest = 1 + int(np.argmax(np.bincount(comp.ravel())[1:]))
        out[comp == biggest] = cid("PavedRoad")

    # 5. Label noise, including an impossible lithology flip.
    flip = np.zeros(out.shape, dtype=bool)
    fr, fc = int(rng.integers(0, h * 0.8)), int(rng.integers(0, w * 0.8))
    flip[fr:fr + int(60 * s), fc:fc + int(60 * s)] = True
    for a, b in (("LimestoneStoneyTerrain", "DolomiteStoneyTerrain"),
                 ("LimestoneRockyTerrain", "DolomiteRockyTerrain")):
        out[flip & (out == cid(a))] = cid(b)

    return LabelRaster(labels=out, gsd=base.gsd, transform=base.transform,
                       crs=base.crs, dem=base.dem)
