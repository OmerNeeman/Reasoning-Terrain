"""S4 -- derived decision products.

This is where 47 classes become useful: a many-to-one mapping from the taxonomy
plus slope plus season into the things someone actually asks for. Tedious to
hand-code for 47 x N combinations, which is exactly why it is worth having a
model help author the tables -- but the evaluation must stay deterministic.

Products implemented (naive):
    trafficability  can a vehicle class cross this
    concealment     how well is a ground object hidden from above
    drainage        where does water pool / run
    fire_fuel       fuel load for fire spread

Each returns a float raster in [0, 1] plus a per-region summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..index import RegionIndex
from ..loader import LabelRaster
from ..taxonomy import BY_ID, N_CLASSES, SUPERCLASS_OF, cid

# --- tunable heuristics (all guesses -- see docs/solutions/S4.md) ----------


@dataclass(frozen=True)
class Vehicle:
    name: str
    max_slope_deg: float
    # Minimum surface trafficability the vehicle needs (taxonomy `traffic`).
    min_surface: float
    # How much wet ground degrades performance, 0 = unaffected.
    wet_sensitivity: float
    # Physical width in metres. S5's corridor query uses it to reject
    # passages narrower than the vehicle (single-pixel percolation paths).
    # 0.0 = no width gate (foot). Guesses -- where these should come from:
    # vehicle spec sheets, the same document that owns max_slope_deg.
    width_m: float = 0.0


VEHICLES = {
    "wheeled": Vehicle("wheeled", max_slope_deg=25.0, min_surface=0.45,
                       wet_sensitivity=0.7, width_m=2.5),
    "tracked": Vehicle("tracked", max_slope_deg=35.0, min_surface=0.20,
                       wet_sensitivity=0.4, width_m=3.5),
    "foot": Vehicle("foot", max_slope_deg=45.0, min_surface=0.05,
                    wet_sensitivity=0.15, width_m=0.0),
}

# Slope response: full score below SLOPE_FREE, zero at the vehicle limit.
SLOPE_FREE_DEG = 5.0

# Classes that hold water / stay wet, driving the wet penalty.
WET_CLASSES = ("HydromorpicSoil", "ClayeyDeepSoil", "ClayeySoil", "Water",
               "IrrigatedField", "IrrigatedOrchard")

# Concealment: canopy comes from the taxonomy; these add the rest.
SHADOW_CONCEALMENT = 0.6
BUILT_CONCEALMENT = 0.5      # adjacent to buildings/walls
RELIEF_CONCEALMENT_WEIGHT = 0.25   # steep, broken ground hides things too

# Window (px) over which slope roughness is measured for the relief term.
# Where this should come from: the footprint of the object being hidden
# divided by the GSD -- a fixed pixel count is only defensible at one GSD.
ROUGHNESS_WINDOW_PX = 5

# Window (metres) for the local relative elevation ("lowness") term in
# drainage. Where this should come from: the drainage-basin scale of the
# terrain -- roughly the hillslope length over which runoff converges. 200 m
# is a guess for dissected Mediterranean terrain.
DRAINAGE_LOCAL_WINDOW_M = 200.0

# Fire fuel: canopy-driven, scaled by dryness of the class.
DRY_CLASSES = ("DryGrassland", "Batha", "Garigue", "Maquis", "UnirrigatedOrchard")
FIRE_DRY_BONUS = 0.3


def _slope(raster: LabelRaster) -> np.ndarray:
    if raster.dem is None:
        return np.zeros(raster.shape, dtype=np.float32)
    gy, gx = np.gradient(raster.dem.astype(np.float64), raster.gsd)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _per_class(attr) -> np.ndarray:
    return np.array([attr(BY_ID[c]) for c in range(N_CLASSES)], dtype=np.float32)


def _mask_of(labels: np.ndarray, names) -> np.ndarray:
    return np.isin(labels, [cid(n) for n in names])


def trafficability(raster: LabelRaster, vehicle: str = "wheeled",
                   wet: bool = False) -> np.ndarray:
    v = VEHICLES[vehicle]
    surface = _per_class(lambda d: d.traffic)[raster.labels]
    slope = _slope(raster)

    slope_term = np.clip(
        1.0 - (slope - SLOPE_FREE_DEG) / max(v.max_slope_deg - SLOPE_FREE_DEG, 1e-6),
        0.0, 1.0,
    )
    score = surface * slope_term
    score[surface < v.min_surface] = 0.0
    if wet:
        score = np.where(_mask_of(raster.labels, WET_CLASSES),
                         score * (1.0 - v.wet_sensitivity), score)
    return score.astype(np.float32)


def _local_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Local standard deviation over a size x size window.

    Vectorised as sqrt(E[x^2] - E[x]^2) with two uniform_filter passes.
    Numerically equivalent -- edges included, both replicate the border via
    mode="nearest" -- to `ndi.generic_filter(arr, np.std, size, mode="nearest")`,
    which invokes a Python callback per pixel and takes hours at mosaic scale.
    """
    a = arr.astype(np.float64)      # float32 E[x^2]-E[x]^2 cancels catastrophically
    m = ndi.uniform_filter(a, size=size, mode="nearest")
    m2 = ndi.uniform_filter(a * a, size=size, mode="nearest")
    # Rounding can push the variance a hair below zero on flat ground; sqrt of
    # that is NaN, so clamp.
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def concealment(raster: LabelRaster) -> np.ndarray:
    canopy = _per_class(lambda d: d.canopy)[raster.labels]
    score = canopy.copy()
    score = np.maximum(score, _mask_of(raster.labels, ("Shadow",)) * SHADOW_CONCEALMENT)

    built = _mask_of(raster.labels, ("House", "BrickWall"))
    near_built = ndi.binary_dilation(built, np.ones((9, 9))) & ~built
    score = np.maximum(score, near_built * BUILT_CONCEALMENT)

    slope = _slope(raster)
    roughness = _local_std(slope, ROUGHNESS_WINDOW_PX) \
        if slope.any() else np.zeros_like(slope)
    if roughness.max() > 0:
        score += RELIEF_CONCEALMENT_WEIGHT * (roughness / roughness.max())
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def drainage(raster: LabelRaster) -> np.ndarray:
    """Where water pools. Naive: locally low ground + flat + a water-holding class.

    Lowness is LOCAL relative elevation, not a whole-raster normalisation: on a
    mosaic spanning regional relief, a valley floor at high absolute elevation
    is exactly where water accumulates, and `1 - (e - min)/ptp` scores it as
    dry ridge simply because somewhere else in the AOI is lower.
    """
    slope = _slope(raster)
    flat = np.clip(1.0 - slope / 10.0, 0.0, 1.0)
    holding = _mask_of(raster.labels, WET_CLASSES).astype(np.float32)
    if raster.dem is not None:
        e = raster.dem.astype(np.float64)
        # Odd window so the neighbourhood is centred on the pixel.
        window_px = max(3, int(round(DRAINAGE_LOCAL_WINDOW_M / raster.gsd)) | 1)
        rel = ndi.uniform_filter(e, size=window_px, mode="nearest") - e
        # rel > 0: below the local mean, i.e. accumulating. Normalise by a
        # robust scale (p95 of |rel|) rather than the max, so one deep pit does
        # not flatten every ordinary valley to ~0; clip keeps [0, 1].
        scale = float(np.percentile(np.abs(rel), 95))
        low = np.clip(rel / max(scale, 1e-9), 0.0, 1.0).astype(np.float32)
    else:
        low = np.full(raster.shape, 0.5, dtype=np.float32)
    return np.clip(0.45 * flat + 0.30 * low + 0.25 * holding, 0, 1).astype(np.float32)


def fire_fuel(raster: LabelRaster) -> np.ndarray:
    canopy = _per_class(lambda d: d.canopy)[raster.labels]
    dry = _mask_of(raster.labels, DRY_CLASSES).astype(np.float32)
    grass = _mask_of(raster.labels, ("DryGrassland",)).astype(np.float32)
    return np.clip(canopy + FIRE_DRY_BONUS * dry + 0.25 * grass, 0, 1).astype(np.float32)


PRODUCTS = {
    "trafficability": trafficability,
    "concealment": concealment,
    "drainage": drainage,
    "fire_fuel": fire_fuel,
}


def compute(raster: LabelRaster, product: str, **kw) -> np.ndarray:
    arr = PRODUCTS[product](raster, **kw)
    if raster.valid is not None:
        # Nodata carries class id 0 (Unclassified), whose traffic score is 0.5.
        # Left alone, a third of an arid crop scores as moderately driveable open
        # ground, and S5's corridor query drives straight across it.
        arr = np.where(raster.valid, arr, 0.0).astype(np.float32)
    return arr


def summarise(raster: LabelRaster, arr: np.ndarray, ridx: RegionIndex | None = None,
              top_k: int = 12) -> str:
    """Per-superclass means, plus the worst/best regions if an index is given."""
    ok = raster.valid
    vals = arr if ok is None else arr[ok]
    lines = [
        f"# value distribution over classified pixels: mean {vals.mean():.2f}, "
        f"p10 {np.percentile(vals, 10):.2f}, p90 {np.percentile(vals, 90):.2f}",
        "",
        "## mean by superclass",
        "superclass\tmean\tarea_frac",
    ]
    sc_of = np.array([SUPERCLASS_OF[c] for c in range(N_CLASSES)])
    per_px = sc_of[raster.labels]
    denom = max(raster.n_valid, 1)
    for sc in sorted(set(SUPERCLASS_OF.values())):
        m = per_px == sc
        if ok is not None:
            m &= ok
        if not m.any():
            continue
        lines.append(f"{sc}\t{arr[m].mean():.2f}\t{m.sum() / denom:.3f}")

    if ridx is not None:
        lines += ["", f"## top {top_k} regions by mean value", "rid\tclass\tarea_m2\tmean"]
        vals = ndi.mean(arr, ridx.label_array,
                        index=[r.id for r in ridx.regions]) if ridx.regions else []
        pairs = sorted(zip(ridx.regions, vals), key=lambda p: -p[1])[:top_k]
        for r, v in pairs:
            lines.append(f"{r.id}\t{r.class_name}\t{r.area_m2:.0f}\t{float(v):.2f}")
    return "\n".join(lines)


def to_png(arr: np.ndarray, path: str) -> None:
    """Viridis-ish ramp without pulling in matplotlib."""
    from PIL import Image

    x = np.clip(arr, 0, 1)
    rgb = np.stack([
        (np.clip(1.6 * x - 0.35, 0, 1) * 255),
        (np.clip(1.2 * x + 0.05, 0, 1) * 255),
        (np.clip(1.1 - 1.3 * x, 0, 1) * 255),
    ], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(path)
