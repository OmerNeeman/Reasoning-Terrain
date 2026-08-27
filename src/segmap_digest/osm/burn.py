"""Vectors -> pixels, and the co-registration check that says whether to trust it.

Three jobs.

**Projection.** lon/lat through the raster's own inverse affine. If the raster is
not in WGS84 the coordinates are reprojected first, via rasterio; there is no
hand-rolled datum maths here, because a 30 m offset from a wrong ellipsoid looks
exactly like a real disagreement between the two maps.

**Rasterisation.** A road is a centreline plus a width, so its footprint is a
capsule per segment -- a rectangle with two round ends. Two implementations, and
the choice is about cost, not correctness:

  the capsule burn walks each segment inside its own bounding box, so the work is
    proportional to the *corridor area* (23% of the aza AOI [measured]) rather
    than to the raster. `binary_dilation(iterations=k)` is k full passes over the
    whole array, which on the 1194 Mpx sinai mosaic with a 122 px trunk-road
    buffer is 146 billion pixel operations for a road network covering a few
    percent of the ground.
  polygons (buildings, water bodies, landuse) are filled with an even-odd
    scanline inside their bounding box, or by `rasterio.features.rasterize` when
    rasterio is installed, which is the same answer faster.

**Per-way composition, during the burn.** While the local mask for a way exists,
its ST class histogram is accumulated. That is the whole input to S1's
reference checks -- "what does ST call the ground under this street" -- and
getting it here costs nothing, where getting it afterwards would need an int32
way-id raster (4.8 GB on the sinai mosaic) or a second pass.

`align_report` is the gate on all of it. Two maps of the same ground are only
comparable if they are registered to each other; the report states the integer
shift that maximises agreement, so a systematic offset shows up as a number
instead of as thousands of "ST missed this road" findings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..taxonomy import N_CLASSES, cid
from . import tags as T
from .vectors import OsmVectors, OsmWay

# Layers this module knows how to burn, and what goes into each.
LAYERS = ("road", "footway", "building", "water", "flow", "barrier", "built_landuse")

# Half-width used for a waterway line (a mapped stream has no width tag). A wadi
# in this terrain is metres wide, not tens.
FLOW_HALF_WIDTH_M = 2.0
# A mapped wall or fence is a line with no width; one metre either side is enough
# to make it findable in a label raster at 0.1-0.5 m/px.
BARRIER_HALF_WIDTH_M = 1.0

# Shift search half-range for the co-registration report, in metres of ground.
# Wide enough to catch a datum or origin blunder, not so wide that it starts
# matching a *different* street.
ALIGN_SEARCH_M = 6.0
# ST road pixels sampled for the shift search. 200k is far more than enough for
# a stable argmax and keeps the search O(shifts * sample) instead of
# O(shifts * pixels) -- 121 full passes over a 1.2 Gpx mask is minutes.
ALIGN_SAMPLE = 200_000


# --- projection ------------------------------------------------------------

def to_pixels(raster, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 degrees -> (row, col) float arrays in this raster's grid."""
    t = getattr(raster, "transform", None)
    if t is None:
        raise ValueError("raster has no affine transform; OSM cannot be projected "
                         "onto it (see osm.fetch.bbox_of_raster)")
    x, y = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    crs = getattr(raster, "crs", None)
    if crs is not None and not _is_wgs84(crs):
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform("EPSG:4326", crs, x.tolist(), y.tolist())
        x, y = np.asarray(xs), np.asarray(ys)
    inv = ~t
    cols, rows = inv * (x, y)
    return np.asarray(rows, dtype=np.float64), np.asarray(cols, dtype=np.float64)


def _is_wgs84(crs) -> bool:
    try:
        return int(getattr(crs, "to_epsg", lambda: 0)() or 0) == 4326
    except Exception:                    # pragma: no cover
        return "4326" in str(crs)


# --- the burn ------------------------------------------------------------

@dataclass
class WayFootprint:
    """One way, on this grid, with what ST says is underneath it."""
    way_id: int
    idx: int                       # index into OsmVectors.ways
    kind: str
    label: str
    grade: str | None
    surface: str
    half_width_m: float
    length_m: float                # inside the raster
    n_px: int                      # footprint pixels inside the valid mask
    hist: np.ndarray               # (N_CLASSES,) pixel counts under the footprint
    bbox: tuple[int, int, int, int]
    # Pixel geometry, kept so the footprint can be re-realised without
    # re-projecting: a few thousand ways of a few dozen vertices is a couple of
    # megabytes, and it is what lets `partition` ask "which blocks does this
    # street bound?" without an int32 way-id raster over the whole mosaic.
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cols: np.ndarray = field(default_factory=lambda: np.zeros(0))
    half_px: float = 1.0
    # Which routine drew it. A polygon and a buffered line pad their bounding
    # boxes differently, so re-realising a filled building with the capsule
    # routine returns a mask of a different SHAPE than `bbox` -- which fails
    # loudly, but only once something indexes the two together.
    is_area: bool = False

    def local_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Re-burn this way's footprint. Same routine, same bbox, same result."""
        if len(self.rows) < 2:
            return None
        if self.is_area:
            local, _bbox, _p = _fill_polygon(self.rows, self.cols,
                                             shape[0], shape[1])
        else:
            local, _bbox, _len = _burn_capsule(self.rows, self.cols, self.half_px,
                                               shape[0], shape[1], 1.0)
        return local

    @property
    def frac(self) -> np.ndarray:
        return self.hist / max(self.n_px, 1)

    def share(self, *class_names: str) -> float:
        ids = [cid(n) for n in class_names]
        return float(self.hist[ids].sum()) / max(self.n_px, 1)

    def dominant(self, k: int = 3) -> list[tuple[int, float]]:
        if self.n_px == 0:
            return []
        order = np.argsort(self.hist)[::-1][:k]
        return [(int(c), float(self.hist[c] / self.n_px)) for c in order
                if self.hist[c] > 0]


@dataclass
class BurnedOsm:
    """Rasterised OSM, on the label raster's grid."""
    shape: tuple[int, int]
    gsd: float
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    footprints: list[WayFootprint] = field(default_factory=list)
    # Ways whose geometry never touched the raster. Not an error -- the query
    # bbox is padded on purpose -- but the count belongs in the provenance.
    n_outside: int = 0
    vectors_fingerprint: str = ""

    def mask(self, name: str) -> np.ndarray:
        m = self.masks.get(name)
        if m is None:
            raise KeyError(f"layer {name!r} was not burned; ask for it in "
                           f"burn(..., layers=(... ,{name!r}))")
        return m

    def has(self, name: str) -> bool:
        return name in self.masks

    def by_kind(self, *kinds: str) -> list[WayFootprint]:
        want = set(kinds)
        return [f for f in self.footprints if f.kind in want]

    def area_m2(self, name: str) -> float:
        return float(self.mask(name).sum()) * self.gsd * self.gsd

    def summary(self) -> str:
        px = self.shape[0] * self.shape[1]
        lines = [f"# burned OSM on a {self.shape[0]}x{self.shape[1]} grid at "
                 f"{self.gsd:.3f} m/px"]
        for name in LAYERS:
            if name in self.masks:
                m = self.masks[name]
                lines.append(f"#   {name:14} {int(m.sum()):>12,} px "
                             f"({100 * m.mean():5.2f}% of the extent, "
                             f"{self.area_m2(name) / 1e4:8.2f} ha)")
        if self.n_outside:
            lines.append(f"#   {self.n_outside} ways fell entirely outside the raster "
                         f"(the query bbox is padded by design)")
        return "\n".join(lines)


def burn(raster, vectors: OsmVectors, layers: tuple[str, ...] = ("road",),
         include_foot_in_road: bool = False) -> BurnedOsm:
    """Rasterise the requested layers and measure ST under every way.

    Only the requested layers get a full-raster mask, because one bool mask over
    the sinai mosaic is 1.2 GB and most callers need one of them.
    """
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"unknown OSM layer(s) {sorted(unknown)}; "
                         f"known: {', '.join(LAYERS)}")
    h, w = raster.shape
    gsd = raster.gsd
    labels = raster.labels
    valid = raster.valid
    out = BurnedOsm(shape=(h, w), gsd=gsd,
                    vectors_fingerprint=vectors.fingerprint())
    for name in layers:
        out.masks[name] = np.zeros((h, w), dtype=bool)

    for idx, way in enumerate(vectors.ways):
        target = _layer_for(way, layers, include_foot_in_road)
        if target is None:
            continue
        rows, cols = to_pixels(raster, way.lon, way.lat)
        if not _touches(rows, cols, h, w):
            out.n_outside += 1
            continue
        mask = out.masks[target]
        as_area = way.is_area and target in ("building", "water", "built_landuse")
        if as_area:
            local, bbox, length_m = _fill_polygon(rows, cols, h, w)
        else:
            hw_m = (FLOW_HALF_WIDTH_M if target == "flow" else
                    BARRIER_HALF_WIDTH_M if target == "barrier" else
                    way.half_width_m)
            local, bbox, length_m = _burn_capsule(rows, cols, hw_m / gsd, h, w, gsd)
        if local is None:
            out.n_outside += 1
            continue
        r0, c0, r1, c1 = bbox
        mask[r0:r1, c0:c1] |= local

        sub_labels = labels[r0:r1, c0:c1]
        sel = local
        if valid is not None:
            sel = local & valid[r0:r1, c0:c1]
        n_px = int(sel.sum())
        hist = np.bincount(sub_labels[sel].ravel(), minlength=N_CLASSES) \
            if n_px else np.zeros(N_CLASSES, dtype=np.int64)
        out.footprints.append(WayFootprint(
            way_id=way.id, idx=idx, kind=way.kind, label=way.label,
            grade=way.grade, surface=way.surface,
            half_width_m=(FLOW_HALF_WIDTH_M if target == "flow" else
                          BARRIER_HALF_WIDTH_M if target == "barrier" else
                          way.half_width_m),
            length_m=length_m, n_px=n_px, hist=hist.astype(np.int64), bbox=bbox,
            rows=rows, cols=cols, is_area=as_area,
            half_px=max((FLOW_HALF_WIDTH_M if target == "flow" else
                         BARRIER_HALF_WIDTH_M if target == "barrier" else
                         way.half_width_m) / gsd, 0.5),
        ))
    return out


def _layer_for(way: OsmWay, layers: tuple[str, ...],
               include_foot_in_road: bool) -> str | None:
    kind = way.kind
    if kind == "road" and "road" in layers:
        return "road"
    if kind == "footway":
        if include_foot_in_road and "road" in layers:
            return "road"
        if "footway" in layers:
            return "footway"
        return None
    if kind == "building" and "building" in layers:
        return "building"
    if kind == "water":
        wt = way.tags.get("waterway")
        if wt in T.FLOW_WATERWAYS and not way.is_area:
            return "flow" if "flow" in layers else None
        return "water" if "water" in layers else None
    if kind == "barrier" and "barrier" in layers:
        return "barrier"
    if kind == "built_landuse" and "built_landuse" in layers:
        return "built_landuse"
    return None


def _touches(rows, cols, h, w, pad: float = 200.0) -> bool:
    return not (rows.max() < -pad or rows.min() > h + pad
                or cols.max() < -pad or cols.min() > w + pad)


def _clip_bbox(rows, cols, h, w, pad_px) -> tuple[int, int, int, int] | None:
    r0 = max(int(math.floor(rows.min() - pad_px)), 0)
    r1 = min(int(math.ceil(rows.max() + pad_px)) + 1, h)
    c0 = max(int(math.floor(cols.min() - pad_px)), 0)
    c1 = min(int(math.ceil(cols.max() + pad_px)) + 1, w)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, c0, r1, c1


def _burn_capsule(rows, cols, half_px: float, h: int, w: int,
                  gsd: float) -> tuple[np.ndarray | None, tuple, float]:
    """Exact buffered polyline inside its own bounding box.

    Per segment: the distance from every local pixel centre to the segment, in
    closed form, thresholded at the half width. Union over segments. Work is
    proportional to the corridor's own area.
    """
    half_px = max(float(half_px), 0.5)
    bbox = _clip_bbox(rows, cols, h, w, half_px + 1.0)
    if bbox is None:
        return None, (0, 0, 0, 0), 0.0
    r0, c0, r1, c1 = bbox
    # A single way can span a whole mosaic; the local box is then the raster and
    # this stays cheap only because segments are handled one at a time.
    local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    length_px = 0.0
    for i in range(len(rows) - 1):
        ar, ac = rows[i] - r0, cols[i] - c0
        br, bc = rows[i + 1] - r0, cols[i + 1] - c0
        seg_len = math.hypot(br - ar, bc - ac)
        length_px += seg_len
        sr0 = max(int(math.floor(min(ar, br) - half_px)), 0)
        sr1 = min(int(math.ceil(max(ar, br) + half_px)) + 1, local.shape[0])
        sc0 = max(int(math.floor(min(ac, bc) - half_px)), 0)
        sc1 = min(int(math.ceil(max(ac, bc) + half_px)) + 1, local.shape[1])
        if sr1 <= sr0 or sc1 <= sc0:
            continue
        yy = np.arange(sr0, sr1, dtype=np.float64)[:, None]
        xx = np.arange(sc0, sc1, dtype=np.float64)[None, :]
        dr, dc = br - ar, bc - ac
        if seg_len < 1e-9:
            d2 = (yy - ar) ** 2 + (xx - ac) ** 2
        else:
            t = ((yy - ar) * dr + (xx - ac) * dc) / (dr * dr + dc * dc)
            np.clip(t, 0.0, 1.0, out=t)
            d2 = (yy - (ar + t * dr)) ** 2 + (xx - (ac + t * dc)) ** 2
        local[sr0:sr1, sc0:sc1] |= d2 <= half_px * half_px
    return local, bbox, length_px * gsd


def _fill_polygon(rows, cols, h: int, w: int) -> tuple[np.ndarray | None, tuple, float]:
    """Even-odd scanline fill inside the polygon's bounding box.

    rasterio would do this too, and does it faster; it is optional here, and a
    building footprint is a convex-ish quad in the overwhelming majority of
    cases, so the scanline is both short and exact.
    """
    bbox = _clip_bbox(rows, cols, h, w, 1.0)
    if bbox is None:
        return None, (0, 0, 0, 0), 0.0
    r0, c0, r1, c1 = bbox
    rr = np.asarray(rows) - r0
    cc = np.asarray(cols) - c0
    if rr[0] != rr[-1] or cc[0] != cc[-1]:
        rr = np.append(rr, rr[0])
        cc = np.append(cc, cc[0])
    nh, nw = r1 - r0, c1 - c0
    local = np.zeros((nh, nw), dtype=bool)
    ay, by = rr[:-1], rr[1:]
    ax, bx = cc[:-1], cc[1:]
    for y in range(nh):
        yc = y + 0.5
        hit = ((ay <= yc) & (by > yc)) | ((by <= yc) & (ay > yc))
        if not hit.any():
            continue
        y0, y1 = ay[hit], by[hit]
        x0, x1 = ax[hit], bx[hit]
        xs = np.sort(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
        for a, b in zip(xs[0::2], xs[1::2]):
            lo = max(int(math.ceil(a - 0.5)), 0)
            hi = min(int(math.floor(b - 0.5)) + 1, nw)
            if hi > lo:
                local[y, lo:hi] = True
    if not local.any():
        # Sub-pixel polygon: keep its outline so it is not silently lost.
        ri = np.clip(np.rint(rr).astype(int), 0, nh - 1)
        ci = np.clip(np.rint(cc).astype(int), 0, nw - 1)
        local[ri, ci] = True
    perim = float(np.hypot(np.diff(rr), np.diff(cc)).sum())
    return local, bbox, perim


# --- distance fields -----------------------------------------------------

def distance_field(mask: np.ndarray, gsd: float,
                   dtype=np.float32) -> np.ndarray:
    """Metres to the nearest True pixel. Transient by design: this is the same
    cost as `index.build_chips`'s anchor fields, and keeping several of them is
    how you turn a 1.2 Gpx mosaic into 20 GB of resident distance rasters."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=dtype)
    return (ndi.distance_transform_edt(~mask) * gsd).astype(dtype)


# --- co-registration -----------------------------------------------------

@dataclass
class AlignReport:
    best_shift_px: tuple[int, int]
    best_shift_m: float
    gain: float                    # overlap at best shift / overlap at zero shift
    overlap_zero: float            # share of sampled ST road px inside the corridor
    overlap_best: float
    n_sampled: int
    st_road_px: int
    corridor_px: int
    verdict: str
    note: str = ""
    corridor_covered: float = 0.0

    def render(self) -> str:
        gain = ("n/a (zero-shift overlap too small to divide by)"
                if self.gain == float("inf")
                else f"+{100 * (self.gain - 1):.1f}%")
        return "\n".join([
            "# OSM/ST co-registration",
            f"# ST road pixels: {self.st_road_px:,}   OSM road corridor: "
            f"{self.corridor_px:,} px "
            f"({self.corridor_px / max(self.st_road_px, 1):.1f}x)",
            f"# sampled {self.n_sampled:,} ST road pixels",
            f"# ST road inside the corridor, zero shift: {self.overlap_zero:.1%}",
            f"# corridor called a road by ST:             "
            f"{self.corridor_covered:.1%}",
            f"# best integer shift: (dr={self.best_shift_px[0]}, "
            f"dc={self.best_shift_px[1]}) = {self.best_shift_m:.2f} m, "
            f"raising the first figure to {self.overlap_best:.1%} ({gain})",
            f"# shift sign convention: ST pixel + (dr, dc) lands in the OSM "
            f"corridor, i.e. ST sits {self.best_shift_m:.2f} m the other way",
            f"# verdict: {self.verdict}",
        ] + ([f"# {self.note}"] if self.note else []))


# A best shift under this many metres means the two maps are registered: the
# residual is the width of a road stripe, not an offset.
ALIGN_OK_M = 2.0
# Above this, every downstream comparison is measuring the offset rather than
# the maps, and the checks should not be run at all.
ALIGN_BAD_M = 5.0
# A shift that buys less than this much extra overlap is not evidence of an
# offset even if it is large -- it is a flat optimum.
ALIGN_MIN_GAIN = 1.10


def align_report(raster, burned: BurnedOsm, seed: int = 7) -> AlignReport:
    """Is ST's idea of where the roads are the same as OSM's, up to a shift?

    Sampling ST road pixels and looking them up in the shifted corridor, rather
    than rolling the whole mask, keeps this affordable on a mosaic.
    """
    corridor = burned.mask("road")
    st_road = np.isin(raster.labels, [cid(n) for n in T.ROAD_CLASSES])
    if raster.valid is not None:
        st_road &= raster.valid
    n_st = int(st_road.sum())
    n_cor = int(corridor.sum())
    if n_st == 0 or n_cor == 0:
        return AlignReport((0, 0), 0.0, 1.0, 0.0, 0.0, 0, n_st, n_cor,
                           "UNCHECKABLE",
                           "one of the two maps has no roads here, so there is "
                           "nothing to register against")
    rr, cc = np.nonzero(st_road)
    if len(rr) > ALIGN_SAMPLE:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(rr), ALIGN_SAMPLE, replace=False)
        rr, cc = rr[keep], cc[keep]
    h, w = corridor.shape
    step = max(int(round(0.5 / raster.gsd)), 1)          # ~0.5 m steps
    reach = max(int(round(ALIGN_SEARCH_M / raster.gsd)), step)
    # Built outwards from zero, not as range(-reach, reach+1, step): with a step
    # that does not divide `reach` the plain range MISSES (0, 0) entirely, and
    # then the zero-shift overlap reads 0.0% and the reported gain is the ratio
    # to a number that was never measured. Found the hard way -- it produced a
    # "+67,056,499,900% more overlap" verdict on the aza AOI.
    offsets = sorted({0} | {s * k for k in range(1, reach // step + 1)
                             for s in (step, -step)}
                     | {reach, -reach})
    best = (-1.0, 0, 0)
    zero = 0.0
    for dr in offsets:
        r = rr + dr
        ok_r = (r >= 0) & (r < h)
        for dc in offsets:
            c = cc + dc
            ok = ok_r & (c >= 0) & (c < w)
            if not ok.any():
                continue
            frac = float(corridor[r[ok], c[ok]].sum()) / len(rr)
            if dr == 0 and dc == 0:
                zero = frac
            if frac > best[0]:
                best = (frac, dr, dc)
    frac_best, dr, dc = best
    shift_m = math.hypot(dr, dc) * raster.gsd
    # A ratio against a near-zero baseline is not a percentage anybody should
    # read, so the gain is capped and the raw pair is reported alongside it.
    gain = frac_best / zero if zero > 0.01 else float("inf")
    # How much of the corridor ST also calls a road. The other direction of the
    # same question, and the one that says whether the corridor width is
    # plausible: OSM buffers are a guess from `tags.HALF_WIDTH_M`, and a
    # corridor three times the area of ST's roads makes the test above lenient.
    corridor_covered = float((st_road & corridor).sum()) / max(n_cor, 1)

    if shift_m <= ALIGN_OK_M or (gain != float("inf") and gain < ALIGN_MIN_GAIN):
        verdict = "REGISTERED"
        note = ("the two maps agree on where the roads are, to within a road "
                "stripe; reference checks are safe to run")
    elif shift_m <= ALIGN_BAD_M:
        verdict = "MARGINAL"
        note = (f"a {shift_m:.1f} m shift raises overlap from "
                f"{zero:.1%} to {frac_best:.1%}. Narrow features will disagree "
                f"for registration reasons; treat per-way findings on tracks "
                f"and alleys as unreliable")
    else:
        verdict = "MISREGISTERED"
        note = (f"a {shift_m:.1f} m shift raises overlap from {zero:.1%} to "
                f"{frac_best:.1%}, and is larger than most of the features being "
                f"compared. Do not run reference checks until the georeference "
                f"is settled -- they would measure the offset, not the maps")
    return AlignReport((dr, dc), shift_m, gain, zero, frac_best, len(rr),
                       n_st, n_cor, verdict, note,
                       corridor_covered=corridor_covered)
