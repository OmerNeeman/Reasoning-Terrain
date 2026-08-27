"""Stage 0: raster -> symbolic index.

Two indices, because they answer different questions:

  Region index -- semantic polygons. Tells you *why*. This is what the
    consistency audit and confusion adjudication reason over.
  Chip index   -- fixed grid at the detector's input size. The *unit of spend*.
    This is what a detection-triage policy scores and ranks.

Everything here is query-independent: compute once per tile, reuse for every
detection query and every question you ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .loader import LabelRaster
from .taxonomy import ANCHOR_CLASSES, BY_ID, N_CLASSES, cid


# --- regions ---------------------------------------------------------------

@dataclass
class Region:
    id: int
    class_id: int
    area_px: int
    area_m2: float
    perimeter_m: float
    compactness: float           # 4*pi*A/P^2; 1.0 = circle, ->0 = filament
    elongation: float            # bbox long side / short side
    bbox: tuple[int, int, int, int]      # r0, c0, r1, c1 (exclusive)
    centroid: tuple[float, float]
    mean_slope: float = 0.0
    std_slope: float = 0.0
    mean_elev: float = 0.0
    aspect_circvar: float = 0.0  # 0 = one coherent facet, 1 = all directions
    neighbors: dict[int, float] = field(default_factory=dict)  # region_id -> shared m

    @property
    def class_name(self) -> str:
        return BY_ID[self.class_id].name


@dataclass
class RegionIndex:
    regions: list[Region]
    label_array: np.ndarray      # (H, W) int32, region ids (0 = none)
    # False when no DEM was supplied. Without it every `mean_slope` is 0.0 and
    # every `aspect_circvar` is 0.0 -- which are perfectly valid-looking numbers
    # that happen to mean "unmeasured". Anything reasoning over them has to know
    # the difference, or it reports "slope 0.0 outside required 8-90" as a
    # finding about the terrain rather than about the missing input.
    has_terrain: bool = False

    def by_class(self, class_id: int) -> list[Region]:
        return [r for r in self.regions if r.class_id == class_id]

    def get(self, rid: int) -> Region:
        return self.regions[rid - 1]

    def neighbor_class_hist(self, r: Region) -> dict[int, float]:
        """Shared boundary length per neighbouring *class* -- the context signal."""
        out: dict[int, float] = {}
        for nid, shared in r.neighbors.items():
            k = self.get(nid).class_id
            out[k] = out.get(k, 0.0) + shared
        return out


def build_regions(raster: LabelRaster, min_area_px: int = 12) -> RegionIndex:
    labels = raster.labels
    h, w = labels.shape
    gsd = raster.gsd
    valid = raster.valid

    # Connected components per class, merged into one global id space. Nodata is
    # excluded outright: on a cropped export the nodata value *is* the id of
    # Unclassified, so leaving it in makes one continent-sized bogus region and
    # gives every real region on the crop edge a fictitious neighbour.
    glob = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    conn = np.ones((3, 3), dtype=bool)
    class_of: list[int] = [0]                     # index 0 unused

    present = np.unique(labels if valid is None else labels[valid])
    for c in present:
        mask = labels == c
        if valid is not None:
            mask &= valid
        comp, n = ndi.label(mask, structure=conn)
        del mask
        if n == 0:
            continue
        nz = comp > 0
        glob[nz] = comp[nz] + (next_id - 1)
        class_of.extend([int(c)] * n)
        next_id += n
        del comp, nz

    n_regions = next_id - 1
    if n_regions == 0:
        return RegionIndex([], glob, has_terrain=raster.dem is not None)

    areas = np.bincount(glob.ravel(), minlength=n_regions + 1)[1:]
    perim = _perimeter_counts(glob, n_regions)
    slices = ndi.find_objects(glob)
    # uint8 weights, not `ones_like(glob)`: an int32 copy of the label array is
    # 4 bytes/px of pure waste, which on a mosaic is several GB.
    centroids = ndi.center_of_mass(np.ones(glob.shape, dtype=np.uint8), glob,
                                   range(1, n_regions + 1))

    # Terrain attributes. Half the taxonomy is morphology -- these are what
    # actually separate a dip slope from a terrace, and RGB cannot.
    slope = elev = northing = easting = None
    if raster.dem is not None:
        gy, gx = np.gradient(raster.dem.astype(np.float64), gsd)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = np.arctan2(-gx, gy)
        northing, easting = np.cos(aspect), np.sin(aspect)
        elev = raster.dem

    def zonal(arr):
        if arr is None:
            return None, None
        s = ndi.sum_labels(arr, glob, index=range(1, n_regions + 1))
        m = np.asarray(s) / np.maximum(areas, 1)
        return m, None

    mean_slope, _ = zonal(slope)
    mean_elev, _ = zonal(elev)
    mean_n, _ = zonal(northing)
    mean_e, _ = zonal(easting)
    if slope is not None:
        sq, _ = zonal(slope ** 2)
        std_slope = np.sqrt(np.maximum(sq - mean_slope ** 2, 0.0))
        circvar = 1.0 - np.hypot(mean_n, mean_e)
    else:
        std_slope = circvar = None

    px_area = gsd * gsd
    regions: list[Region] = []
    keep: dict[int, Region] = {}
    for i in range(n_regions):
        rid = i + 1
        a = int(areas[i])
        if a < min_area_px:
            continue
        sl = slices[i]
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        bh, bw = r1 - r0, c1 - c0
        p_m = float(perim[i]) * gsd
        reg = Region(
            id=rid,
            class_id=class_of[rid],
            area_px=a,
            area_m2=a * px_area,
            perimeter_m=p_m,
            compactness=float(4 * np.pi * a * px_area / (p_m ** 2)) if p_m else 0.0,
            elongation=max(bh, bw) / max(min(bh, bw), 1),
            bbox=(r0, c0, r1, c1),
            centroid=(float(centroids[i][0]), float(centroids[i][1])),
            mean_slope=float(mean_slope[i]) if mean_slope is not None else 0.0,
            std_slope=float(std_slope[i]) if std_slope is not None else 0.0,
            mean_elev=float(mean_elev[i]) if mean_elev is not None else 0.0,
            aspect_circvar=float(circvar[i]) if circvar is not None else 0.0,
        )
        regions.append(reg)
        keep[rid] = reg

    _attach_neighbors(glob, keep, gsd)

    # Renumber so region.id indexes the list, keeping RegionIndex.get() O(1).
    # The label array must be renumbered with it -- otherwise zonal statistics
    # keyed by region.id silently read a different region.
    remap = {r.id: i + 1 for i, r in enumerate(regions)}
    lut = np.zeros(n_regions + 1, dtype=np.int32)
    for old, new in remap.items():
        lut[old] = new
    glob = lut[glob]

    for r in regions:
        r.neighbors = {remap[n]: v for n, v in r.neighbors.items() if n in remap}
    for r in regions:
        r.id = remap[r.id]
    return RegionIndex(regions, glob, has_terrain=raster.dem is not None)


def _perimeter_counts(glob: np.ndarray, n: int) -> np.ndarray:
    """Boundary pixel-edge count per region, including the tile border."""
    counts = np.zeros(n + 1, dtype=np.int64)
    padded = np.pad(glob, 1, constant_values=0)
    for da, db in ((1, 0), (0, 1)):
        a = padded[1:-1, 1:-1]
        b = padded[1 + da:padded.shape[0] - 1 + da, 1 + db:padded.shape[1] - 1 + db]
        diff = a != b
        counts += np.bincount(a[diff].ravel(), minlength=n + 1)
        counts += np.bincount(b[diff].ravel(), minlength=n + 1)
    return counts[1:]


def _attach_neighbors(glob: np.ndarray, keep: dict[int, Region], gsd: float) -> None:
    # `glob.max()` is a full-array scan. It used to sit inside the per-pair loop,
    # which made this O(n_pairs * n_pixels) -- unnoticeable on a 1024 px fixture,
    # and effectively a hang on a real 8192 px tile. Hoist it.
    stride = int(glob.max()) + 1
    pairs: dict[tuple[int, int], int] = {}
    for da, db in ((1, 0), (0, 1)):
        a = glob[:glob.shape[0] - da, :glob.shape[1] - db]
        b = glob[da:, db:]
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        ka, kb = a[m].astype(np.int64), b[m].astype(np.int64)
        lo, hi = np.minimum(ka, kb), np.maximum(ka, kb)
        key = lo * stride + hi
        uniq, cnt = np.unique(key, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            i, j = divmod(k, stride)
            pairs[(i, j)] = pairs.get((i, j), 0) + c

    for (i, j), c in pairs.items():
        shared = c * gsd
        if i in keep:
            keep[i].neighbors[j] = keep[i].neighbors.get(j, 0.0) + shared
        if j in keep:
            keep[j].neighbors[i] = keep[j].neighbors.get(i, 0.0) + shared


# --- chips -----------------------------------------------------------------

@dataclass
class Chip:
    id: int
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    class_frac: np.ndarray            # (N_CLASSES,) area fractions over valid px
    entropy: float                    # 0 = uniform chip, high = interface-rich
    n_classes: int
    edge_density: float               # class boundary length / chip area (1/m)
    dist_to: dict[str, float]         # anchor class -> min distance in metres
    interfaces: dict[tuple[int, int], float]   # class pair -> shared length (m)
    # Features from a joined reference map, filled by `osm.chipfeat.attach`.
    # Empty when no OSM layer was joined -- and deliberately NOT cached with the
    # rest of the chip index: these depend on an OSM snapshot that changes on a
    # different clock than the raster, and keying the chip cache on both would
    # throw away a 20-minute chip build every time somebody maps an alley.
    osm: dict[str, float] = field(default_factory=dict)

    @property
    def unit(self) -> str:
        """What one row of the index is. `blocks_as_chips` overrides it."""
        return "chip"

    def frac(self, class_id: int) -> float:
        return float(self.class_frac[class_id])


@dataclass
class ChipIndex:
    chips: list[Chip]
    size: int
    stride: int
    gsd: float

    def as_array(self) -> np.ndarray:
        """(n_chips, N_CLASSES) fraction matrix -- the input to a policy scorer."""
        return np.stack([c.class_frac for c in self.chips]) if self.chips else np.zeros((0, N_CLASSES))


def build_chips(
    raster: LabelRaster,
    size: int = 256,
    stride: int | None = None,
    anchors: tuple[str, ...] = ANCHOR_CLASSES,
) -> ChipIndex:
    labels = raster.labels
    valid = raster.valid
    h, w = labels.shape
    stride = stride or size
    gsd = raster.gsd

    boxes = [(r0, c0, min(r0 + size, h), min(c0 + size, w))
             for r0 in range(0, max(h - size + 1, 1), stride)
             for c0 in range(0, max(w - size + 1, 1), stride)]

    # Distance transforms once per tile, reused by every query. This is the
    # expensive part and it is query-independent by design.
    #
    # One anchor at a time, reduced to per-chip minima before the next is
    # computed. Keeping all of them costs 8 bytes/px/anchor -- 112 GB for seven
    # anchors over a 2 Gpx mosaic -- and the only thing ever read back out is
    # one number per chip.
    dist: dict[str, list[float]] = {}
    for name in anchors:
        mask = labels == cid(name)
        if valid is not None:
            mask &= valid
        if not mask.any():
            dist[name] = [float("inf")] * len(boxes)
            continue
        field = ndi.distance_transform_edt(~mask)
        del mask
        dist[name] = [float(field[r0:r1, c0:c1].min()) * gsd
                      for r0, c0, r1, c1 in boxes]
        del field

    chips: list[Chip] = []
    for n, (r0, c0, r1, c1) in enumerate(boxes):
        sub = labels[r0:r1, c0:c1]
        sub_valid = valid[r0:r1, c0:c1] if valid is not None else None
        npx = sub.size if sub_valid is None else int(sub_valid.sum())
        counted = sub if sub_valid is None else sub[sub_valid]
        hist = np.bincount(counted.ravel(), minlength=N_CLASSES).astype(np.float64)
        frac = hist / npx if npx else hist
        nz = frac[frac > 0]
        ent = float(-(nz * np.log2(nz)).sum())

        ifaces: dict[tuple[int, int], float] = {}
        total_edges = 0
        for da, db in ((1, 0), (0, 1)):
            a = sub[:sub.shape[0] - da, :sub.shape[1] - db]
            b = sub[da:, db:]
            m = a != b
            if sub_valid is not None:
                m &= sub_valid[:sub.shape[0] - da, :sub.shape[1] - db] & sub_valid[da:, db:]
            if not m.any():
                continue
            ka, kb = a[m].astype(np.int32), b[m].astype(np.int32)
            lo, hi = np.minimum(ka, kb), np.maximum(ka, kb)
            key = lo.astype(np.int64) * N_CLASSES + hi
            uk, uc = np.unique(key, return_counts=True)
            total_edges += int(uc.sum())
            for k, c in zip(uk.tolist(), uc.tolist()):
                i, j = divmod(k, N_CLASSES)
                ifaces[(i, j)] = ifaces.get((i, j), 0.0) + c * gsd

        chips.append(Chip(
            id=n, row=r0 // stride, col=c0 // stride, bbox=(r0, c0, r1, c1),
            class_frac=frac, entropy=ent, n_classes=int((hist > 0).sum()),
            edge_density=total_edges * gsd / (npx * gsd * gsd) if npx else 0.0,
            dist_to={k: v[n] for k, v in dist.items()},
            interfaces=ifaces,
        ))

    return ChipIndex(chips, size, stride, gsd)
