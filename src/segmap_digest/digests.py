"""Stage 1: the representation ladder -- raster -> text an LLM can reason over.

Never send the raw label raster as text: a 2048x2048 tile is ~4M cells. Each
rung below trades fidelity for tokens. Pick the cheapest rung that answers the
question.

    L0   class histogram              ~200 tok    composition, sanity checks
    L1   NxN grid digest              1-3k tok    "where is what"
    L1q  quadtree                     0.5-5k tok  spatial structure, compressed
    L2   region table                 5-50k tok   the workhorse: audit, adjudication
    L3   adjacency graph              +2-5k tok   connectivity, corridors

TSV over JSON throughout -- roughly 2-3x fewer tokens for the same content,
with one legend emitted separately instead of repeating key names per row.
"""

from __future__ import annotations

import numpy as np

from .index import ChipIndex, RegionIndex
from .loader import LabelRaster
from .taxonomy import BY_ID, N_CLASSES, SUPERCLASS_OF


def estimate_tokens(text: str) -> int:
    """Cheap offline estimate (~3.7 chars/token for this kind of dense TSV).

    Use `--count-tokens` on the CLI for the exact count from the Anthropic API.
    """
    return max(1, round(len(text) / 3.7))


# --- L0: histogram ---------------------------------------------------------

def l0_histogram(raster: LabelRaster, min_frac: float = 0.0) -> str:
    labels = raster.labels
    counted = labels if raster.valid is None else labels[raster.valid]
    hist = np.bincount(counted.ravel(), minlength=N_CLASSES)
    total = max(counted.size, 1)
    area_km2 = total * raster.pixel_area_m2 / 1e6

    lines = [
        f"# tile {labels.shape[0]}x{labels.shape[1]} px, gsd {raster.gsd} m, "
        f"classified area {area_km2:.3f} km2",
        "class_id\tname\tsuperclass\tfrac\tarea_m2",
    ]
    if raster.valid is not None:
        nod = labels.size - counted.size
        lines.insert(1, f"# {nod} px ({raster.nodata_frac:.1%} of the extent) are "
                        f"nodata and are excluded from every fraction below")
    for c in np.argsort(-hist):
        frac = hist[c] / total
        if hist[c] == 0 or frac < min_frac:
            continue
        lines.append(
            f"{c}\t{BY_ID[c].name}\t{SUPERCLASS_OF[c]}\t{frac:.4f}\t"
            f"{hist[c] * raster.pixel_area_m2:.0f}"
        )
    return "\n".join(lines)


# --- L1: grid digest -------------------------------------------------------

def l1_grid(raster: LabelRaster, n: int = 16, top_k: int = 3) -> str:
    labels = raster.labels
    h, w = labels.shape
    rs = np.linspace(0, h, n + 1).astype(int)
    cs = np.linspace(0, w, n + 1).astype(int)

    lines = [
        f"# {n}x{n} grid over {h}x{w} px; each cell lists top-{top_k} classes as id:frac",
        "cell\tclasses",
    ]
    for i in range(n):
        for j in range(n):
            sub = labels[rs[i]:rs[i + 1], cs[j]:cs[j + 1]]
            if raster.valid is not None:
                sub = sub[raster.valid[rs[i]:rs[i + 1], cs[j]:cs[j + 1]]]
            if sub.size == 0:
                lines.append(f"{i},{j}\t-")       # all nodata; say so, don't skip
                continue
            hist = np.bincount(sub.ravel(), minlength=N_CLASSES)
            order = np.argsort(-hist)[:top_k]
            parts = [f"{c}:{hist[c] / sub.size:.2f}" for c in order if hist[c] > 0]
            lines.append(f"{i},{j}\t{' '.join(parts)}")
    return "\n".join(lines)


# --- L1q: quadtree ---------------------------------------------------------

# np.bincount promotes its input to intp, i.e. 8 bytes per pixel regardless of
# the array's own dtype. The quadtree's root block on a mosaic is 65536^2, so a
# single naive call there asks for 34 GB. Histogram in row bands instead: exact,
# same answer, bounded peak.
_HIST_CHUNK_PX = 64 << 20


def _block_hist(sub: np.ndarray, n_bins: int) -> np.ndarray:
    if sub.size <= _HIST_CHUNK_PX:
        return np.bincount(sub.ravel(), minlength=n_bins)
    rows = max(1, _HIST_CHUNK_PX // max(sub.shape[1], 1))
    out = np.zeros(n_bins, dtype=np.int64)
    for r in range(0, sub.shape[0], rows):
        out += np.bincount(sub[r:r + rows].ravel(), minlength=n_bins)
    return out


def l1q_quadtree(raster: LabelRaster, purity: float = 0.92, min_block: int = 8) -> str:
    """Morton-ordered quadtree. Homogeneous areas collapse to one token, so a
    terrain map with large uniform regions compresses hard while keeping the
    spatial structure an LLM can actually navigate.

    Grammar: a leaf is a bare class id; a node is `(nw ne sw se)`.

    Nodata is padding: a cropped tile's nodata is indistinguishable from being
    outside the tile, and both collapse to `.`. Both are carried as a uint8
    sentinel rather than a negative int16 -- the square power-of-two pad is the
    single biggest array this module allocates, and on a rectangular mosaic it
    is already several times the raster itself.
    """
    labels = raster.labels
    h, w = labels.shape
    sent = N_CLASSES
    size = 1 << int(np.ceil(np.log2(max(labels.shape))))
    pad = np.full((size, size), sent, dtype=np.uint8)
    pad[:h, :w] = labels
    if raster.valid is not None:
        pad[:h, :w][~raster.valid] = sent

    def rec(r0: int, c0: int, s: int) -> str:
        sub = pad[r0:r0 + s, c0:c0 + s]
        hist = _block_hist(sub, sent + 1)
        n_valid = int(sub.size - hist[sent])
        if n_valid == 0:
            return "."
        dom = int(np.argmax(hist[:sent]))
        if hist[dom] / n_valid >= purity or s <= min_block:
            return str(dom)
        half = s // 2
        kids = [
            rec(r0, c0, half), rec(r0, c0 + half, half),
            rec(r0 + half, c0, half), rec(r0 + half, c0 + half, half),
        ]
        if all(k == kids[0] for k in kids):
            return kids[0]
        return "(" + " ".join(kids) + ")"

    body = rec(0, 0, size)
    header = (
        f"# quadtree over {size}x{size} px (tile {labels.shape[0]}x{labels.shape[1]}, "
        f"gsd {raster.gsd} m); leaf = class id, node = (NW NE SW SE), '.' = outside tile\n"
    )
    return header + body


# --- L2: region table ------------------------------------------------------

L2_COLUMNS = (
    "rid", "class_id", "name", "area_m2", "perim_m", "compact", "elong",
    "cy", "cx", "slope", "slope_sd", "elev", "aspect_cv", "n_nbrs", "top_nbr_classes",
)


def l2_regions(
    ridx: RegionIndex,
    min_area_m2: float = 0.0,
    limit: int | None = None,
    top_nbrs: int = 3,
) -> str:
    regions = [r for r in ridx.regions if r.area_m2 >= min_area_m2]
    regions.sort(key=lambda r: -r.area_m2)
    if limit is not None:
        dropped = max(len(regions) - limit, 0)
        regions = regions[:limit]
    else:
        dropped = 0

    lines = [
        "# region table; one row per connected component. aspect_cv 0=coherent facet, "
        "1=all directions",
        "\t".join(L2_COLUMNS),
    ]
    for r in regions:
        nh = ridx.neighbor_class_hist(r)
        top = sorted(nh.items(), key=lambda kv: -kv[1])[:top_nbrs]
        nbr = " ".join(f"{BY_ID[k].name}:{v:.0f}m" for k, v in top) or "-"
        lines.append("\t".join([
            str(r.id), str(r.class_id), r.class_name,
            f"{r.area_m2:.0f}", f"{r.perimeter_m:.0f}", f"{r.compactness:.2f}",
            f"{r.elongation:.1f}", f"{r.centroid[0]:.0f}", f"{r.centroid[1]:.0f}",
            f"{r.mean_slope:.1f}", f"{r.std_slope:.1f}", f"{r.mean_elev:.0f}",
            f"{r.aspect_circvar:.2f}", str(len(r.neighbors)), nbr,
        ]))
    if dropped:
        # Never let a cap read as "this is everything".
        lines.append(f"# NOTE: {dropped} smaller regions omitted by limit={limit}")
    return "\n".join(lines)


# --- L3: adjacency graph ---------------------------------------------------

def l3_adjacency(ridx: RegionIndex, min_shared_m: float = 1.0,
                 min_area_m2: float = 0.0) -> str:
    keep = {r.id for r in ridx.regions if r.area_m2 >= min_area_m2}
    lines = ["# region adjacency; shared boundary length in metres",
             "rid_a\trid_b\tclass_a\tclass_b\tshared_m"]
    seen: set[tuple[int, int]] = set()
    for r in ridx.regions:
        if r.id not in keep:
            continue
        for nid, shared in r.neighbors.items():
            if nid not in keep or shared < min_shared_m:
                continue
            key = (min(r.id, nid), max(r.id, nid))
            if key in seen:
                continue
            seen.add(key)
            o = ridx.get(nid)
            lines.append(
                f"{key[0]}\t{key[1]}\t{ridx.get(key[0]).class_name}\t"
                f"{ridx.get(key[1]).class_name}\t{shared:.0f}"
            )
    return "\n".join(lines)


# --- chip digest (the detection-triage unit) -------------------------------

def chip_table(cidx: ChipIndex, top_k: int = 4, limit: int | None = None) -> str:
    chips = cidx.chips[:limit] if limit else cidx.chips
    anchors = list(cidx.chips[0].dist_to) if cidx.chips else []
    lines = [
        f"# chip index: {cidx.size}px chips, stride {cidx.stride}, gsd {cidx.gsd} m",
        "\t".join(["chip", "r", "c", "entropy", "n_cls", "edge_den",
                   *[f"d_{a}" for a in anchors], "top_classes"]),
    ]
    for ch in chips:
        order = np.argsort(-ch.class_frac)[:top_k]
        top = " ".join(
            f"{BY_ID[int(k)].name}:{ch.class_frac[k]:.2f}"
            for k in order if ch.class_frac[k] > 0
        )
        d = [f"{ch.dist_to[a]:.0f}" if np.isfinite(ch.dist_to[a]) else "inf"
             for a in anchors]
        lines.append("\t".join([
            str(ch.id), str(ch.row), str(ch.col), f"{ch.entropy:.2f}",
            str(ch.n_classes), f"{ch.edge_density:.3f}", *d, top,
        ]))
    if limit and len(cidx.chips) > limit:
        lines.append(f"# NOTE: {len(cidx.chips) - limit} chips omitted by limit={limit}")
    return "\n".join(lines)


LEVELS = {
    "l0": "class histogram",
    "l1": "NxN grid digest",
    "l1q": "quadtree",
    "l2": "region table",
    "l3": "adjacency graph",
    "chips": "chip index",
}
