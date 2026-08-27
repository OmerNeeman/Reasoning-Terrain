"""Stage 0.5: the indices -> one summary per screen tile.

A third index, because a map view asks a third question.

  Region index -- semantic polygons. Answers *why*: this thing is mislabelled.
  Chip index   -- the detector's input size. Answers *where to spend*.
  Tile index   -- a fixed screen grid. Answers *what is under the cursor*.

The first two are keyed by an object (a region, a chip); this one is keyed by a
place, and every solution has to be foldable into it at once. That is the whole
difference: a region belongs to exactly one tile but a tile holds findings,
adjudications, block context and four continuous products that were computed by
four passes that never agreed on a common unit. Nothing downstream of this
module re-reads the raster -- the returned dict is the payload a web page gets,
so it holds plain ints, plain floats and nothing numpy.

Cost model: raster statistics are taken tile by tile over an array slice (~190
slices on a 12 Mpx raster, each one vectorised), and everything keyed by a
region -- findings, adjudications -- is placed by its *centroid* in a single
vectorised pass. Scanning the label array once per solution per tile is what
makes this slow, and it buys nothing: a finding is a statement about a region,
and a region's centroid is the only place a reader would look for it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .taxonomy import BY_ID, N_CLASSES

TOP_CLASSES = 4          # what a tooltip can show without becoming a table
TOP_CAUSES = 3
MAX_RIDS = 5             # matches s1_audit.CAUSE_EXAMPLE_RIDS: enough to see a pattern
MAX_EXAMPLES = 2

def _product_keys() -> tuple[str, ...]:
    """The S4 product set, read from the solution rather than restated here.

    A name added to `s4_products.PRODUCTS` and not to a private copy in this
    module is a product that silently never reaches a tile -- which is how
    `drainage` and `fire_fuel` outlived their own deletion in four other files.
    `traversability` is appended because it comes from the provider seam rather
    than from PRODUCTS, but lands in the same per-tile block.
    """
    try:
        from .solutions.s4_products import PRODUCTS

        keys = tuple(sorted(PRODUCTS))
    except ImportError:                               # pragma: no cover
        keys = ("trafficability", "concealment")
    return keys + ("traversability",)


PRODUCT_KEYS = _product_keys()

EMPTY_OSM = {"block": 0, "block_label": "", "road_frac": 0.0,
             "building_frac": 0.0, "n_junctions": 0}


def _f(x) -> float:
    """Plain rounded float. Every number leaving this module goes through here
    or through `int()` -- `json.dumps` raises on np.float32, and a raster
    statistic is np.float32 unless something says otherwise."""
    v = float(x)
    return round(v, 3) if np.isfinite(v) else 0.0


def _group_by_tile(tile_of: np.ndarray, n_tiles: int) -> list[np.ndarray]:
    """items -> per-tile arrays of item indices, by one sort + one searchsorted.

    Cheaper and less fiddly than a dict of lists, and it keeps items in their
    original order inside each tile (stable sort), which is what makes "first
    two examples" reproducible across runs.
    """
    empty = np.empty(0, dtype=np.int64)
    keep = np.nonzero(tile_of >= 0)[0]
    if keep.size == 0:
        return [empty] * n_tiles
    order = keep[np.argsort(tile_of[keep], kind="stable")]
    sorted_tiles = tile_of[order]
    bounds = np.searchsorted(sorted_tiles, np.arange(n_tiles + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_tiles)]


def _class_name(k: int) -> str:
    c = BY_ID.get(int(k))
    return c.name if c is not None else f"class-{int(k)}"


def _switch_target(verdict: str, incumbent: str) -> str:
    return verdict.split(":", 1)[1] if verdict.startswith("SWITCH:") else incumbent


def build_tile_index(raster, ridx, cidx, osm, report, adjudications, products,
                     tile_px: int = 256) -> dict:
    """Fold all four solutions onto a `tile_px` grid over `raster`.

    Everything is taken as already built -- this function computes no regions,
    no chips, no products, and joins nothing that the caller has not joined.
    """
    labels = raster.labels
    h, w = int(labels.shape[0]), int(labels.shape[1])
    valid = raster.valid
    tile_px = max(int(tile_px), 1)
    cols = max(1, -(-w // tile_px))          # ceil; partial edge tiles are kept
    rows = max(1, -(-h // tile_px))
    n_tiles = rows * cols

    boxes = [(r * tile_px, c * tile_px, min((r + 1) * tile_px, h), min((c + 1) * tile_px, w))
             for r in range(rows) for c in range(cols)]

    def tile_of_rc(rr, cc):
        """Row/col arrays -> flat tile ids. Clipped, because a centroid may sit
        exactly on the far edge and a junction may sit a hair outside it."""
        tr = np.clip(np.asarray(rr, dtype=np.float64) // tile_px, 0, rows - 1).astype(np.int64)
        tc = np.clip(np.asarray(cc, dtype=np.float64) // tile_px, 0, cols - 1).astype(np.int64)
        return tr * cols + tc

    # --- region id -> tile, once; every solution keyed by region reuses it ----
    regions = list(getattr(ridx, "regions", []) or [])
    n_reg = len(regions)
    reg_tile = np.full(n_reg + 1, -1, dtype=np.int64)      # index by region id
    if n_reg:
        cent = np.asarray([r.centroid for r in regions], dtype=np.float64)
        reg_tile[1:] = tile_of_rc(cent[:, 0], cent[:, 1])

    def tiles_for_rids(rids: np.ndarray) -> np.ndarray:
        if rids.size == 0:
            return rids.astype(np.int64)
        ok = (rids >= 1) & (rids <= n_reg)
        return np.where(ok, reg_tile[np.where(ok, rids, 0)], -1)

    # --- S1: findings ---------------------------------------------------------
    findings = list(getattr(report, "findings", []) or [])
    f_sev = np.array([float(f.severity) for f in findings], dtype=np.float64)
    f_cause = [(f.cause or f.kind or "") for f in findings]
    f_rid = np.array([int(f.region_id) for f in findings], dtype=np.int64)
    f_groups = _group_by_tile(tiles_for_rids(f_rid), n_tiles)

    # --- S2: adjudications ----------------------------------------------------
    adjs = list(adjudications or [])
    a_rid = np.array([int(a.region_id) for a in adjs], dtype=np.int64)
    a_groups = _group_by_tile(tiles_for_rids(a_rid), n_tiles)

    # --- S3: chips, placed by the centre of their bbox ------------------------
    chips = list(getattr(cidx, "chips", []) or [])
    score = np.zeros(n_tiles, dtype=np.float64)
    chosen = np.zeros(n_tiles, dtype=np.float64)
    if chips:
        bb = np.asarray([c.bbox for c in chips], dtype=np.float64)
        ct = tile_of_rc((bb[:, 0] + bb[:, 2]) * 0.5, (bb[:, 1] + bb[:, 3]) * 0.5)
        vals = np.array([float((c.osm or {}).get("score", 0.0)) for c in chips])
        # `selected` is written onto the chip by the caller after triage runs --
        # a tile counts as selected if ANY chip in it was dispatched, because the
        # question a reader is asking is "would a detector have looked here".
        picked = np.array([float((c.osm or {}).get("selected", 0.0)) for c in chips])
        # Several chips can share a tile (stride < tile_px): keep the max.
        np.maximum.at(score, ct, vals)
        np.maximum.at(chosen, ct, picked)

    # --- OSM rasters, resolved once ------------------------------------------
    burned = getattr(osm, "burned", None) if osm is not None else None
    road = burned.mask("road") if burned is not None and burned.has("road") else None
    bldg = burned.mask("building") if burned is not None and burned.has("building") else None
    bidx = getattr(osm, "blocks", None) if osm is not None else None
    block_arr = getattr(bidx, "label_array", None) if bidx is not None else None

    n_junc = np.zeros(n_tiles, dtype=np.int64)
    graph = getattr(osm, "graph", None) if osm is not None else None
    juncs = [j for j in (getattr(graph, "junctions", None) or []) if getattr(j, "inside", True)]
    if juncs:
        jt = tile_of_rc([j.row for j in juncs], [j.col for j in juncs])
        n_junc = np.bincount(jt, minlength=n_tiles)

    prods = {k: products.get(k) for k in PRODUCT_KEYS} if products else {}
    prods = {k: v for k, v in prods.items()
             if v is not None and getattr(v, "shape", None) == labels.shape}

    tiles: list[dict] = []
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        sub_valid = valid[r0:r1, c0:c1] if valid is not None else None
        npx = (r1 - r0) * (c1 - c0) if sub_valid is None else int(sub_valid.sum())

        classes: list[list] = []
        s4 = {k: 0.0 for k in PRODUCT_KEYS}
        block_id, block_label, road_frac, bldg_frac = 0, "", 0.0, 0.0

        if npx:
            lab = labels[r0:r1, c0:c1]
            counted = lab if sub_valid is None else lab[sub_valid]
            hist = np.bincount(counted.ravel(), minlength=N_CLASSES)
            for k in np.argsort(hist)[::-1][:TOP_CLASSES].tolist():
                if hist[k] > 0:
                    classes.append([_class_name(k), _f(hist[k] / npx)])

            for k, arr in prods.items():
                sub = arr[r0:r1, c0:c1]
                s4[k] = _f(np.nanmean(sub if sub_valid is None else sub[sub_valid]))

            if road is not None:
                m = road[r0:r1, c0:c1]
                road_frac = _f((m if sub_valid is None else m & sub_valid).sum() / npx)
            if bldg is not None:
                m = bldg[r0:r1, c0:c1]
                bldg_frac = _f((m if sub_valid is None else m & sub_valid).sum() / npx)

            if block_arr is not None:
                sb = block_arr[r0:r1, c0:c1]
                bc = np.bincount((sb if sub_valid is None else sb[sub_valid]).ravel())
                if bc.size > 1:
                    bc[0] = 0            # 0 is corridor or nodata, never a block
                    if bc.max() > 0:
                        block_id = int(bc.argmax())
                        try:
                            block_label = str(bidx.get(block_id).label)
                        except (IndexError, AttributeError):
                            block_id, block_label = 0, ""

        # S1 rollup for this tile.
        fi = f_groups[i]
        if fi.size:
            sev = f_sev[fi]
            causes = [[c, int(n)] for c, n in
                      Counter(f_cause[j] for j in fi.tolist()).most_common(TOP_CAUSES)]
            # Worst first, deduped: five *distinct* regions is the useful list.
            rids, seen = [], set()
            for j in fi[np.argsort(-sev, kind="stable")].tolist():
                rid = int(f_rid[j])
                if rid not in seen:
                    seen.add(rid)
                    rids.append(rid)
                if len(rids) == MAX_RIDS:
                    break
            s1 = {"n": int(fi.size), "max_sev": _f(sev.max()),
                  "causes": causes, "rids": rids}
        else:
            s1 = {"n": 0, "max_sev": 0.0, "causes": [], "rids": []}

        # S2 rollup. A SWITCH is the only verdict a reviewer must act on, so it
        # is what gets shown when the tile has more adjudications than room.
        ai = a_groups[i]
        if ai.size:
            picked = [adjs[j] for j in ai.tolist()]
            verdicts = [[str(v), int(n)] for v, n in
                        Counter(a.verdict for a in picked).most_common()]
            picked.sort(key=lambda a: a.verdict == "KEEP")
            examples = [{"rid": int(a.region_id), "from": str(a.incumbent),
                         "to": _switch_target(str(a.verdict), str(a.incumbent)),
                         "why": str(a.rationale)}
                        for a in picked[:MAX_EXAMPLES]]
            s2 = {"verdicts": verdicts, "examples": examples}
        else:
            s2 = {"verdicts": [], "examples": []}

        tiles.append({
            "i": int(i), "r": int(i // cols), "c": int(i % cols),
            "bbox": [int(r0), int(c0), int(r1), int(c1)],
            "classes": classes,
            "s1": s1,
            "s2": s2,
            # No Selection object is passed in, so a tile is never marked
            "s3": {"score": _f(score[i]), "selected": bool(chosen[i] > 0.5)},
            "s4": {k: s4[k] for k in PRODUCT_KEYS},
            "osm": dict(EMPTY_OSM) if osm is None else {
                "block": int(block_id), "block_label": block_label,
                "road_frac": road_frac, "building_frac": bldg_frac,
                "n_junctions": int(n_junc[i]),
            },
        })

    return {"tile_px": int(tile_px), "cols": int(cols), "rows": int(rows),
            "gsd": _f(raster.gsd), "shape": [h, w], "n_tiles": int(n_tiles),
            "tiles": tiles}
