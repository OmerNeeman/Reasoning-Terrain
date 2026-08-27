"""OSM features on the unit of spend -- and the block as an alternative unit.

Two things S3 gets out of the join.

**Features on the chip record.** `osm.dist_road`, `osm.road_frac`,
`osm.building_frac`, `osm.n_junctions`, `osm.block_area` -- usable in exactly
the same policy-rule syntax as `dist_to.PavedRoad` and `entropy`. This keeps the
architectural claim intact: the model still compiles one small cacheable policy,
a deterministic scorer still applies it to millions of chips, and the amortised
reasoning cost per chip is still zero. It just gains a settlement prior that is
a fact rather than an inference from `House` density.

**The block as the unit of spend.** A 256 px grid square is an artefact of a
detector's input size. A block is a piece of ground with a name and edges that
exist. Ranking 98 named blocks is something a human can approve in thirty
seconds, which is the stated bar for a policy that throws away 85% of an AOI --
and reviewing "block 84, bounded by شارع النخيل" is a different experience from
reviewing "chip 1,192 at row 14, col 33".

The honest catch, and it is handled rather than hidden: a block carries a class
histogram but not the chip index's distance fields or class-pair interfaces. A
policy rule referring to `dist_to.PavedRoad` under the block unit would evaluate
false for every block and quietly disable itself. `missing_features` finds those
rules and `s3_triage.render` prints them, because a rule that silently stops
firing is a policy that no longer means what its author approved.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..index import Chip, ChipIndex
from ..taxonomy import N_CLASSES
from .layer import OsmLayer

# Features this module provides on a chip, and on a block. The two sets differ,
# which is the whole reason `missing_features` exists.
CHIP_FEATURES = ("road_frac", "building_frac", "water_frac", "dist_road",
                 "n_junctions", "block_id", "block_area", "n_blocks")
BLOCK_FEATURES = ("road_frac", "building_frac", "water_frac", "n_junctions",
                  "block_id", "block_area", "n_streets", "n_named_streets")

# Distance is capped rather than left infinite: `inf` in a policy comparison is
# a silent always-false, and a chip 4 km from the nearest street is not usefully
# different from one 40 km away.
DIST_CAP_M = 5000.0


def attach(cidx: ChipIndex, raster, layer: OsmLayer) -> ChipIndex:
    """Fill `chip.osm` in place. One distance transform, then per-chip reductions.

    Same shape as `index.build_chips`'s anchor fields, and the same cost profile:
    the EDT is the expensive part, it is computed once, reduced to one number per
    chip, and freed before anything else allocates.
    """
    if not cidx.chips:
        return cidx
    masks = {name: layer.burned.mask(name)
             for name in ("road", "building", "water")
             if layer.burned.has(name)}

    dist = None
    if "road" in masks and masks["road"].any():
        dist = (ndi.distance_transform_edt(~masks["road"]) * raster.gsd)

    junc = np.zeros(raster.shape, dtype=bool)
    if layer.graph is not None:
        h, w = raster.shape
        for i in layer.graph.junctions:
            r, c = int(round(i.row)), int(round(i.col))
            if 0 <= r < h and 0 <= c < w:
                junc[r, c] = True

    blab = layer.blocks.label_array
    for ch in cidx.chips:
        r0, c0, r1, c1 = ch.bbox
        n = max((r1 - r0) * (c1 - c0), 1)
        feats: dict[str, float] = {}
        for name, m in masks.items():
            feats[f"{name}_frac"] = float(m[r0:r1, c0:c1].sum()) / n
        feats["dist_road"] = (min(float(dist[r0:r1, c0:c1].min()), DIST_CAP_M)
                              if dist is not None else DIST_CAP_M)
        feats["n_junctions"] = float(junc[r0:r1, c0:c1].sum())
        sub = np.asarray(blab[r0:r1, c0:c1])
        ids, counts = np.unique(sub[sub > 0], return_counts=True)
        if len(ids):
            top = int(ids[int(np.argmax(counts))])
            feats["block_id"] = float(top)
            feats["block_area"] = float(layer.blocks.get(top).area_m2)
            feats["n_blocks"] = float(len(ids))
        else:
            feats["block_id"] = 0.0
            feats["block_area"] = 0.0
            feats["n_blocks"] = 0.0
        ch.osm = feats
    del dist
    return cidx


class BlockChipIndex(ChipIndex):
    """A `ChipIndex` whose rows are blocks. Same interface, different unit."""


def blocks_as_chips(raster, layer: OsmLayer,
                    min_area_m2: float = 200.0) -> ChipIndex:
    """The block partition, wearing the chip index's interface.

    Everything downstream of `score_chips` works on `class_frac`, `entropy` and
    the rule features, so a block that presents those is a drop-in unit of spend.
    What it cannot present -- `dist_to`, `interfaces` -- is left EMPTY rather
    than zero-filled, so `missing_features` can see the difference between "this
    chip is 0 m from a road" and "this unit does not measure that".
    """
    blocks = layer.blocks.interesting(min_area_m2)
    chips: list[Chip] = []
    masks = {name: layer.burned.mask(name)
             for name in ("road", "building", "water")
             if layer.burned.has(name)}
    junc_by_block: dict[int, int] = {}
    if layer.graph is not None:
        for b in blocks:
            junc_by_block[b.id] = len(b.junction_node_ids)

    for n, b in enumerate(blocks):
        frac = b.frac
        nz = frac[frac > 0]
        ent = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
        r0, c0, r1, c1 = b.bbox
        area_px = max((r1 - r0) * (c1 - c0), 1)
        feats = {"block_id": float(b.id), "block_area": b.area_m2,
                 "n_junctions": float(junc_by_block.get(b.id, 0)),
                 "n_streets": float(len(b.boundary_way_ids)),
                 "n_named_streets": float(len([s for s in b.street_names if s]))}
        for name, m in masks.items():
            # Over the block's bounding box: the block itself is the complement
            # of the corridor, so `road_frac` measured inside it would be zero by
            # construction and would say nothing. The bbox says how
            # street-surrounded the block is, which is the useful quantity.
            feats[f"{name}_frac"] = float(m[r0:r1, c0:c1].sum()) / area_px
        chips.append(Chip(
            id=b.id, row=int(b.centroid[0]), col=int(b.centroid[1]),
            bbox=b.bbox, class_frac=frac, entropy=ent,
            n_classes=b.n_classes, edge_density=0.0,
            dist_to={}, interfaces={}, osm=feats,
        ))
    idx = BlockChipIndex(chips, size=0, stride=0, gsd=raster.gsd)
    return idx


def missing_features(policy, unit_features: tuple[str, ...],
                     has_dist: bool, has_interfaces: bool) -> list[str]:
    """Policy rules that reference something this unit does not measure.

    Returned rather than raised: a policy is a reusable object and running it on
    a coarser unit is legitimate. What is not legitimate is doing so silently.
    """
    out: list[str] = []
    for rule in policy.context_boosts:
        expr = rule.when.split()[0] if rule.when.split() else rule.when
        if expr.startswith("dist_to.") and not has_dist:
            out.append(rule.when)
        elif expr.startswith("interface.") and not has_interfaces:
            out.append(rule.when)
        elif expr.startswith("osm.") and expr.split(".", 1)[1] not in unit_features:
            out.append(rule.when)
    return out
