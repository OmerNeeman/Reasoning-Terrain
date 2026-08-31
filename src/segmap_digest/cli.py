"""CLI: segmap <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import cache as cache_mod
from . import digests, loader, synth
from .taxonomy import legend as taxonomy_legend


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-i", "--input", help="label raster (.tif/.npy/.png), or a "
                                         "directory of adjacent GeoTIFFs to mosaic. "
                                         "Omit to use a synthetic tile.")
    p.add_argument("--gsd", type=float, default=None, help="metres per pixel")
    p.add_argument("--classes", help="JSON {wire_id: class_name} mapping. Only needed "
                                     "when the GeoTIFF has no ID_TO_LABEL_MAPPING tag.")
    p.add_argument("--size", type=int, default=1024, help="synthetic tile size")
    p.add_argument("--seed", type=int, default=7, help="synthetic tile seed")
    p.add_argument("--dem", help="elevation raster (.tif/.npy) to attach")
    p.add_argument("--max-mpx", type=float, default=None,
                   help="crop to a centred window of at most N megapixels, at full "
                        "resolution, and say so. For poking at an AOI whose full "
                        "index is too expensive to wait for.")
    _add_cache_args(p)
    _add_osm_args(p)


def _add_osm_args(p: argparse.ArgumentParser) -> None:
    """The OSM join. Off unless asked for: it is a second map of the same ground,
    and whether to bring it in is a decision, not a default."""
    p.add_argument("--osm", nargs="?", const="auto", default=None,
                   metavar="PATH",
                   help="join OpenStreetMap. Bare --osm fetches the AOI's bbox "
                        "from Overpass once and caches the snapshot; give a path "
                        "to use a local Overpass-JSON, GeoJSON or .osm file.")
    p.add_argument("--osm-filters", default="all",
                   choices=("roads", "roads+buildings", "all"),
                   help="what to ask Overpass for (default all)")
    p.add_argument("--osm-cut", default="road",
                   help="comma-separated layers that separate blocks "
                        "(default road; e.g. road,water,barrier)")
    p.add_argument("--osm-foot", action="store_true",
                   help="treat footways and paths as block boundaries too")
    p.add_argument("--osm-bbox", default=None, metavar="S,W,N,E",
                   help="query this WGS84 bbox instead of the raster's own "
                        "footprint (for a raster with no georeference)")
    p.add_argument("--osm-refresh", action="store_true",
                   help="refetch from Overpass, replacing the cached snapshot")
    p.add_argument("--osm-offline", action="store_true",
                   help="fail rather than touch the network")
    p.add_argument("--no-osm-align", action="store_true",
                   help="skip the co-registration check (it samples 200k pixels)")


def _add_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache", default=str(cache_mod.DEFAULT_CACHE_DIR),
                   help=f"index cache directory "
                        f"(default {cache_mod.DEFAULT_CACHE_DIR}; $SEGMAP_CACHE)")
    p.add_argument("--no-cache", action="store_true",
                   help="always rebuild, never read or write the cache")
    p.add_argument("--refresh-index", action="store_true",
                   help="rebuild and overwrite the cached index")


def _load(args) -> loader.LabelRaster:
    if not args.input:
        return synth.generate(size=args.size, gsd=args.gsd or 0.3, seed=args.seed)
    try:
        if Path(args.input).is_dir():
            # Stitching twenty tiles is 19 s; once the indices are cached that is
            # the whole cost of a question, so the mosaic is memoised too.
            r = cache_mod.mosaic_raster(
                args.input, gsd=args.gsd, dem=getattr(args, "dem", None),
                classes=getattr(args, "classes", None),
                source=_cache_source(args, mosaic=True), cache_dir=_cache_dir(args),
                refresh=getattr(args, "refresh_index", False),
            )
        else:
            r = loader.load(args.input, gsd=args.gsd, dem=getattr(args, "dem", None),
                            classes=getattr(args, "classes", None))
    except (ValueError, FileNotFoundError) as exc:
        # An AOI whose data has not landed, a missing id mapping, mixed
        # resolutions: all things the operator can act on. A traceback is not an
        # answer to any of them.
        raise SystemExit(f"segmap: cannot read {args.input}: {exc}")

    h, w = r.shape
    print(f"# input: {args.input} -> {h}x{w} px @ {r.gsd:.3f} m/px, "
          f"{h * w / 1e6:.1f} Mpx, {1 - r.nodata_frac:.1%} classified",
          file=sys.stderr)
    if getattr(args, "max_mpx", None):
        r = loader.crop_to_max_mpx(r, args.max_mpx)
        if r.subset_note:
            print(f"# {r.subset_note}", file=sys.stderr)
    return r


# --- index cache ------------------------------------------------------------
#
# Every command that needs an index goes through these two, so a build is paid
# for once per AOI rather than once per question. `--no-cache` and the synthetic
# fixture both route straight to the builders.

def _cache_source(args, mosaic: bool = False) -> dict | None:
    """The input fingerprint for a cache key.

    `mosaic=True` drops the DEM and the crop window: the stitched label arrays
    are the same object whatever `--dem` and `--max-mpx` say, and keying on them
    would store the same 2.4 GB twice.
    """
    if getattr(args, "no_cache", False) or not getattr(args, "input", None):
        return None
    try:
        return cache_mod.source_fingerprint(
            args.input, gsd=args.gsd,
            dem=None if mosaic else getattr(args, "dem", None),
            classes=getattr(args, "classes", None),
            subset=(None if mosaic
                    else loader.crop_spec(getattr(args, "max_mpx", None))),
        )
    except (OSError, ValueError):
        # Un-stattable input is the loader's problem to report, not a reason to
        # fail here. Fall back to building without a cache.
        return None


def _cache_dir(args):
    c = getattr(args, "cache", None)
    return Path(c) if c else cache_mod.DEFAULT_CACHE_DIR


def _regions(raster, args):
    return cache_mod.get(
        "regions", raster, source=_cache_source(args),
        params=cache_mod.region_params(getattr(args, "min_px", 12)),
        cache_dir=_cache_dir(args),
        refresh=getattr(args, "refresh_index", False),
    )


def _chips(raster, args, size: int | None = None):
    return cache_mod.get(
        "chips", raster, source=_cache_source(args),
        params=cache_mod.chip_params(size if size is not None
                                     else getattr(args, "chip", 256)),
        cache_dir=_cache_dir(args),
        refresh=getattr(args, "refresh_index", False),
    )


def _osm(raster, args, layers=("road",)):
    """The OSM layer, or None when `--osm` was not given.

    Every command that can use OSM goes through here, so the fetch-or-cache
    decision, the layer list and the co-registration warning are in one place.
    """
    if not getattr(args, "osm", None):
        return None
    from .osm import layer as osm_layer

    bbox = None
    if getattr(args, "osm_bbox", None):
        try:
            bbox = tuple(float(x) for x in args.osm_bbox.split(","))
            if len(bbox) != 4:
                raise ValueError("expected four numbers")
        except ValueError as exc:
            raise SystemExit(f"segmap: --osm-bbox wants S,W,N,E in degrees: {exc}")
    cut = tuple(x.strip() for x in getattr(args, "osm_cut", "road").split(",")
                if x.strip())
    try:
        return osm_layer.build(
            raster,
            source=None if args.osm == "auto" else args.osm,
            cache_dir=_cache_dir(args), layers=tuple(layers), cut_layers=cut,
            include_foot=getattr(args, "osm_foot", False),
            refresh=getattr(args, "refresh_index", False),
            refresh_osm=getattr(args, "osm_refresh", False),
            allow_network=not getattr(args, "osm_offline", False),
            bbox=bbox, do_align=not getattr(args, "no_osm_align", False),
            cache_source=_cache_source(args),
        )
    except (ValueError, FileNotFoundError, ConnectionError, KeyError) as exc:
        raise SystemExit(f"segmap: OSM layer unavailable: {exc}")


# Levels that exist only when a second map has been joined.
OSM_LEVELS = ("blocks",)


def _policy_names() -> list[str]:
    from .solutions import s3_triage

    return list(s3_triage.BUILTIN)


def _product_names() -> list[str]:
    from .solutions import s4_products

    return sorted(s4_products.PRODUCTS)


def _build_digest(level: str, raster, args) -> str:
    if level == "l0":
        return digests.l0_histogram(raster)
    if level == "l1":
        return digests.l1_grid(raster, n=getattr(args, "grid", 16))
    if level == "l1q":
        return digests.l1q_quadtree(raster, purity=getattr(args, "purity", 0.92))
    if level == "l2":
        return digests.l2_regions(_regions(raster, args),
                                  min_area_m2=getattr(args, "min_area", 0.0),
                                  limit=getattr(args, "limit", None))
    if level == "l3":
        return digests.l3_adjacency(_regions(raster, args),
                                    min_area_m2=getattr(args, "min_area", 0.0))
    if level == "chips":
        return digests.chip_table(_chips(raster, args),
                                  limit=getattr(args, "limit", None))
    if level == "blocks":
        osm = _osm(raster, args)
        if osm is None:
            raise SystemExit("the `blocks` level is the OSM road partition; "
                             "it needs --osm")
        return digests.block_table(osm.blocks, limit=getattr(args, "limit", None))
    raise SystemExit(f"unknown level: {level}")


# --- commands --------------------------------------------------------------

def cmd_synth(args) -> None:
    r = synth.generate(size=args.size, gsd=args.gsd or 0.3, seed=args.seed)
    out = Path(args.out)
    np.save(out.with_suffix(".npy"), r.labels)
    if r.dem is not None:
        np.save(out.with_suffix(".dem.npy"), r.dem)
    from PIL import Image
    Image.fromarray(loader.colourise(r.labels)).save(out.with_suffix(".png"))
    print(f"wrote {out.with_suffix('.npy')} ({r.shape[0]}x{r.shape[1]}), "
          f"{out.with_suffix('.dem.npy')}, {out.with_suffix('.png')}")


def cmd_preview(args) -> None:
    r = _load(args)
    from PIL import Image
    Image.fromarray(loader.colourise(r.labels)).save(args.out)
    print(f"wrote {args.out}")


def cmd_legend(args) -> None:
    if getattr(args, "notes", False):
        from . import class_notes as cn

        ns = cn.load()
        if not ns:
            print("# no analyst notes are filled in yet; showing definitions "
                  "only. `segmap notes` says what is missing.",
                  file=sys.stderr)
        print(cn.legend_with_notes(ns))
        return
    print(taxonomy_legend(compact=not args.full))


def cmd_digest(args) -> None:
    r = _load(args)
    text = _build_digest(args.level, r, args)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} (~{digests.estimate_tokens(text)} tokens est.)",
              file=sys.stderr)
    else:
        print(text)


def cmd_compare(args) -> None:
    r = _load(args)
    print(f"tile {r.shape[0]}x{r.shape[1]} px @ {r.gsd} m/px "
          f"({r.shape[0] * r.shape[1]} cells)\n")
    print(f"{'level':6} {'chars':>9} {'tokens':>9}  description")
    print("-" * 62)
    raw_cells = r.shape[0] * r.shape[1]
    print(f"{'raw':6} {raw_cells * 3:>9} {raw_cells * 3 // 4:>9}  "
          f"label raster as text -- never do this")
    osm = _osm(r, args) if getattr(args, "osm", None) else None
    for level, desc in digests.LEVELS.items():
        if level in OSM_LEVELS and osm is None:
            print(f"{level:6} {'-':>9} {'-':>9}  {desc} -- skipped, needs --osm")
            continue
        text = _build_digest(level, r, args)
        tok = (digests.estimate_tokens(text) if not args.count_tokens
               else __import__("segmap_digest.ask", fromlist=["x"]).count_tokens(text))
        print(f"{level:6} {len(text):>9} {tok:>9}  {desc}")
        if args.dump:
            Path(f"{args.dump}/{level}.txt").write_text(text)
    if args.dump:
        print(f"\ndumped to {args.dump}/")


def cmd_osm(args) -> None:
    """Everything the OSM layer knows about this AOI, and nothing derived from it.

    Deliberately its own command: before any solution is allowed to reason over
    two maps, somebody has to look at whether the two maps are describing the
    same ground. That is what `--preview` and the co-registration block are for.
    """
    r = _load(args)
    args.osm = args.osm or "auto"
    layers = ["road"]
    if args.buildings:
        layers.append("building")
    if args.water:
        layers += ["water", "flow"]
    osm = _osm(r, args, layers=tuple(layers))
    print(osm.summary())

    if args.window or args.tile:
        from .osm import partition as osm_partition

        try:
            if args.tile:
                win = osm_partition.window_of_tile(r, args.tile)
                lab = f"{Path(args.tile).name} at "
            else:
                win = tuple(int(x) for x in args.window.split(","))
                lab = ""
                if len(win) != 4:
                    raise ValueError("expected four integers R0,C0,R1,C1")
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"segmap: cannot locate that window: {exc}")
        print()
        print(osm_partition.render_window(osm.blocks, win, raster=r,
                                          limit=args.limit, label=lab))
        return

    if args.graph:
        print()
        print(osm.graph.render(limit=args.limit))
    if not args.no_blocks:
        print()
        print(digests.block_table(osm.blocks, limit=args.limit))
    if args.out:
        from .osm import preview as osm_preview

        path = osm_preview.write(r, osm, args.out)
        print(f"\n# wrote {path}", file=sys.stderr)


def cmd_playground(args) -> None:
    """Upload a raster, get all four no-LLM solutions over it plus current OSM."""
    from . import playground

    playground.serve(host=args.host, port=args.port, cache_dir=_cache_dir(args),
                     open_browser=not args.no_open)


def cmd_notes(args) -> None:
    """The analyst's class notes: what is filled, what is missing, what is wrong.

    `--coverage` ranks by area on the loaded AOI rather than alphabetically,
    because the point is to spend expert time where the map actually is.
    """
    from . import class_notes as cn

    if args.template:
        print(cn.template(seed_definitions=not args.blank))
        return

    ns = cn.load(args.notes) if args.notes else cn.load()
    if args.check:
        problems = cn.check(ns)
        print("\n".join(f"# {p}" for p in problems)
              if problems else "# class notes: nothing to fix")
        raise SystemExit(1 if problems else 0)

    if args.klass:
        note = ns.get(args.klass)
        if note is None:
            raise SystemExit(f"no notes on file for {args.klass}"
                             if args.klass in cn.NAMES else
                             f"{args.klass} is not one of the 47 class names")
        print(note.render())
        return

    class_area = None
    if args.input:
        r = _load(args)
        ridx = _regions(r, args)
        class_area = {}
        for reg in ridx.regions:
            class_area[reg.class_id] = class_area.get(reg.class_id, 0.0) + reg.area_m2
    print(cn.coverage(ns, class_area, limit=args.limit))


def cmd_audit(args) -> None:
    r = _load(args)
    ridx = _regions(r, args)
    osm = _osm(r, args, layers=("road", "building", "water", "flow"))
    findings = audit_mod.audit(ridx, min_area_m2=args.min_area, osm=osm, raster=r)
    print(f"# {len(ridx.regions)} regions; {audit_mod.summary(findings)}",
          file=sys.stderr)
    print(audit_mod.to_tsv(findings, limit=args.limit))


def cmd_report(args) -> None:
    from . import report as report_mod

    r = _load(args)
    second = (loader.load(args.second, gsd=args.gsd, dem=args.dem,
                          classes=args.classes)
              if args.second else None)
    index = report_mod.build(
        r, args.out,
        source=args.input or f"synthetic fixture (seed {args.seed})",
        second=second, chip_px=args.chip, limit=args.limit,
        synthetic=not args.input,
        ridx=_regions(r, args), cidx=_chips(r, args, size=args.chip),
    )
    print(f"wrote {index}\nopen it with:  xdg-open {index}")


def cmd_index(args) -> None:
    """Build the indices and stop. The point of the whole cache: pay the build
    once, deliberately, instead of by accident inside every question."""
    if args.list:
        rows = cache_mod.entries(args.out)
        if not rows:
            print(f"no cached indices in {args.out}")
            return
        print(f"{'kind':8} {'built':>12} {'build_s':>9} {'size_mb':>9}  source")
        for m in rows:
            src = m.get("source", {})
            files = src.get("files") or []
            where = (Path(files[0][0]).parent if len(files) > 1
                     else (files[0][0] if files else src.get("synthetic", "?")))
            print(f"{m['kind']:8} {cache_mod.describe_age(m):>12} "
                  f"{m.get('build_seconds', 0):>9.0f} {m['bytes'] / 1e6:>9.1f}  "
                  f"{where}{'  [SUBSET]' if src.get('subset') else ''}")
        return

    if not args.input:
        raise SystemExit("segmap index needs -i <tile.tif|directory>; there is "
                         "nothing to cache about the synthetic fixture, which "
                         "builds in under a second")
    args.cache = args.out
    r = _load(args)
    ridx = _regions(r, args)
    cidx = _chips(r, args)
    print(f"indexed {args.input}: {len(ridx.regions)} regions, "
          f"{len(cidx.chips)} chips of {args.chip} px -> {args.out}")


def cmd_solve(args) -> None:
    from .solutions import s1_audit, s2_adjudicate, s3_triage, s4_products, s5_query, s6_change

    r = _load(args)

    if args.solution == "s1":
        ridx = _regions(r, args)
        osm = _osm(r, args, layers=("road", "building", "water", "flow"))
        print(s1_audit.render(
            s1_audit.run(ridx, min_area_m2=args.min_area, osm=osm, raster=r),
            budget=args.budget))

    elif args.solution == "s2":
        ridx = _regions(r, args)
        osm = _osm(r, args, layers=("road", "building", "water", "flow",
                                    "built_landuse"))
        if args.region:
            print(s2_adjudicate.render(
                s2_adjudicate.adjudicate(ridx, args.region, osm=osm, raster=r)))
            return
        # No region given: adjudicate whatever S1 flagged hardest.
        report = s1_audit.run(ridx, min_area_m2=args.min_area, osm=osm, raster=r)
        rids, seen = [], set()
        for f in report.findings:
            if f.region_id not in seen:
                seen.add(f.region_id)
                rids.append(f.region_id)
            if len(rids) >= args.budget:
                break
        adjs = [s2_adjudicate.adjudicate(ridx, rid, osm=osm, raster=r)
                for rid in rids]
        print(f"# adjudicating the top {len(adjs)} regions flagged by S1\n")
        for a in adjs:
            print(s2_adjudicate.render(a), "\n")
        print(s2_adjudicate.ledger(adjs))

    elif args.solution == "s3":
        osm = _osm(r, args, layers=("road", "building", "water"))
        if args.unit == "block":
            if osm is None:
                raise SystemExit("--unit block is the OSM road partition as the "
                                 "unit of spend; it needs --osm")
            policy, sel, missing, labels = s3_triage.run_blocks(
                r, osm, args.policy, budget_frac=args.budget_frac)
            print(s3_triage.render(sel, policy, unit="block", missing=missing,
                                   labels=labels))
        else:
            cidx = _chips(r, args)
            if osm is not None:
                from .osm import chipfeat

                chipfeat.attach(cidx, r, osm)
            if args.query:
                # `--query` is registered on the shared `solve` parser, so s3
                # accepted it and then answered from the BUILT-IN policy -- and
                # printed that policy's own query text, actively confirming a
                # compile that never happened. A flag that is accepted and
                # ignored is worse than one that is rejected.
                policy, sel, diag = s3_triage.run_query(
                    cidx, args.query, budget_frac=args.budget_frac, gsd=r.gsd)
                print(s3_triage.render(sel, policy))
                if diag.get("frac_not_applicable"):
                    print(f"\n# NOTE: {diag['n_not_applicable']} of "
                          f"{diag['n_units']} units "
                          f"({diag['frac_not_applicable']:.0%}) are ground this "
                          f"query does not apply to -- a different answer from "
                          f"'searched and found nothing'.", file=sys.stderr)
                if args.save_policy:
                    policy.to_json(args.save_policy)
                return
            policy, sel = s3_triage.run(cidx, args.policy,
                                        budget_frac=args.budget_frac)
            missing = None
            if osm is not None:
                from .osm import chipfeat

                missing = chipfeat.missing_features(
                    policy, chipfeat.CHIP_FEATURES, has_dist=True,
                    has_interfaces=True)
            print(s3_triage.render(sel, policy, missing=missing))
        if args.save_policy:
            policy.to_json(args.save_policy)
            print(f"\n# policy written to {args.save_policy}", file=sys.stderr)

    elif args.solution == "s4":
        kw = ({"vehicle": args.vehicle, "wet": args.wet}
              if args.product == "trafficability"
              else {"target": args.target} if args.product == "concealment"
              else {})
        # `built_fabric` reads the landuse layer, so it has to be burned or the
        # product's OSM term silently never fires.
        osm = _osm(r, args, layers=("road", "building", "barrier",
                                    "built_landuse"))
        arr = s4_products.compute(r, args.product, osm=osm, **kw)
        ridx = _regions(r, args) if args.regions else None
        title = args.product + (f" ({args.vehicle}{', wet' if args.wet else ''})"
                                if args.product == "trafficability" else "")
        print(f"# S4 product: {title}"
              + ("  [+OSM overlay]" if osm is not None else ""))
        if osm is not None:
            print(s4_products.osm_delta(r, args.product, osm, **kw))
        print(s4_products.summarise(r, arr, ridx))
        if args.out:
            s4_products.to_png(arr, args.out)
            print(f"\n# wrote {args.out}", file=sys.stderr)

    elif args.solution == "s5":
        ridx = _regions(r, args)
        if not args.query:
            raise SystemExit("s5 needs a query, e.g. --query 'corridor PavedRoad'")
        try:
            print(s5_query.query(args.query, r, ridx).render(limit=args.budget))
        except s5_query.QueryError as exc:
            raise SystemExit(f"query error: {exc}")

    elif args.solution == "s6":
        t2 = (loader.load(args.second, gsd=args.gsd, classes=args.classes) if args.second
              else synth.second_date(r, seed=args.seed + 92))
        print(s6_change.render(s6_change.compare(r, t2), limit=args.budget))


def cmd_tools(args) -> None:
    """The tool definitions, offline. No SDK and no API key needed to see the
    surface the model is given."""
    import json

    from . import tools as tools_mod

    print(json.dumps(tools_mod.definitions(), indent=2))


def cmd_ask(args) -> None:
    from . import ask as ask_mod

    r = _load(args)
    # A crop is part of the question, not a detail of how it was run: the model
    # has to know it is looking at a window or it will answer about the AOI.
    question = (f"{r.subset_note}\n\n{args.question}" if r.subset_note
                else args.question)

    if not args.digest:
        # Default: the model plans, the code computes. Nothing in the answer is
        # estimated from a table.
        ridx = _regions(r, args)
        print(f"# {len(ridx.regions)} regions indexed; answering with tool calls",
              file=sys.stderr)
        answer = ask_mod.ask_tools(
            question, r, ridx,
            legend=taxonomy_legend(compact=False),
            model=args.model, effort=args.effort, show_thinking=args.thinking,
        )
        print(answer.text)
        print("\n---")
        print(answer.audit())
        if answer.usage:
            print(f"# {answer.usage}")
        return

    text = _build_digest(args.level, r, args)
    if args.with_audit:
        ridx = _regions(r, args)
        text += "\n\n" + audit_mod.to_tsv(audit_mod.audit(ridx), limit=60)
    print(f"# digest: {args.level}, ~{digests.estimate_tokens(text)} tokens "
          f"(interpretive path -- numbers in the answer are the model's reading "
          f"of a table, not computed)", file=sys.stderr)
    print(ask_mod.ask(
        text, question,
        legend=taxonomy_legend(compact=False),
        model=args.model, effort=args.effort, show_thinking=args.thinking,
    ))


def cmd_showcase(args) -> None:
    """One HTML page explaining every solution, illustrated from this raster."""
    from . import showcase

    r = _load(args)
    print("# building/loading indices", file=sys.stderr)
    ridx = _regions(r, args)
    cidx = _chips(r, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"# rendering {out}", file=sys.stderr)
    showcase.build(r, ridx, cidx, input_path=args.input, out_path=str(out),
                   classes=args.classes)
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB, images inlined -- open it directly)")


def cmd_ui(args) -> None:
    """Serve the demo UI. Everything is loaded once here rather than per request:
    the index build is the expensive part, and a browser that takes 50 s to
    answer its first question reads as broken."""
    from . import webui

    r = _load(args)
    print("# building/loading indices before the server starts", file=sys.stderr)
    ridx = _regions(r, args)
    cidx = _chips(r, args)
    webui.serve(
        r, ridx, cidx,
        input_path=args.input, classes=args.classes, dem=args.dem,
        cache=args.cache, host=args.host, port=args.port,
        legend=taxonomy_legend(compact=False),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="segmap", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write a synthetic label raster")
    _add_input_args(p)
    p.add_argument("-o", "--out", default="examples/tile")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("preview", help="write a colourised PNG of a label raster")
    _add_input_args(p)
    p.add_argument("-o", "--out", default="preview.png")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("legend", help="print the class legend")
    p.add_argument("--full", action="store_true", help="include definitions")
    p.add_argument("--notes", action="store_true",
                   help="fold in the analyst notes (bundled default; "
                        "$SEGMAP_CLASS_NOTES)")
    p.set_defaults(func=cmd_legend)

    p = sub.add_parser("digest", help="emit one representation level")
    _add_input_args(p)
    p.add_argument("level", choices=list(digests.LEVELS))
    p.add_argument("--grid", type=int, default=16, help="l1 grid size")
    p.add_argument("--purity", type=float, default=0.92, help="l1q leaf purity")
    p.add_argument("--chip", type=int, default=256, help="chip size in px")
    p.add_argument("--min-px", type=int, default=12, help="drop regions below N px")
    p.add_argument("--min-area", type=float, default=0.0, help="drop regions below N m2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("compare", help="all levels side by side with token counts")
    _add_input_args(p)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--purity", type=float, default=0.92)
    p.add_argument("--chip", type=int, default=256)
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--min-area", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--count-tokens", action="store_true",
                   help="exact counts from the Anthropic API instead of an estimate")
    p.add_argument("--dump", help="directory to write each level to")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("playground", help="upload a segmentation raster in the "
                                          "browser and run S1-S4 over it with "
                                          "live OSM")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--no-open", action="store_true",
                   help="do not open a browser window")
    _add_cache_args(p)
    p.set_defaults(func=cmd_playground)

    p = sub.add_parser("notes", help="the analyst's class notes: coverage, "
                                     "one class, validation, or a fresh template")
    _add_input_args(p)
    p.add_argument("--notes", help="notes file (default: the bundled "
                                   "class_notes.md; $SEGMAP_CLASS_NOTES)")
    p.add_argument("--class", dest="klass", help="print one class's note")
    p.add_argument("--check", action="store_true",
                   help="validate and exit non-zero if anything is wrong")
    p.add_argument("--template", action="store_true",
                   help="write a fresh template with all 47 classes stubbed")
    p.add_argument("--blank", action="store_true",
                   help="--template without the seeded taxonomy definitions")
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("osm", help="join OpenStreetMap: road corridor, "
                                   "intersections, and the block partition")
    _add_input_args(p)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--graph", action="store_true",
                   help="also print the intersection table")
    p.add_argument("--no-blocks", action="store_true",
                   help="skip the block table")
    p.add_argument("--buildings", action="store_true",
                   help="also burn building footprints")
    p.add_argument("--water", action="store_true",
                   help="also burn water bodies and flow lines")
    p.add_argument("-o", "--out", help="write a PNG of the partition")
    p.add_argument("--window", metavar="R0,C0,R1,C1",
                   help="address one window into the AOI's block partition, "
                        "in pixels of the loaded raster")
    p.add_argument("--tile", metavar="PATH",
                   help="same, for a tile that belongs to the mosaic in -i: "
                        "locates it by its affine transform")
    p.set_defaults(func=cmd_osm)

    p = sub.add_parser("audit", help="run the deterministic consistency audit")
    _add_input_args(p)
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--min-area", type=float, default=25.0)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("index", help="build the region + chip index once and cache "
                                     "it, so every later command loads in seconds")
    _add_input_args(p)
    p.add_argument("-o", "--out", default=str(cache_mod.DEFAULT_CACHE_DIR),
                   help="cache directory")
    p.add_argument("--min-px", type=int, default=12, help="drop regions below N px")
    p.add_argument("--chip", type=int, default=256, help="chip size in px")
    p.add_argument("--list", action="store_true", help="list cached indices and exit")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("report", help="run the whole pipeline and write an HTML page")
    _add_input_args(p)
    p.add_argument("-o", "--out", default="out", help="output directory")
    p.add_argument("--second", help="second-date label raster, for S6")
    p.add_argument("--chip", type=int, default=128)
    p.add_argument("--limit", type=int, default=25, help="rows per table")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("solve", help="run one of the six map-consuming solutions")
    _add_input_args(p)
    p.add_argument("solution", choices=["s1", "s2", "s3", "s4", "s5", "s6"])
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--min-area", type=float, default=25.0)
    p.add_argument("--budget", type=int, default=25,
                   help="s1/s2: findings to review; s5/s6: rows to print")
    p.add_argument("--region", type=int, help="s2: region id to adjudicate")
    p.add_argument("--unit", default="chip", choices=("chip", "block"),
                   help="s3: unit of spend -- the fixed chip grid, or the OSM "
                        "road-bounded block (needs --osm)")
    # Read from the registry rather than repeated here: a policy added to
    # `s3_triage.BUILTIN` and not to this list is invisible from the CLI, which
    # is how `settlement` was unreachable for exactly one commit.
    p.add_argument("--policy", default="vehicles",
                   choices=_policy_names(), help="s3")
    p.add_argument("--budget-frac", type=float, default=0.20,
                   help="s3: fraction of chips to dispatch")
    p.add_argument("--save-policy", help="s3: write the policy JSON here")
    p.add_argument("--chip", type=int, default=256, help="s3: chip size in px")
    p.add_argument("--product", default="trafficability",
                   choices=_product_names(),
                   help="s4")
    p.add_argument("--vehicle", default="wheeled",
                   choices=["wheeled", "tracked", "foot"], help="s4")
    p.add_argument("--wet", action="store_true", help="s4: wet-season trafficability")
    p.add_argument("--target", default="person",
                   choices=("person", "vehicle", "structure"),
                   help="s4: what the concealment score is concealing -- the "
                        "same canopy hides a crouching person and not a truck")
    p.add_argument("--regions", action="store_true", help="s4: include per-region table")
    p.add_argument("--query", help="s5: e.g. 'find House minarea 40' or "
                                   "'corridor PavedRoad'. s3: a detection query "
                                   "compiled into a policy, e.g. 'find missing "
                                   "trees' -- overrides --policy")
    p.add_argument("--second", help="s6: second-date label raster (default: synthetic)")
    p.add_argument("-o", "--out", help="s4: write a PNG of the product")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("tools", help="print the query tool definitions as JSON")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("ask", help="ask Claude a question; answers are computed "
                                   "by the query tools, not estimated")
    _add_input_args(p)
    p.add_argument("question")
    p.add_argument("--digest", action="store_true",
                   help="interpretive path: send a digest instead of the tools, "
                        "for questions no verb can answer")
    p.add_argument("--level", choices=list(digests.LEVELS), default="l2",
                   help="--digest only: which representation level to send")
    p.add_argument("--with-audit", action="store_true",
                   help="--digest only: append audit findings to the digest")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--purity", type=float, default=0.92)
    p.add_argument("--chip", type=int, default=256)
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--min-area", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--thinking", action="store_true", help="show summarised reasoning")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("ui", help="serve a local web UI: the map, the query verbs, "
                                  "`ask`, and all six solutions")
    _add_input_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--chip", type=int, default=256)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("showcase", help="write one HTML page explaining every "
                                        "solution, illustrated from this raster")
    _add_input_args(p)
    p.add_argument("-o", "--out", default="out/showcase.html")
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--chip", type=int, default=256)
    p.set_defaults(func=cmd_showcase)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
