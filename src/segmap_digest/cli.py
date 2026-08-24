"""CLI: segmap <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import digests, loader, synth
from .index import build_chips, build_regions
from .taxonomy import legend as taxonomy_legend


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-i", "--input", help="label raster (.tif/.npy/.png). "
                                         "Omit to use a synthetic tile.")
    p.add_argument("--gsd", type=float, default=None, help="metres per pixel")
    p.add_argument("--size", type=int, default=1024, help="synthetic tile size")
    p.add_argument("--seed", type=int, default=7, help="synthetic tile seed")
    p.add_argument("--dem", help="elevation raster (.tif/.npy) to attach")


def _load(args) -> loader.LabelRaster:
    if args.input:
        return loader.load(args.input, gsd=args.gsd, dem=getattr(args, "dem", None))
    return synth.generate(size=args.size, gsd=args.gsd or 0.3, seed=args.seed)


def _build_digest(level: str, raster, args) -> str:
    if level == "l0":
        return digests.l0_histogram(raster)
    if level == "l1":
        return digests.l1_grid(raster, n=getattr(args, "grid", 16))
    if level == "l1q":
        return digests.l1q_quadtree(raster, purity=getattr(args, "purity", 0.92))
    if level == "l2":
        ridx = build_regions(raster, min_area_px=getattr(args, "min_px", 12))
        return digests.l2_regions(ridx, min_area_m2=getattr(args, "min_area", 0.0),
                                  limit=getattr(args, "limit", None))
    if level == "l3":
        ridx = build_regions(raster, min_area_px=getattr(args, "min_px", 12))
        return digests.l3_adjacency(ridx, min_area_m2=getattr(args, "min_area", 0.0))
    if level == "chips":
        cidx = build_chips(raster, size=getattr(args, "chip", 256))
        return digests.chip_table(cidx, limit=getattr(args, "limit", None))
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
    for level, desc in digests.LEVELS.items():
        text = _build_digest(level, r, args)
        tok = (digests.estimate_tokens(text) if not args.count_tokens
               else __import__("segmap_digest.ask", fromlist=["x"]).count_tokens(text))
        print(f"{level:6} {len(text):>9} {tok:>9}  {desc}")
        if args.dump:
            Path(f"{args.dump}/{level}.txt").write_text(text)
    if args.dump:
        print(f"\ndumped to {args.dump}/")


def cmd_audit(args) -> None:
    r = _load(args)
    ridx = build_regions(r, min_area_px=args.min_px)
    findings = audit_mod.audit(ridx, min_area_m2=args.min_area)
    print(f"# {len(ridx.regions)} regions; {audit_mod.summary(findings)}",
          file=sys.stderr)
    print(audit_mod.to_tsv(findings, limit=args.limit))


def cmd_report(args) -> None:
    from . import report as report_mod

    r = _load(args)
    second = (loader.load(args.second, gsd=args.gsd, dem=args.dem)
              if args.second else None)
    index = report_mod.build(
        r, args.out,
        source=args.input or f"synthetic fixture (seed {args.seed})",
        second=second, chip_px=args.chip, limit=args.limit,
        synthetic=not args.input,
    )
    print(f"wrote {index}\nopen it with:  xdg-open {index}")


def cmd_solve(args) -> None:
    from .solutions import s1_audit, s2_adjudicate, s3_triage, s4_products, s5_query, s6_change

    r = _load(args)

    if args.solution == "s1":
        ridx = build_regions(r, min_area_px=args.min_px)
        print(s1_audit.render(s1_audit.run(ridx, min_area_m2=args.min_area),
                              budget=args.budget))

    elif args.solution == "s2":
        ridx = build_regions(r, min_area_px=args.min_px)
        if args.region:
            print(s2_adjudicate.render(s2_adjudicate.adjudicate(ridx, args.region)))
            return
        # No region given: adjudicate whatever S1 flagged hardest.
        report = s1_audit.run(ridx, min_area_m2=args.min_area)
        rids, seen = [], set()
        for f in report.findings:
            if f.region_id not in seen:
                seen.add(f.region_id)
                rids.append(f.region_id)
            if len(rids) >= args.budget:
                break
        adjs = [s2_adjudicate.adjudicate(ridx, rid) for rid in rids]
        print(f"# adjudicating the top {len(adjs)} regions flagged by S1\n")
        for a in adjs:
            print(s2_adjudicate.render(a), "\n")
        print(s2_adjudicate.ledger(adjs))

    elif args.solution == "s3":
        cidx = build_chips(r, size=args.chip)
        policy, sel = s3_triage.run(cidx, args.policy, budget_frac=args.budget_frac)
        print(s3_triage.render(sel, policy))
        if args.save_policy:
            policy.to_json(args.save_policy)
            print(f"\n# policy written to {args.save_policy}", file=sys.stderr)

    elif args.solution == "s4":
        kw = {"vehicle": args.vehicle, "wet": args.wet} if args.product == "trafficability" else {}
        arr = s4_products.compute(r, args.product, **kw)
        ridx = build_regions(r, min_area_px=args.min_px) if args.regions else None
        title = args.product + (f" ({args.vehicle}{', wet' if args.wet else ''})"
                                if args.product == "trafficability" else "")
        print(f"# S4 product: {title}")
        print(s4_products.summarise(r, arr, ridx))
        if args.out:
            s4_products.to_png(arr, args.out)
            print(f"\n# wrote {args.out}", file=sys.stderr)

    elif args.solution == "s5":
        ridx = build_regions(r, min_area_px=args.min_px)
        if not args.query:
            raise SystemExit("s5 needs a query, e.g. --query 'corridor PavedRoad'")
        try:
            print(s5_query.query(args.query, r, ridx).render(limit=args.budget))
        except s5_query.QueryError as exc:
            raise SystemExit(f"query error: {exc}")

    elif args.solution == "s6":
        t2 = (loader.load(args.second, gsd=args.gsd) if args.second
              else synth.second_date(r, seed=args.seed + 92))
        print(s6_change.render(s6_change.compare(r, t2), limit=args.budget))


def cmd_ask(args) -> None:
    from . import ask as ask_mod

    r = _load(args)
    text = _build_digest(args.level, r, args)
    if args.with_audit:
        ridx = build_regions(r, min_area_px=args.min_px)
        text += "\n\n" + audit_mod.to_tsv(audit_mod.audit(ridx), limit=60)
    print(f"# digest: {args.level}, ~{digests.estimate_tokens(text)} tokens",
          file=sys.stderr)
    print(ask_mod.ask(
        text, args.question,
        legend=taxonomy_legend(compact=False),
        model=args.model, effort=args.effort, show_thinking=args.thinking,
    ))


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

    p = sub.add_parser("audit", help="run the deterministic consistency audit")
    _add_input_args(p)
    p.add_argument("--min-px", type=int, default=12)
    p.add_argument("--min-area", type=float, default=25.0)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_audit)

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
    p.add_argument("--policy", default="vehicles",
                   choices=["vehicles", "structures", "oov"], help="s3")
    p.add_argument("--budget-frac", type=float, default=0.20,
                   help="s3: fraction of chips to dispatch")
    p.add_argument("--save-policy", help="s3: write the policy JSON here")
    p.add_argument("--chip", type=int, default=256, help="s3: chip size in px")
    p.add_argument("--product", default="trafficability",
                   choices=["trafficability", "concealment", "drainage", "fire_fuel"],
                   help="s4")
    p.add_argument("--vehicle", default="wheeled",
                   choices=["wheeled", "tracked", "foot"], help="s4")
    p.add_argument("--wet", action="store_true", help="s4: wet-season trafficability")
    p.add_argument("--regions", action="store_true", help="s4: include per-region table")
    p.add_argument("--query", help="s5: e.g. 'find House minarea 40' or 'corridor PavedRoad'")
    p.add_argument("--second", help="s6: second-date label raster (default: synthetic)")
    p.add_argument("-o", "--out", help="s4: write a PNG of the product")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("ask", help="send a digest + question to Claude")
    _add_input_args(p)
    p.add_argument("question")
    p.add_argument("--level", choices=list(digests.LEVELS), default="l2")
    p.add_argument("--with-audit", action="store_true",
                   help="append audit findings to the digest")
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

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
