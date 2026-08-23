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


def _load(args) -> loader.LabelRaster:
    if args.input:
        return loader.load(args.input, gsd=args.gsd)
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
