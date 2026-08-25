"""Run the whole pipeline and write an HTML page you can look at.

Point of this module: judge the output with your eyes before anyone tunes a
constant. Every section says what it is, what it assumed, and what is still
synthetic -- a number in here is not a finding.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import digests, loader, synth
from .index import build_chips, build_regions
from .loader import LabelRaster, colourise
from .solutions import s1_audit, s2_adjudicate, s3_triage, s4_products, s5_query, s6_change
from .loader import colormap_hex
from .taxonomy import BY_ID, N_CLASSES

CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#fbfaf8; --mut:#666;
        --line:#ddd; --card:#fff; --warn:#8a5a00; --warnbg:#fff8e6; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e4e0; --bg:#151514; --mut:#9a968f; --line:#333;
          --card:#1e1e1c; --warn:#e0b45a; --warnbg:#2a2312; } }
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.5rem 6rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
       max-width:1180px; margin-inline:auto; }
h1 { font-size:1.7rem; margin:0 0 .25rem; letter-spacing:-.02em; }
h2 { font-size:1.15rem; margin:2.75rem 0 .5rem; padding-top:1rem;
     border-top:1px solid var(--line); letter-spacing:-.01em; }
h3 { font-size:.95rem; margin:1.5rem 0 .4rem; color:var(--mut);
     text-transform:uppercase; letter-spacing:.06em; }
p  { margin:.5rem 0; max-width:75ch; }
.sub { color:var(--mut); margin-bottom:1.5rem; }
.note { color:var(--mut); font-size:.87rem; max-width:80ch; }
.warn { background:var(--warnbg); border-left:3px solid var(--warn);
        padding:.7rem .9rem; margin:1rem 0; font-size:.88rem; color:var(--warn);
        max-width:80ch; border-radius:0 4px 4px 0; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
        gap:1.1rem; margin:1rem 0; }
figure { margin:0; background:var(--card); border:1px solid var(--line);
         border-radius:6px; padding:.6rem; }
figure img { width:100%; height:auto; display:block; border-radius:3px; }
figcaption { font-size:.82rem; color:var(--mut); margin-top:.45rem; }
.scroll { overflow-x:auto; margin:.6rem 0; border:1px solid var(--line);
          border-radius:6px; background:var(--card); }
table { border-collapse:collapse; font:12.5px/1.45 ui-monospace,SFMono-Regular,
        Menlo,monospace; width:100%; }
th,td { text-align:left; padding:.32rem .6rem; white-space:nowrap;
        border-bottom:1px solid var(--line); }
th { position:sticky; top:0; background:var(--card); font-weight:600;
     color:var(--mut); }
tr:last-child td { border-bottom:none; }
td.msg { white-space:normal; min-width:26rem; color:var(--mut); }
.legend { display:flex; flex-wrap:wrap; gap:.3rem .9rem; margin:.6rem 0;
          font-size:12px; font-family:ui-monospace,monospace; }
.legend span { display:flex; align-items:center; gap:.35rem; }
.sw { width:11px; height:11px; border-radius:2px; border:1px solid #0003; }
.kpi { display:flex; flex-wrap:wrap; gap:1.6rem; margin:.8rem 0 1.2rem; }
.kpi div { min-width:8rem; }
.kpi b { display:block; font-size:1.5rem; letter-spacing:-.02em; }
.kpi span { font-size:.78rem; color:var(--mut); text-transform:uppercase;
            letter-spacing:.05em; }
pre { background:var(--card); border:1px solid var(--line); border-radius:6px;
      padding:.7rem .9rem; overflow-x:auto; font-size:12.5px; margin:.6rem 0; }
nav { font-size:.85rem; color:var(--mut); margin:1rem 0 0; }
nav a { color:inherit; margin-right:1rem; }
"""


def _esc(s) -> str:
    return html.escape(str(s))


def tsv_table(text: str, msg_cols=("message", "why", "notes", "top_classes",
                                   "top_nbr_classes")) -> str:
    """TSV (with leading `#` comment lines) -> an HTML table plus the comments."""
    comments, rows = [], []
    for line in text.splitlines():
        if line.startswith("#"):
            comments.append(line.lstrip("# ").strip())
        elif line.strip():
            rows.append(line.split("\t"))
    out = []
    for c in comments:
        out.append(f'<p class="note">{_esc(c)}</p>')
    if rows:
        head, body = rows[0], rows[1:]
        wide = {i for i, h in enumerate(head) if h in msg_cols}
        out.append('<div class="scroll"><table><thead><tr>')
        out += [f"<th>{_esc(h)}</th>" for h in head]
        out.append("</tr></thead><tbody>")
        for r in body:
            out.append("<tr>")
            for i, cell in enumerate(r):
                cls = ' class="msg"' if i in wide else ""
                out.append(f"<td{cls}>{_esc(cell)}</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")
    return "\n".join(out)


def _legend(raster: LabelRaster) -> str:
    hexes = colormap_hex()
    counted = (raster.labels if raster.valid is None
               else raster.labels[raster.valid])
    hist = np.bincount(counted.ravel(), minlength=N_CLASSES)
    total = max(counted.size, 1)
    parts = ['<div class="legend">']
    for c in np.argsort(-hist):
        if hist[c] == 0:
            continue
        parts.append(
            f'<span><i class="sw" style="background:{hexes[c]}"></i>'
            f'{_esc(BY_ID[c].name)} {hist[c] / total:.1%}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


# A 1.2 Gpx mosaic PNG is a 40000 px image nobody can open and nothing reads
# back. The *analysis* stays at full resolution; only this picture is reduced,
# and the caption says by how much.
PREVIEW_MAX_PX = 4_000_000


def _save_preview(raster: LabelRaster, path, *, labels: np.ndarray | None = None) -> int:
    """Write a colourised PNG, decimated if huge. Returns the decimation step."""
    from PIL import Image

    arr = raster.labels if labels is None else labels
    step = 1
    while (arr.shape[0] // step) * (arr.shape[1] // step) > PREVIEW_MAX_PX:
        step += 1
    view = arr[::step, ::step]
    rgb = colourise(view)
    if raster.valid is not None:
        rgb[~raster.valid[::step, ::step]] = 255      # nodata reads as blank
    Image.fromarray(rgb).save(path)
    return step


def build(
    raster: LabelRaster,
    outdir: str | Path,
    *,
    source: str = "synthetic fixture",
    second: LabelRaster | None = None,
    chip_px: int = 128,
    limit: int = 25,
    synthetic: bool = True,
    ridx=None,
    cidx=None,
) -> Path:
    out = Path(outdir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    h, w = raster.shape
    # Use the caller's indices when it has them -- the CLI hands over cached ones
    # so a second report on the same AOI does not rebuild. Same defaults as
    # before when it does not.
    ridx = build_regions(raster) if ridx is None else ridx
    cidx = build_chips(raster, size=chip_px) if cidx is None else cidx
    synthetic_dem = raster.dem is not None and synthetic

    # --- images ------------------------------------------------------------
    step = _save_preview(raster, out / "img" / "labels.png")
    # Each product is float32 at full raster size. Keeping all four alive is
    # 16 bytes/px, which on a mosaic is tens of GB for three arrays that are only
    # ever reduced to a PNG and a mean. Retain the one the summary table needs.
    product_means = {}
    trafficability = None
    for name in ("trafficability", "concealment", "drainage", "fire_fuel"):
        arr = s4_products.compute(raster, name)
        s4_products.to_png(arr[::step, ::step], str(out / "img" / f"{name}.png"))
        # over classified pixels, matching the summary table below -- averaging
        # in the zeroed nodata would report a different number for the same thing
        product_means[name] = float(arr.mean() if raster.valid is None
                                    else arr[raster.valid].mean())
        if name == "trafficability":
            trafficability = arr
        else:
            del arr

    S: list[str] = []

    def sec(title: str, anchor: str) -> None:
        S.append(f'<h2 id="{anchor}">{_esc(title)}</h2>')

    # --- header ------------------------------------------------------------
    area_km2 = raster.n_valid * raster.pixel_area_m2 / 1e6
    S.append(f"<h1>Smart Terrain — segmentation map report</h1>")
    S.append(f'<p class="sub">{_esc(source)} · {h}×{w} px · {raster.gsd:.3f} m/px · '
             f'{area_km2:.3f} km² classified · {len(ridx.regions)} regions</p>')
    S.append('<nav>' + " ".join(
        f'<a href="#{a}">{t}</a>' for a, t in
        [("map", "map"), ("s1", "S1 audit"), ("s2", "S2 adjudicate"),
         ("s3", "S3 attention"), ("s4", "S4 products"), ("s5", "S5 query"),
         ("s6", "S6 change"), ("cost", "digest cost")]) + '</nav>')

    if raster.subset_note:
        S.append(f'<div class="warn">{_esc(raster.subset_note)}</div>')

    if raster.dem is None:
        S.append('<div class="warn">No DEM supplied. Slope and aspect are zero, '
                 'so every morphology check (dip slope, terrace, badlands) and all '
                 'slope terms in S4 are inert. Roughly half the reasoning is '
                 'switched off.</div>')
    elif synthetic_dem:
        S.append('<div class="warn">DEM is synthetic. Slope-derived numbers here '
                 'describe the fixture, not real terrain.</div>')

    if raster.valid is not None:
        S.append(f'<div class="warn">{raster.nodata_frac:.1%} of this extent is '
                 f'nodata. The export\'s nodata value is 0, which is also the id of '
                 f'<code>Unclassified</code> — the two are not distinguishable in the '
                 f'file, so every 0 here is treated as no-data. If the segmenter '
                 f'genuinely emits Unclassified, those pixels are being discarded and '
                 f'the producer needs to give nodata its own value. All fractions '
                 f'below are over the {raster.n_valid:,} classified pixels.</div>')

    # --- map ---------------------------------------------------------------
    sec("The map", "map")
    dec = (f' Decimated {step}× for display ({h // step}×{w // step} px shown); '
           f'all numbers below are computed at full resolution.' if step > 1 else '')
    S.append('<div class="grid"><figure><img src="img/labels.png" alt="label map">'
             '<figcaption>Class labels, hue by superclass. White is nodata. For '
             'human eyes only — recovering class ids from these colours is lossy.'
             f'{_esc(dec)}</figcaption>'
             '</figure></div>')
    S.append(_legend(raster))
    S.append("<h3>Composition</h3>")
    S.append(tsv_table(digests.l0_histogram(raster)))

    # --- S1 ----------------------------------------------------------------
    sec("S1 — consistency audit", "s1")
    rep = s1_audit.run(ridx)
    S.append(f'<div class="kpi">'
             f'<div><b>{len(ridx.regions)}</b><span>regions</span></div>'
             f'<div><b>{len(rep.findings)}</b><span>candidate findings</span></div>'
             f'<div><b>{len(rep.systematic)}</b><span>systematic classes</span></div>'
             f'</div>')
    S.append('<p class="note">Candidates, not confirmed errors. Precision is '
             'unmeasured until someone reviews a sample.</p>')
    if rep.systematic:
        S.append("<h3>Systematic — a class-level problem, not noise</h3>")
        rows = ["class\tn_regions\tn_flagged\tflag_rate\tmean_sev"]
        rows += [f"{n}\t{s['n_regions']}\t{s['n_flagged']}\t{s['flag_rate']:.0%}\t"
                 f"{s['mean_severity']:.2f}" for n, s in rep.systematic]
        S.append(tsv_table("\n".join(rows)))
    S.append("<h3>Review worklist</h3>")
    S.append(tsv_table(s1_audit.worklist(rep, budget=limit)))

    # --- S2 ----------------------------------------------------------------
    sec("S2 — confusion adjudication", "s2")
    rids, seen = [], set()
    for f in rep.findings:
        if f.region_id not in seen:
            seen.add(f.region_id)
            rids.append(f.region_id)
        if len(rids) >= 8:
            break
    adjs = [s2_adjudicate.adjudicate(ridx, r) for r in rids]
    if adjs:
        S.append(f'<p class="note">Adjudicating the {len(adjs)} regions S1 flagged '
                 f'hardest. UNDECIDABLE is a wanted outcome, not a failure.</p>')
        rows = ["rid\tincumbent\tverdict\tbest_alt\trationale"]
        for a in adjs:
            alt = next((n for n, _ in a.ranked if n != a.incumbent), "-")
            rows.append(f"{a.region_id}\t{a.incumbent}\t{a.verdict}\t{alt}\t"
                        f"{a.rationale}")
        S.append(tsv_table("\n".join(rows), msg_cols=("rationale",)))
        S.append("<h3>Evidence for the first one</h3>")
        S.append(tsv_table(s2_adjudicate.render(adjs[0])))
        S.append(tsv_table(s2_adjudicate.ledger(adjs)))
    else:
        S.append('<p class="note">S1 flagged nothing, so there is nothing to '
                 'adjudicate.</p>')

    # --- S3 ----------------------------------------------------------------
    sec("S3 — attention ranking", "s3")
    S.append('<div class="warn">Reframed: this was written as detector-dispatch '
             'triage. With no detector in the pipeline, the budgeted resources are '
             'analyst attention and LLM tokens — same scorer, different consumer. '
             'The dispatch/cost wording below has not been rewritten yet.</div>')
    policy, sel = s3_triage.run(cidx, "vehicles", budget_frac=0.20)
    S.append(tsv_table(s3_triage.render(sel, policy, top_k=limit)))

    # --- S4 ----------------------------------------------------------------
    sec("S4 — derived products", "s4")
    S.append('<p class="note">Where a classification becomes an answer. Bright = '
             'high. Every constant behind these is a guess (docs/solutions/S4).</p>')
    S.append('<div class="grid">')
    caps = {
        "trafficability": "wheeled vehicle, dry. surface × slope, zeroed below the "
                          "vehicle's minimum surface",
        "concealment": "how well a ground object is hidden from above: canopy, "
                       "shadow, adjacency to built, terrain roughness",
        "drainage": "where water pools: flatness + low ground + water-holding class",
        "fire_fuel": "fuel load: canopy plus a dryness bonus",
    }
    for name in product_means:
        S.append(f'<figure><img src="img/{name}.png" alt="{name}">'
                 f'<figcaption><b>{name}</b> — {caps[name]}. '
                 f'mean {product_means[name]:.2f}</figcaption></figure>')
    S.append("</div>")
    S.append("<h3>trafficability by superclass</h3>")
    S.append(tsv_table(s4_products.summarise(raster, trafficability)))

    # --- S5 ----------------------------------------------------------------
    sec("S5 — queries", "s5")
    S.append('<p class="note">Answered in code, not by a model. The LLM\'s job is '
             'to choose the verb and arguments — these verbs are the tool surface '
             'behind <code>segmap ask</code> — and the arithmetic stays here. '
             'Each result lists the region ids it was computed from.</p>')
    for q in ("count Car", "count House", "area Maquis",
              "find House minarea 10", "distance House PavedRoad",
              "corridor PavedRoad vehicle wheeled",
              "corridor PavedRoad vehicle foot"):
        try:
            S.append(f"<h3>{_esc(q)}</h3>")
            S.append(tsv_table(s5_query.query(q, raster, ridx).render(limit=8)))
        except s5_query.QueryError as exc:
            S.append(f'<p class="note">query error: {_esc(exc)}</p>')

    # --- S6 ----------------------------------------------------------------
    sec("S6 — change", "s6")
    t2 = second if second is not None else synth.second_date(raster, seed=17)
    if second is None:
        S.append('<div class="warn">No second date supplied — comparing against a '
                 'synthetically aged copy (season change, succession, a new '
                 'building cluster, a sealed road, and a deliberate lithology '
                 'flip). Numbers here describe the fixture.</div>')
    _save_preview(raster, out / "img" / "labels_t2.png", labels=t2.labels)
    S.append('<div class="grid">'
             '<figure><img src="img/labels.png"><figcaption>date 1</figcaption></figure>'
             '<figure><img src="img/labels_t2.png"><figcaption>date 2</figcaption></figure>'
             '</div>')
    S.append(tsv_table(s6_change.render(s6_change.compare(raster, t2), limit=limit)))

    # --- cost --------------------------------------------------------------
    sec("Digest cost", "cost")
    S.append('<p class="note">What it costs to hand each representation to an LLM. '
             'Estimated at ~3.7 chars/token; use <code>segmap compare '
             '--count-tokens</code> for exact counts.</p>')
    rows = ["level\tchars\ttokens\tdescription"]
    raw = h * w
    rows.append(f"raw\t{raw * 3}\t{raw * 3 // 4}\tlabel raster as text — never do this")
    built = {
        "l0": digests.l0_histogram(raster),
        "l1": digests.l1_grid(raster),
        "l1q": digests.l1q_quadtree(raster),
        "l2": digests.l2_regions(ridx),
        "l3": digests.l3_adjacency(ridx),
        "chips": digests.chip_table(cidx),
    }
    for k, text in built.items():
        rows.append(f"{k}\t{len(text)}\t{digests.estimate_tokens(text)}\t"
                    f"{digests.LEVELS[k]}")
    S.append(tsv_table("\n".join(rows), msg_cols=("description",)))

    page = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Smart Terrain report</title><style>{CSS}</style></head>"
            f"<body>{''.join(S)}</body></html>")
    index = out / "index.html"
    index.write_text(page)
    return index
