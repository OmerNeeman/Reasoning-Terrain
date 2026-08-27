"""A small local web UI: look at the map, ask it questions, read the solutions.

Three panels, and the split between them is the point of the whole repo:

  Verbs       -- `s5_query` runs offline. No API key, no model, no estimate.
                 This is the surface the model is given, driven by hand.
  Ask         -- the model plans, the verbs compute. Needs a key. Every number
                 in the answer is echoed in the tool-call log beneath it, so the
                 answer and its evidence are on screen together.
  Solutions   -- S1..S6, each rendered exactly as its CLI renders it.

The raster and its indices are loaded once at startup and shared by every
request, so a question costs a second rather than the 50 s an index build costs.
Stdlib only: this is a demo server, not a deployment.
"""

from __future__ import annotations

import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from . import digests, loader, taxonomy

# One tile, one index, one lock. The solutions are not reentrant and there is
# exactly one user, so serialising is honest and cheap.
_LOCK = threading.Lock()
_STATE: SimpleNamespace = SimpleNamespace()

SOLUTIONS = [
    {"id": "s1", "title": "Consistency audit",
     "blurb": "Regions whose label contradicts their context, slope or geometry, "
              "rolled up into distinct facts and ranked by what a reviewer's next "
              "minute is best spent on.",
     "controls": [{"name": "budget", "label": "findings", "type": "number", "value": 25}]},
    {"id": "s2", "title": "Confusion adjudication",
     "blurb": "S1 says region 245 is wrong -- then what is it? Scores a candidate "
              "list on three evidence terms and refuses when the evidence cannot "
              "separate them. The refusal is the feature.",
     "controls": [{"name": "region", "label": "region id (blank = top S1 flags)",
                   "type": "number", "value": ""},
                  {"name": "budget", "label": "regions", "type": "number", "value": 5}]},
    {"id": "s3", "title": "Detection triage",
     "blurb": "Use the cheap segmentation as a prior to decide which image chips "
              "are worth an expensive detector. The model compiles the policy "
              "once; a deterministic scorer applies it per chip.",
     "controls": [{"name": "policy", "label": "policy", "type": "select",
                   "options": ["vehicles", "structures", "oov"]},
                  {"name": "budget_frac", "label": "budget fraction", "type": "number",
                   "value": 0.02, "step": 0.01}]},
    {"id": "s4", "title": "Derived products",
     "blurb": "47 classes + slope + season -> the raster someone actually asked "
              "for. Authored with a model, evaluated deterministically: no LLM in "
              "the runtime path.",
     "controls": [{"name": "product", "label": "product", "type": "select",
                   "options": ["trafficability", "concealment", "built_fabric",
                               "change_volatility"]},
                  {"name": "vehicle", "label": "vehicle", "type": "select",
                   "options": ["wheeled", "tracked", "foot"]},
                  {"name": "wet", "label": "wet season", "type": "checkbox"}]},
    {"id": "s5", "title": "Query verbs",
     "blurb": "The quantitative half of natural-language querying. Same engine "
              "the Verbs tab drives and the same one the model calls.",
     "controls": [{"name": "query", "label": "query", "type": "text",
                   "value": "describe"}]},
    {"id": "s6", "title": "Change reasoning",
     "blurb": "Two dates, classified transitions -- phenology vs succession vs "
              "real change -- instead of a pixel diff that is mostly noise. "
              "Needs a second raster of the same footprint.",
     "controls": [{"name": "second", "label": "second-date raster path", "type": "text",
                   "value": ""}]},
]

EXAMPLES = [
    "describe",
    "count House minarea 40",
    "area DirtRoad",
    "area PavedRoad",
    "count Car",
    "distance MaralBadlands DirtRoad",
    "corridor DirtRoad vehicle wheeled",
    "find House minarea 200",
    "find Car near PavedRoad 20",
]


def _args(**kw) -> SimpleNamespace:
    """The CLI's argparse defaults, as a plain object the helpers accept."""
    base = dict(input=_STATE.input, gsd=None, classes=_STATE.classes, dem=_STATE.dem,
                size=512, seed=0, max_mpx=None, cache=_STATE.cache, no_cache=False,
                refresh_index=False, min_px=12, min_area=0.0, budget=25, region=None,
                policy="vehicles", budget_frac=0.02, save_policy=None, chip=256,
                product="trafficability", vehicle="wheeled", wet=False, regions=False,
                query=None, second=None, out=None, grid=16, purity=0.92, limit=None)
    base.update(kw)
    return SimpleNamespace(**base)


# --- the three panels ------------------------------------------------------

def run_query(q: str) -> dict:
    """A verb, straight into `s5_query`. No model involved."""
    from .solutions import s5_query

    try:
        res = s5_query.query(q, _STATE.raster, _STATE.ridx)
    except s5_query.QueryError as exc:
        # A refusal is an answer here, not a crash: 'Truck' is outside the
        # vocabulary and saying so is the correct output.
        return {"status": "unanswerable", "text": f"query error: {exc}"}
    return {"status": "ok", "text": res.render(limit=200),
            "ids": list(res.ids)[:200], "id_kind": res.id_kind}


def run_ask(question: str, model: str, effort: str) -> dict:
    """The model plans, the verbs compute. The tool log is returned separately
    from the prose so the evidence can be read next to the claim."""
    from . import ask as ask_mod

    try:
        ans = ask_mod.ask_tools(
            question, _STATE.raster, _STATE.ridx,
            legend=taxonomy.legend(compact=False),
            model=model, effort=effort,
        )
    except SystemExit as exc:            # client() raises this with the fix in it
        return {"status": "no-key", "text": str(exc), "calls": []}
    except Exception as exc:             # auth, rate limit, overload
        name = type(exc).__name__
        hint = ""
        if "Authentication" in name:
            hint = ("\n\nThe key in the environment was rejected by the API. "
                    "The Verbs and Solutions tabs do not need one and still work.")
        return {"status": "error", "text": f"{name}: {exc}{hint}", "calls": []}

    return {"status": "ok", "text": ans.text, "usage": ans.usage,
            "calls": [{"status": c["status"],
                       "query": c.get("query", json.dumps(c.get("arguments", {}))),
                       "summary": c["summary"]} for c in ans.calls]}


def run_solution(sid: str, opts: dict) -> dict:
    """Each solution, rendered exactly as the CLI renders it."""
    from .solutions import (s1_audit, s2_adjudicate, s3_triage, s4_products,
                            s5_query, s6_change)

    r, ridx = _STATE.raster, _STATE.ridx

    def num(k, default=None, cast=int):
        v = opts.get(k)
        if v in (None, "", "null"):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    if sid == "s1":
        rep = s1_audit.run(ridx, min_area_m2=25.0)
        return {"text": s1_audit.render(rep, budget=num("budget", 25))}

    if sid == "s2":
        rid = num("region")
        if rid:
            return {"text": s2_adjudicate.render(s2_adjudicate.adjudicate(ridx, rid))}
        rep = s1_audit.run(ridx, min_area_m2=25.0)
        rids, seen = [], set()
        for f in rep.findings:
            if f.region_id not in seen:
                seen.add(f.region_id)
                rids.append(f.region_id)
            if len(rids) >= num("budget", 5):
                break
        adjs = [s2_adjudicate.adjudicate(ridx, i) for i in rids]
        out = [f"# adjudicating the top {len(adjs)} regions flagged by S1\n"]
        out += [s2_adjudicate.render(a) + "\n" for a in adjs]
        out.append(s2_adjudicate.ledger(adjs))
        return {"text": "\n".join(out)}

    if sid == "s3":
        policy, sel = s3_triage.run(_STATE.cidx, opts.get("policy", "vehicles"),
                                    budget_frac=num("budget_frac", 0.02, float))
        return {"text": s3_triage.render(sel, policy)}

    if sid == "s4":
        product = opts.get("product", "trafficability")
        kw = ({"vehicle": opts.get("vehicle", "wheeled"),
               "wet": bool(opts.get("wet"))} if product == "trafficability" else {})
        arr = s4_products.compute(r, product, **kw)
        title = product + (f" ({kw['vehicle']}{', wet' if kw['wet'] else ''})"
                           if kw else "")
        return {"text": f"# S4 product: {title}\n"
                        + s4_products.summarise(r, arr, ridx)}

    if sid == "s5":
        q = (opts.get("query") or "describe").strip()
        try:
            return {"text": s5_query.query(q, r, ridx).render(limit=200)}
        except s5_query.QueryError as exc:
            return {"text": f"query error: {exc}"}

    if sid == "s6":
        path = (opts.get("second") or "").strip()
        if not path:
            return {"text": "S6 needs a second-date raster of the same footprint. "
                            "Give a path to one -- there is only one date in "
                            "data/incoming for this tile, so this panel has "
                            "nothing to diff."}
        try:
            t2 = loader.load(path, gsd=None, classes=_STATE.classes)
        except (ValueError, FileNotFoundError) as exc:
            return {"text": f"could not load {path}: {exc}"}
        return {"text": s6_change.render(s6_change.compare(r, t2))}

    return {"text": f"unknown solution: {sid}"}


# --- server ----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):      # one line per request, not three
        print(f"  {self.command} {self.path} -> {a[1] if len(a) > 1 else ''}")

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/preview.png"):
            return self._send(_STATE.preview, "image/png")
        if self.path.startswith("/api/meta"):
            return self._json(_STATE.meta)
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"text": "bad request body"}, 400)
        try:
            with _LOCK:
                if self.path.startswith("/api/query"):
                    return self._json(run_query(body.get("query", "describe")))
                if self.path.startswith("/api/ask"):
                    return self._json(run_ask(body.get("question", ""),
                                              body.get("model", "claude-opus-5"),
                                              body.get("effort", "high")))
                if self.path.startswith("/api/solve"):
                    return self._json(run_solution(body.get("id", "s1"),
                                                   body.get("opts", {})))
        except Exception:
            return self._json({"status": "error",
                               "text": traceback.format_exc()}, 500)
        self._json({"text": "not found"}, 404)


def _build_state(raster, ridx, cidx, *, input_path, classes, dem, cache):
    from PIL import Image

    from .report import PREVIEW_MAX_PX

    step = 1
    while ((raster.labels.shape[0] // step) * (raster.labels.shape[1] // step)
           > PREVIEW_MAX_PX):
        step += 1
    rgb = loader.colourise(raster.labels[::step, ::step])
    if raster.valid is not None:
        rgb[~raster.valid[::step, ::step]] = 255
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")

    cmap = loader.colormap()
    hist = digests.l0_histogram(raster)
    classes_present = []
    for line in hist.splitlines():
        if line.startswith(("#", "class_id")):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        cid, name, superclass, frac, area = parts[0], parts[1], parts[2], parts[3], parts[4]
        r_, g_, b_ = (int(v) for v in cmap[int(cid)])
        classes_present.append({"id": int(cid), "name": name, "superclass": superclass,
                                "frac": float(frac), "area_m2": float(area),
                                "rgb": f"rgb({r_},{g_},{b_})"})

    h, w = raster.labels.shape
    _STATE.raster, _STATE.ridx, _STATE.cidx = raster, ridx, cidx
    _STATE.input, _STATE.classes, _STATE.dem, _STATE.cache = input_path, classes, dem, cache
    _STATE.preview = buf.getvalue()
    _STATE.legend = ""
    _STATE.meta = {
        "tile": input_path or "(synthetic)",
        "shape": f"{h}x{w} px",
        "mpx": round(h * w / 1e6, 1),
        "gsd": round(raster.gsd, 3),
        "extent_m": f"{round(w * raster.gsd)} x {round(h * raster.gsd)} m",
        "regions": len(ridx.regions),
        "chips": len(cidx.chips),
        "preview_step": step,
        "has_dem": raster.dem is not None,
        "classes": classes_present,
        "examples": EXAMPLES,
        "solutions": SOLUTIONS,
    }


def serve(raster, ridx, cidx, *, input_path, classes, dem, cache,
          host="127.0.0.1", port=8765, legend=""):
    _build_state(raster, ridx, cidx, input_path=input_path, classes=classes,
                 dem=dem, cache=cache)
    _STATE.legend = legend
    srv = ThreadingHTTPServer((host, port), Handler)
    m = _STATE.meta
    print(f"\n  reasoning-terrain UI  http://{host}:{port}")
    print(f"  tile     {m['tile']}")
    print(f"  {m['shape']} @ {m['gsd']} m/px, {m['extent_m']}, "
          f"{m['regions']} regions, {m['chips']} chips")
    print(f"  DEM      {'attached' if m['has_dem'] else 'none -- slope questions abstain'}")
    print("  Ctrl-C to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        srv.server_close()


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>reasoning-terrain</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
:root{
  --bg:#f7f8f9; --panel:#ffffff; --ink:#14181d; --dim:#5d6772; --line:#dde2e7;
  --accent:#2f5d7c; --soft:#eaeff3; --ok:#2c7a52; --warn:#8a6320; --bad:#a33b2a;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --serif:"IBM Plex Serif",ui-serif,Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1418; --panel:#161d23; --ink:#e6ecf2; --dim:#8b97a3; --line:#26303a;
  --accent:#7fb3d5; --soft:#1b242c; --ok:#6fc79a; --warn:#d9ae62; --bad:#e89078;
}}
:root[data-theme="dark"]{
  --bg:#0f1418; --panel:#161d23; --ink:#e6ecf2; --dim:#8b97a3; --line:#26303a;
  --accent:#7fb3d5; --soft:#1b242c; --ok:#6fc79a; --warn:#d9ae62; --bad:#e89078;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 var(--sans)}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
  gap:18px;align-items:baseline;flex-wrap:wrap;background:var(--panel)}
header h1{font-size:15px;margin:0;letter-spacing:.02em}
header .meta{font:12px/1.4 var(--mono);color:var(--dim)}
.badge{font:11px var(--mono);padding:2px 7px;border-radius:99px;
  background:var(--accent-soft);color:var(--accent);border:1px solid var(--line)}
.badge.warn{color:var(--warn)}
main{display:grid;grid-template-columns:minmax(280px,340px) 1fr;gap:0;
  align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
aside{border-right:1px solid var(--line);padding:16px;position:sticky;top:0;
  max-height:100vh;overflow:auto}
@media(max-width:900px){aside{position:static;max-height:none;border-right:0;
  border-bottom:1px solid var(--line)}}
aside img{width:100%;border:1px solid var(--line);border-radius:4px;display:block;
  image-rendering:pixelated;background:#fff}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
  margin:20px 0 8px;font-weight:600}
.legend{font:12px var(--mono)}
.legend div{display:flex;gap:7px;align-items:center;padding:2px 4px;border-radius:3px;
  cursor:pointer}
.legend div:hover{background:var(--accent-soft)}
.sw{width:11px;height:11px;border-radius:2px;border:1px solid rgba(128,128,128,.45);
  flex:0 0 auto}
.legend .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.legend .pc{color:var(--dim);font-variant-numeric:tabular-nums}
section{padding:16px 20px;max-width:1100px}
nav{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:16px}
nav button{background:none;border:0;border-bottom:2px solid transparent;padding:8px 14px;
  font:inherit;font-size:14px;color:var(--dim);cursor:pointer}
nav button.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.tab{display:none}.tab.on{display:block}
.note{font-size:13px;color:var(--dim);margin:0 0 14px;max-width:70ch}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
input[type=text],textarea,select,input[type=number]{font:13px var(--mono);
  background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:4px;padding:7px 9px}
input[type=text],textarea{width:100%}
textarea{min-height:64px;resize:vertical;font-family:inherit;font-size:14px}
button.go{background:var(--accent);color:#fff;border:0;border-radius:4px;
  padding:8px 16px;font:inherit;font-size:14px;font-weight:600;cursor:pointer}
button.go:disabled{opacity:.5;cursor:default}
.chip{font:12px var(--mono);background:var(--panel);border:1px solid var(--line);
  border-radius:99px;padding:4px 11px;cursor:pointer;color:var(--dim)}
.chip:hover{border-color:var(--accent);color:var(--accent)}
pre{font:12.5px/1.5 var(--mono);background:var(--panel);border:1px solid var(--line);
  border-radius:5px;padding:12px;overflow-x:auto;white-space:pre;margin:0}
.answer{font-size:14.5px;line-height:1.6;background:var(--panel);
  border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:5px;padding:14px 16px;white-space:pre-wrap;margin:0 0 14px}
.calls{font:12px var(--mono)}
.calls table{border-collapse:collapse;width:100%}
.calls td{border-top:1px solid var(--line);padding:5px 8px;vertical-align:top}
.calls .st{white-space:nowrap;font-weight:600}
.st.ok{color:var(--ok)}.st.empty{color:var(--warn)}
.st.unanswerable{color:var(--bad)}.st.error{color:var(--bad)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px}
.card h3{margin:0 0 4px;font-size:14px}
.card h3 code{font:12px var(--mono);color:var(--accent);margin-right:6px}
.card p{margin:0 0 10px;font-size:12.5px;color:var(--dim);line-height:1.5}
.card label{font:11px var(--mono);color:var(--dim);display:block;margin-bottom:8px}
.card label input,.card label select{width:100%;margin-top:3px}
.card label.cb{display:flex;gap:6px;align-items:center}
.card label.cb input{width:auto;margin:0}
.spin{color:var(--dim);font:13px var(--mono)}
nav .off{font:10px var(--mono);text-transform:uppercase;letter-spacing:.06em;
  color:var(--warn);border:1px solid var(--line);border-radius:99px;
  padding:1px 6px;margin-left:6px;vertical-align:1px}
.tag{font:10px var(--mono);text-transform:uppercase;letter-spacing:.06em;
  color:var(--ok);border:1px solid var(--line);background:var(--panel);
  border-radius:99px;padding:2px 7px;margin-right:7px;white-space:nowrap}
</style>

<header>
  <h1>reasoning&#8209;terrain</h1>
  <span class="meta" id="meta"></span>
  <span id="badges"></span>
</header>

<main>
<aside>
  <img id="prev" src="/preview.png" alt="colourised label raster">
  <div class="meta" id="prevnote" style="font:11px var(--mono);color:var(--dim);margin-top:6px"></div>
  <h2>Classes present</h2>
  <div class="legend" id="legend"></div>
</aside>

<section>
  <nav>
    <button class="on" data-t="verbs">Verbs</button>
    <button data-t="sol">Solutions</button>
    <button data-t="ask">Ask <span class="off">needs a key</span></button>
  </nav>

  <div class="tab on" id="t-verbs">
    <p class="note"><span class="tag">no key needed</span> The query verbs, run
      straight against the index. No model, nothing estimated &mdash; this is the
      surface a model would be given, driven by hand. Every answer carries the
      region ids it was computed from, so any number here can be checked against
      the map.</p>
    <div class="row"><input type="text" id="q" value="describe"
      placeholder="describe &middot; count House minarea 40 &middot; distance MaralBadlands DirtRoad">
    </div>
    <div class="row" id="ex"></div>
    <div class="row"><button class="go" id="qgo">Run</button></div>
    <pre id="qout">Pick an example or type a verb, then Run.</pre>
  </div>

  <div class="tab" id="t-ask">
    <p class="note"><b>This is the only path in the repo that calls a model, and
      it is the only one that needs a key.</b> The model plans; the verbs
      compute. It cannot state a number the code did not produce &mdash; the
      tool&#8209;call log renders below the answer as the audit trail, and every
      call is reproducible with <code>segmap solve s5 --query</code>.
      Everything in Verbs and Solutions runs without it.</p>
    <textarea id="qq" placeholder="How many buildings larger than 40 square meters are there?"></textarea>
    <div class="row" style="margin-top:10px">
      <select id="model">
        <option value="claude-opus-5">claude-opus-5</option>
        <option value="claude-sonnet-5">claude-sonnet-5</option>
        <option value="claude-haiku-4-5-20251001">claude-haiku-4.5</option>
      </select>
      <select id="effort">
        <option value="low">effort: low</option>
        <option value="medium">effort: medium</option>
        <option value="high" selected>effort: high</option>
      </select>
      <button class="go" id="ago">Ask</button>
    </div>
    <div class="row" id="exq"></div>
    <div id="aout"></div>
  </div>

  <div class="tab" id="t-sol">
    <p class="note"><span class="tag">no key needed</span> The six
      map&#8209;consuming solutions, each rendered exactly as its CLI renders it.
      None of them calls a model at runtime: S1 feeds the segmenter, S2 refuses
      when the evidence cannot separate candidates, S3 carries a cost line, S4 is
      a checked-in lookup table, S5 is the Verbs tab, S6 needs a second date.</p>
    <div class="cards" id="cards"></div>
    <h2 id="solh" style="display:none">Output</h2>
    <pre id="sout" style="display:none"></pre>
  </div>
</section>
</main>

<script>
const $ = s => document.querySelector(s);
const el = (t,c,x) => { const n=document.createElement(t); if(c)n.className=c;
  if(x!==undefined)n.textContent=x; return n; };
let META = null;

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); $('#t-'+b.dataset.t).classList.add('on');
});

async function post(url, body){
  const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  return r.json();
}

fetch('/api/meta').then(r=>r.json()).then(m => {
  META = m;
  $('#meta').textContent = `${m.tile}  ·  ${m.shape} @ ${m.gsd} m/px  ·  `
    + `${m.extent_m}  ·  ${m.regions.toLocaleString()} regions  ·  ${m.chips} chips`;
  const b = $('#badges');
  b.append(el('span','badge', m.classes.length + ' classes'));
  b.append(el('span','badge'+(m.has_dem?'':' warn'),
    m.has_dem ? 'DEM attached' : 'no DEM — slope questions abstain'));
  $('#prevnote').textContent = m.preview_step > 1
    ? `preview decimated ${m.preview_step}× for display; every number is computed at full resolution`
    : 'preview at full resolution';

  const L = $('#legend');
  m.classes.forEach(c => {
    const d = el('div'); d.title = `${c.superclass} · ${Math.round(c.area_m2).toLocaleString()} m²`;
    const sw = el('span','sw'); sw.style.background = c.rgb;
    d.append(sw, el('span','nm',c.name), el('span','pc',(100*c.frac).toFixed(2)+'%'));
    d.onclick = () => {
      const t = document.querySelector('nav button.on').dataset.t;
      if(t==='ask'){ $('#qq').value += (($('#qq').value?' ':'')+c.name); $('#qq').focus(); }
      else { $('#q').value = 'area ' + c.name; $('#q').focus(); }
    };
    L.append(d);
  });

  m.examples.forEach(x => {
    const c = el('button','chip',x);
    c.onclick = () => { $('#q').value = x; runQuery(); };
    $('#ex').append(c);
  });
  [ 'What kind of ground is this? Give me a short analyst brief of the area.',
    'How many buildings larger than 40 square meters are there?',
    'How close do the badlands get to the nearest dirt road?',
    'Are there any trucks or buses here?',
    'Which parts of the area are too steep for vehicles?'
  ].forEach(x => {
    const c = el('button','chip', x.length>44 ? x.slice(0,42)+'…' : x);
    c.title = x; c.onclick = () => { $('#qq').value = x; $('#qq').focus(); };
    $('#exq').append(c);
  });

  const CD = $('#cards');
  m.solutions.forEach(s => {
    const card = el('div','card');
    const h = el('h3'); h.append(el('code','',s.id.toUpperCase()),
      document.createTextNode(s.title));
    card.append(h, el('p','',s.blurb));
    const inputs = {};
    (s.controls||[]).forEach(ct => {
      if(ct.type === 'checkbox'){
        const lb = el('label','cb'); const i = el('input'); i.type='checkbox';
        lb.append(i, document.createTextNode(ct.label)); card.append(lb);
        inputs[ct.name] = () => i.checked; return;
      }
      const lb = el('label','',ct.label);
      let i;
      if(ct.type === 'select'){ i = el('select');
        ct.options.forEach(o => { const op = el('option','',o); op.value=o; i.append(op); });
      } else { i = el('input'); i.type = ct.type;
        if(ct.value!==undefined) i.value = ct.value;
        if(ct.step) i.step = ct.step; }
      lb.append(i); card.append(lb); inputs[ct.name] = () => i.value;
    });
    const go = el('button','go','Run ' + s.id.toUpperCase());
    go.onclick = async () => {
      const opts = {}; Object.keys(inputs).forEach(k => opts[k] = inputs[k]());
      $('#solh').style.display=''; $('#sout').style.display='';
      $('#sout').textContent = `running ${s.id}… (S1/S2 walk every region; this can take a while)`;
      $('#sout').scrollIntoView({behavior:'smooth',block:'nearest'});
      go.disabled = true;
      try { const r = await post('/api/solve',{id:s.id,opts}); $('#sout').textContent = r.text; }
      catch(e){ $('#sout').textContent = 'request failed: ' + e; }
      go.disabled = false;
    };
    card.append(go); CD.append(card);
  });
});

async function runQuery(){
  $('#qgo').disabled = true;
  $('#qout').textContent = 'running…';
  try {
    const r = await post('/api/query',{query:$('#q').value});
    $('#qout').textContent = r.text;
  } catch(e){ $('#qout').textContent = 'request failed: ' + e; }
  $('#qgo').disabled = false;
}
$('#qgo').onclick = runQuery;
$('#q').addEventListener('keydown', e => { if(e.key==='Enter') runQuery(); });

$('#ago').onclick = async () => {
  const q = $('#qq').value.trim(); if(!q) return;
  $('#ago').disabled = true;
  const out = $('#aout'); out.innerHTML = '';
  out.append(el('div','spin','asking '+$('#model').value+' at effort '+$('#effort').value
    +'… the model may make several tool calls before it answers.'));
  try {
    const r = await post('/api/ask',
      {question:q, model:$('#model').value, effort:$('#effort').value});
    out.innerHTML = '';
    out.append(el('div','answer', r.text));
    if(r.calls && r.calls.length){
      out.append(el('h2','', r.calls.length + ' tool call'
        + (r.calls.length>1?'s':'') + ' — every number above came from one of these'));
      const w = el('div','calls'); const tb = el('table');
      r.calls.forEach(c => {
        const tr = el('tr');
        tr.append(el('td','st '+c.status, c.status), el('td','',c.query),
                  el('td','',c.summary));
        tb.append(tr);
      });
      w.append(tb); out.append(w);
      if(r.usage) out.append(el('div','spin', r.usage));
    } else if(r.status === 'ok'){
      out.append(el('div','spin','no tool calls — the answer above rests on nothing computed.'));
    }
  } catch(e){ out.innerHTML=''; out.append(el('div','answer','request failed: '+e)); }
  $('#ago').disabled = false;
};
</script>
"""
