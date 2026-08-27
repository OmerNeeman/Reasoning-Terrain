"""The tile map -- point at ground, see all four solutions for *that* ground.

Every other view in this package is a list: findings ranked by severity, regions
ranked by score, verdicts tallied by class. Lists answer "what is worst"; they
cannot answer "what is going on *here*". This section answers the second
question. The colourised raster is drawn once, a grid of tiles is laid over it,
and clicking a tile pulls up everything the four solutions concluded about that
square of earth -- S1's findings, S2's verdicts, S3's score, S4's terrain
products, plus the OSM block it falls in -- side by side, for the same ground.

Recolouring the grid by a metric turns it into a heat map of the whole raster:
where the findings cluster, where S3 wants to look, where the mud is. That is
the map talking back.

Returns one ``<section class="card">`` with its own <style> and <script>, meant
to be dropped into an existing page. No libraries, no CDN, no network. Colours
come entirely from the host page's CSS custom properties, so it follows the
host's light/dark theme.
"""

from __future__ import annotations

import html
import json

METRICS = [
    ("none", "none", ""),
    ("s1", "S1 findings", "findings"),
    ("s3", "S3 score", "score"),
    ("trafficability", "trafficability", "0-1"),
    ("concealment", "concealment", "0-1"),
    ("built_fabric", "built fabric", "0-1"),
    ("change_volatility", "change volatility", "0-1"),
    ("traversability", "traversability", "0-1"),
]

_CSS = """
.tilemap-wrap{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);
  gap:18px;align-items:start}
@media (max-width:900px){.tilemap-wrap{grid-template-columns:minmax(0,1fr)}}
.tilemap-controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;
  margin:0 0 12px;font:12px var(--mono)}
.tilemap-controls .lbl{color:var(--dim);text-transform:uppercase;
  letter-spacing:.07em;margin-right:4px}
.tilemap-controls label.seg{border:1px solid var(--line);border-radius:4px;
  padding:3px 9px;cursor:pointer;color:var(--dim);background:var(--panel);
  user-select:none}
.tilemap-controls label.seg:hover{color:var(--ink);border-color:var(--accent)}
.tilemap-controls input{position:absolute;opacity:0;width:0;height:0}
.tilemap-controls input:checked + span{color:var(--ink);font-weight:600}
.tilemap-controls label.seg:has(input:checked){background:var(--soft);
  border-color:var(--accent);color:var(--ink)}
.tilemap-controls label.seg:has(input:focus-visible){outline:2px solid
  var(--accent);outline-offset:2px}
.tilemap-controls label.ovl{display:inline-flex;gap:6px;align-items:center;
  color:var(--dim);cursor:pointer;margin-left:auto}
.tilemap-controls label.ovl input{position:static;opacity:1;width:auto;
  height:auto}
.tilemap-stage{position:relative;width:100%;line-height:0;
  border:1px solid var(--line);border-radius:5px;overflow:hidden;
  background:var(--panel)}
.tilemap-stage img{display:block;width:100%;height:auto}
.tilemap-stage img.ovl{position:absolute;inset:0;opacity:0;
  transition:opacity .25s ease;pointer-events:none}
.tilemap-stage img.ovl.on{opacity:.85}
.tilemap-grid{position:absolute;inset:0}
.tilemap-grid button{position:absolute;margin:0;padding:0;border:0;
  background:transparent;cursor:pointer;font:inherit;color:inherit;
  box-shadow:inset 0 0 0 1px rgba(128,128,128,.28)}
.tilemap-grid button:hover{box-shadow:inset 0 0 0 2px var(--accent)}
.tilemap-grid button:focus-visible{outline:3px solid var(--accent);
  outline-offset:-3px;z-index:3}
.tilemap-grid button[aria-pressed="true"]{box-shadow:inset 0 0 0 3px
  var(--accent);z-index:2}
.tilemap-legend{display:flex;gap:8px;align-items:center;margin-top:8px;
  font:11px var(--mono);color:var(--dim)}
.tilemap-legend .ramp{flex:0 0 120px;height:9px;border-radius:2px;
  border:1px solid var(--line)}
.tilemap-panel{border:1px solid var(--line);border-radius:5px;
  background:var(--panel);padding:14px 16px;font:13px/1.55 var(--sans);
  color:var(--ink);min-height:200px}
.tilemap-panel .empty{color:var(--dim);font:12.5px/1.7 var(--mono)}
.tilemap-panel h4{font:600 12px var(--mono);text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim);margin:0 0 6px}
.tilemap-panel .head{font:600 16px var(--sans);margin:0 0 2px}
.tilemap-panel .sub{font:11.5px var(--mono);color:var(--dim);margin:0 0 12px}
.tilemap-blk{border-top:1px solid var(--line);padding:10px 0 2px}
.tilemap-blk:first-of-type{border-top:0}
.tilemap-blk .row{display:flex;justify-content:space-between;gap:10px;
  font:12px var(--mono);padding:1px 0}
.tilemap-blk .row b{font-weight:600}
.tilemap-blk .muted{color:var(--dim)}
.tilemap-bar{display:grid;grid-template-columns:1fr 46px;gap:8px;
  align-items:center;font:11.5px var(--mono);margin:3px 0}
.tilemap-bar .track{height:8px;border-radius:2px;background:var(--soft);
  overflow:hidden}
.tilemap-bar .fill{height:100%;background:var(--accent)}
.tilemap-bar .name{font:11.5px var(--mono);color:var(--ink)}
.tilemap-bar .val{text-align:right;color:var(--dim)}
.tilemap-ex{border-left:3px solid var(--accent);background:var(--soft);
  padding:6px 10px;margin:6px 0 2px;font:11.5px/1.5 var(--mono)}
.tilemap-osm{border-top:1px solid var(--line);margin-top:8px;padding-top:8px;
  font:11.5px/1.6 var(--mono);color:var(--dim)}
.tilemap-osm b{color:var(--ink);font-weight:600}
"""

_JS = """
(function(){
  var root=document.getElementById(ROOT_ID);
  if(!root||!TILES||!TILES.tiles) return;
  var H=TILES.shape[0], W=TILES.shape[1];
  var COLS=TILES.cols, ROWS=TILES.rows;
  var grid=root.querySelector('.tilemap-grid');
  var panel=root.querySelector('.tilemap-panel');
  var metric='none', pinned=null, btns=[];

  function val(t,m){
    if(m==='none') return null;
    if(m==='s1') return (t.s1&&t.s1.n)||0;
    if(m==='s3') return (t.s3&&t.s3.score)||0;
    return (t.s4&&t.s4[m])||0;
  }
  function span(m){
    if(m==='none') return [0,1];
    var hi=0;
    TILES.tiles.forEach(function(t){var v=val(t,m); if(v>hi) hi=v;});
    if(m!=='s1') hi=Math.max(hi,1);
    return [0,Math.max(hi,1e-9)];
  }
  function ramp(f){
    f=Math.max(0,Math.min(1,f));
    var r=Math.round(58+(198-58)*f), g=Math.round(122+(96-122)*f),
        b=Math.round(170+(62-170)*f);
    return 'rgba('+r+','+g+','+b+','+(0.10+0.42*f).toFixed(3)+')';
  }
  function paint(){
    var s=span(metric), lo=s[0], hi=s[1];
    btns.forEach(function(b){
      var t=TILES.tiles[b.dataset.i|0];
      if(metric==='none'){ b.style.background='transparent'; return; }
      var v=val(t,metric);
      b.style.background=ramp(hi>lo?(v-lo)/(hi-lo):0);
    });
    var lg=root.querySelector('.tilemap-legend');
    lg.style.display=metric==='none'?'none':'flex';
    if(metric!=='none'){
      root.querySelector('.lg-hi').textContent=
        (metric==='s1'? String(Math.round(hi)) : hi.toFixed(2));
      root.querySelector('.lg-name').textContent=metric.replace('_',' ');
    }
  }

  function el(tag,cls,txt){
    var e=document.createElement(tag);
    if(cls) e.className=cls;
    if(txt!==undefined&&txt!==null) e.textContent=String(txt);
    return e;
  }
  function bar(name,frac,label){
    var w=el('div','tilemap-bar');
    var l=el('div','name',name);
    var track=el('div','track'); var fill=el('div','fill');
    fill.style.width=(Math.max(0,Math.min(1,frac))*100).toFixed(1)+'%';
    track.appendChild(fill);
    var wrapl=el('div'); wrapl.appendChild(l); wrapl.appendChild(track);
    w.appendChild(wrapl); w.appendChild(el('div','val',label));
    return w;
  }
  function row(k,v,cls){
    var r=el('div','row'+(cls?' '+cls:''));
    r.appendChild(el('span',null,k));
    r.appendChild(el('b',null,v));
    return r;
  }
  function blk(title){
    var b=el('div','tilemap-blk');
    b.appendChild(el('h4',null,title));
    return b;
  }

  function show(t){
    panel.textContent='';
    var m=(TILES.tile_px*(TILES.gsd||0));
    panel.appendChild(el('div','head','row '+t.r+' , column '+t.c));
    panel.appendChild(el('div','sub',
      TILES.tile_px+' px tile  '+m.toFixed(1)+' x '+m.toFixed(1)+' m on the '+
      'ground  bbox ['+t.bbox.join(', ')+']'));

    var cb=blk('classes here');
    if(t.classes&&t.classes.length){
      t.classes.forEach(function(c){
        cb.appendChild(bar(c[0],c[1],(c[1]*100).toFixed(0)+'%'));
      });
    } else { cb.appendChild(el('div','row muted','no class breakdown')); }
    panel.appendChild(cb);

    var s1=t.s1||{}, b1=blk('S1  audit findings');
    b1.appendChild(row('findings',(s1.n||0)));
    b1.appendChild(row('max severity',(s1.max_sev||0).toFixed(2)));
    (s1.causes||[]).slice(0,4).forEach(function(c){
      b1.appendChild(row(c[0],c[1],'muted'));
    });
    if(!(s1.causes||[]).length) b1.appendChild(el('div','row muted','no causes'));
    if((s1.rids||[]).length)
      b1.appendChild(row('region ids',s1.rids.slice(0,8).join(' ')+
        (s1.rids.length>8?' ...':''),'muted'));
    panel.appendChild(b1);

    var s2=t.s2||{}, b2=blk('S2  verdicts');
    if((s2.verdicts||[]).length){
      s2.verdicts.forEach(function(v){ b2.appendChild(row(v[0],v[1])); });
    } else { b2.appendChild(el('div','row muted','no verdicts')); }
    (s2.examples||[]).slice(0,2).forEach(function(x){
      var e=el('div','tilemap-ex');
      e.appendChild(el('div',null,'#'+x.rid+'  '+x.from+'  ->  '+x.to));
      e.appendChild(el('div','muted',x.why||''));
      b2.appendChild(e);
    });
    panel.appendChild(b2);

    var s3=t.s3||{}, b3=blk('S3  worth a look?');
    b3.appendChild(row('score',(s3.score||0).toFixed(2)));
    b3.appendChild(row('selected',s3.selected?'yes':'no'));
    panel.appendChild(b3);

    var s4=t.s4||{}, b4=blk('S4  terrain products');
    Object.keys(t.s4 || {}).sort().forEach(function(k){
      var v=s4[k]||0;
      b4.appendChild(bar(k.replace('_',' '),v,v.toFixed(2)));
    });
    panel.appendChild(b4);

    var o=t.osm||{}, ob=el('div','tilemap-osm');
    ob.appendChild(el('div',null,'OSM block '+(o.block!==undefined?o.block:'-')+
      (o.block_label?'  '+o.block_label:'')));
    var l2=el('div');
    l2.appendChild(el('b',null,((o.road_frac||0)*100).toFixed(0)+'%'));
    l2.appendChild(document.createTextNode(' road  '));
    l2.appendChild(el('b',null,((o.building_frac||0)*100).toFixed(0)+'%'));
    l2.appendChild(document.createTextNode(' building  '));
    l2.appendChild(el('b',null,String(o.n_junctions||0)));
    l2.appendChild(document.createTextNode(' junctions'));
    ob.appendChild(l2);
    panel.appendChild(ob);
  }

  TILES.tiles.forEach(function(t,i){
    var b=document.createElement('button');
    b.type='button'; b.dataset.i=i; b.dataset.r=t.r; b.dataset.c=t.c;
    b.setAttribute('aria-pressed','false');
    b.setAttribute('aria-label','tile row '+t.r+' column '+t.c);
    b.title='row '+t.r+' column '+t.c;
    b.tabIndex = i===0 ? 0 : -1;
    b.style.left=(t.bbox[1]/W*100).toFixed(4)+'%';
    b.style.top=(t.bbox[0]/H*100).toFixed(4)+'%';
    b.style.width=((t.bbox[3]-t.bbox[1])/W*100).toFixed(4)+'%';
    b.style.height=((t.bbox[2]-t.bbox[0])/H*100).toFixed(4)+'%';
    b.addEventListener('mouseenter',function(){ if(!pinned) show(t); });
    b.addEventListener('focus',function(){ if(!pinned) show(t); });
    b.addEventListener('click',function(){
      if(pinned===b){ pinned=null; b.setAttribute('aria-pressed','false'); }
      else{
        if(pinned) pinned.setAttribute('aria-pressed','false');
        pinned=b; b.setAttribute('aria-pressed','true'); show(t);
      }
    });
    grid.appendChild(b); btns.push(b);
  });

  var byrc={};
  btns.forEach(function(b){ byrc[b.dataset.r+':'+b.dataset.c]=b; });
  grid.addEventListener('keydown',function(ev){
    var d={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}[ev.key];
    if(!d) return;
    var b=ev.target; if(!b.dataset||b.dataset.r===undefined) return;
    var r=(b.dataset.r|0)+d[0], c=(b.dataset.c|0)+d[1];
    r=Math.max(0,Math.min(ROWS-1,r)); c=Math.max(0,Math.min(COLS-1,c));
    var nx=byrc[r+':'+c];
    if(nx){ ev.preventDefault(); btns.forEach(function(x){x.tabIndex=-1;});
      nx.tabIndex=0; nx.focus(); }
  });

  root.querySelectorAll('.tilemap-controls input[name="tilemap-metric"]')
    .forEach(function(inp){
      inp.addEventListener('change',function(){ metric=inp.value; paint(); });
    });
  var ov=root.querySelector('.tilemap-ovl-toggle');
  if(ov) ov.addEventListener('change',function(){
    root.querySelector('.tilemap-stage img.ovl')
        .classList.toggle('on',ov.checked);
  });
  paint();
})();
"""


def render_map_section(tile_index: dict, base_png_data_uri: str,
                       overlay_png_data_uri: str = "") -> str:
    """Render the interactive tile map as one self-contained ``<section>``.

    ``tile_index`` is the dict from :func:`segmap_digest.tilemap.build_tile_index`;
    ``base_png_data_uri`` is the colourised preview of the whole raster, and the
    optional ``overlay_png_data_uri`` is an OSM overlay of identical dimensions
    that a checkbox cross-fades on top of it.
    """
    root_id = "tilemap"
    ti = tile_index or {}
    shape = list(ti.get("shape") or [1, 1])
    rows, cols = int(ti.get("rows") or 0), int(ti.get("cols") or 0)
    tile_px, gsd = int(ti.get("tile_px") or 0), float(ti.get("gsd") or 0.0)
    n_tiles = int(ti.get("n_tiles") or len(ti.get("tiles") or []))
    ground = tile_px * gsd

    sub = (f"{n_tiles} tiles  {rows} x {cols}  {tile_px} px "
           f"({ground:.1f} m) each  raster {shape[0]} x {shape[1]} px")

    segs = []
    for value, label, _unit in METRICS:
        checked = " checked" if value == "none" else ""
        segs.append(
            f'<label class="seg"><input type="radio" name="tilemap-metric" '
            f'value="{html.escape(value)}"{checked}>'
            f'<span>{html.escape(label)}</span></label>'
        )

    overlay_img = ""
    overlay_ctl = ""
    if overlay_png_data_uri:
        overlay_img = (f'<img class="ovl" alt="OSM roads and blocks overlay" '
                       f'src="{html.escape(overlay_png_data_uri, quote=True)}">')
        overlay_ctl = ('<label class="ovl"><input type="checkbox" '
                       'class="tilemap-ovl-toggle"> OSM overlay</label>')

    ramp_css = ("linear-gradient(90deg,rgba(58,122,170,.10),"
                "rgba(198,96,62,.52))")
    empty = ("Click a tile to pin it. Hover or arrow-key across the grid to "
             "preview. Whatever square you point at, this panel shows what all "
             "four solutions concluded about that same ground.")

    js = _JS.replace("ROOT_ID", json.dumps(root_id))

    return (
        f'<section class="card" id="{root_id}">'
        f"<style>{_CSS}</style>"
        f'<h2>Tile map</h2>'
        f'<p class="sub" style="font:11.5px var(--mono);color:var(--dim)">'
        f"{html.escape(sub)}</p>"
        f'<div class="tilemap-controls">'
        f'<span class="lbl">colour by</span>{"".join(segs)}{overlay_ctl}</div>'
        f'<div class="tilemap-wrap">'
        f"<div>"
        f'<div class="tilemap-stage">'
        f'<img alt="segmentation preview of the raster" '
        f'src="{html.escape(base_png_data_uri, quote=True)}">'
        f"{overlay_img}"
        f'<div class="tilemap-grid" role="group" '
        f'aria-label="tile grid, use arrow keys"></div></div>'
        f'<div class="tilemap-legend" style="display:none">'
        f'<span class="lg-name"></span><span>0</span>'
        f'<span class="ramp" style="background:{ramp_css}"></span>'
        f'<span class="lg-hi"></span></div>'
        f"</div>"
        f'<div class="tilemap-panel" aria-live="polite">'
        f'<p class="empty">{html.escape(empty)}</p></div>'
        f"</div>"
        f"<script>const TILES = {json.dumps(tile_index)};</script>"
        f"<script>{js}</script>"
        f"</section>"
    )
