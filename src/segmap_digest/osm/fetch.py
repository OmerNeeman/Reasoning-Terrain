"""Getting the vectors: a file on disk, or Overpass over the raster's own bbox.

Order of preference, and it matters:

  1. an explicit path the caller gave;
  2. a snapshot already cached for this bbox and filter set;
  3. Overpass -- and only then, only if allowed, and it says so on stderr first.

The snapshot is the reproducibility unit. OSM changes under you: a block
partition built on Tuesday and rebuilt on Friday can differ because someone in
Gaza mapped an alley, and nothing in the output would say so. So a fetch writes
the response to disk with its query, its bbox, its server and its UTC time, and
every later run reads that file. Refetching is a deliberate act
(`--osm-refresh`), not a side effect of asking a question twice.

Nothing here is imported unless an OSM command is actually run: `urllib` is
stdlib, but the *decision* to touch the network never happens implicitly.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .vectors import OsmVectors, from_osm_xml, load_json_doc, to_overpass_json

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "reasoning-terrain/0.1 (segmentation QA; contact via repo owner)"

# Named filter sets. `all` is the default because the layer is cheap and the
# expensive part -- the burn and the block build -- is cached downstream; but a
# roads-only fetch over a city is 20x smaller, and on a slow link that matters.
FILTER_SETS: dict[str, tuple[str, ...]] = {
    "roads": ('way["highway"]',),
    "roads+buildings": ('way["highway"]', 'way["building"]'),
    "all": (
        'way["highway"]',
        'way["building"]',
        'way["waterway"]',
        'way["natural"="water"]',
        'way["landuse"]',
        'way["barrier"]',
        'way["railway"]',
    ),
}

# Overpass server-side timeout, seconds. A 175 km2 AOI with 6,000 ways answers
# in a few seconds; the ceiling is for the day someone points this at a province.
QUERY_TIMEOUT_S = 300

# Waits between whole rounds of mirror attempts, seconds. The last entry is 0:
# it marks the final round, after which the failure is reported.
RETRY_DELAYS_S: tuple[int, ...] = (5, 20, 0)

# Metres of ground added on every side of the raster. Blocks are closed by roads
# that may lie just outside the tile: without a margin every edge block leaks
# and the partition merges half the AOI into one component.
BBOX_MARGIN_M = 150.0

# One degree of latitude, metres. Longitude is scaled by cos(lat) below. This is
# a spherical approximation and it is used only to pad a query bbox, where being
# 0.3% out is irrelevant.
DEG_LAT_M = 111_320.0


# --- where to look ---------------------------------------------------------

def bbox_of_raster(raster, margin_m: float = BBOX_MARGIN_M) -> tuple[float, float, float, float]:
    """(s, w, n, e) in WGS84 degrees, padded.

    Raises rather than guessing when the raster is not georeferenced: an OSM
    join against a raster whose position is unknown produces a plausible-looking
    partition of the wrong ground, which is worse than no partition at all.
    """
    t = getattr(raster, "transform", None)
    if t is None:
        raise ValueError(
            "this raster has no affine transform, so its position on the Earth "
            "is unknown and OSM cannot be joined to it. A GeoTIFF input carries "
            "one; .npy and PNG inputs do not. Load the .tif, or pass an explicit "
            "--osm-bbox S,W,N,E."
        )
    h, w = raster.shape
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    xs, ys = [], []
    for c, r in corners:
        x, y = t * (c, r)
        xs.append(x)
        ys.append(y)
    west, east, south, north = min(xs), max(xs), min(ys), max(ys)

    crs = getattr(raster, "crs", None)
    if crs is not None and not _is_wgs84(crs):
        try:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(
                crs, "EPSG:4326", west, south, east, north, densify_pts=21)
        except ImportError as exc:      # pragma: no cover - only on a projected CRS
            raise ValueError(
                f"raster CRS is {crs}, not WGS84, and reprojecting the bbox to "
                f"lon/lat needs rasterio (pip install -e '.[geo]')"
            ) from exc

    lat_mid = (south + north) / 2.0
    dlat = margin_m / DEG_LAT_M
    import math
    dlon = margin_m / max(DEG_LAT_M * math.cos(math.radians(lat_mid)), 1.0)
    return (south - dlat, west - dlon, north + dlat, east + dlon)


def _is_wgs84(crs) -> bool:
    try:
        return int(getattr(crs, "to_epsg", lambda: 0)() or 0) == 4326
    except Exception:                    # pragma: no cover - exotic CRS objects
        return "4326" in str(crs)


def snapshot_key(bbox, filters: str) -> str:
    b = ",".join(f"{v:.6f}" for v in bbox)
    return hashlib.sha256(f"{b}|{filters}".encode()).hexdigest()[:16]


def snapshot_path(cache_dir, bbox, filters: str) -> Path:
    return Path(cache_dir) / "osm" / f"{snapshot_key(bbox, filters)}.json"


# --- reading -------------------------------------------------------------

def load_file(path) -> OsmVectors:
    """Overpass JSON, GeoJSON, or .osm XML -- sniffed, not trusted to the suffix."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no OSM file at {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    head = text.lstrip()[:200]
    prov = {"source": str(p), "bytes": p.stat().st_size}
    if head.startswith("<"):
        v = from_osm_xml(text, provenance=prov)
    else:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{p} is neither XML nor JSON ({exc}). Accepted: Overpass JSON "
                f"(`out geom;`), GeoJSON, or .osm XML."
            ) from exc
        v = load_json_doc(doc, provenance={**(doc.get("rt_provenance") or {}), **prov})
    if not v.ways:
        raise ValueError(
            f"{p} parsed but contains no usable ways. If it came from Overpass, "
            f"the query needs `out geom;` (not `out;`) so each way carries its "
            f"geometry inline."
        )
    return v


# --- fetching ------------------------------------------------------------

def build_query(bbox, filters: str = "all", timeout_s: int = QUERY_TIMEOUT_S) -> str:
    if filters not in FILTER_SETS:
        raise ValueError(f"unknown OSM filter set {filters!r}; "
                         f"choose from {', '.join(FILTER_SETS)}")
    s, w, n, e = bbox
    box = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    body = "".join(f"  {f}{box};\n" for f in FILTER_SETS[filters])
    return f"[out:json][timeout:{timeout_s}];\n(\n{body});\nout geom;\n"


def fetch(bbox, filters: str = "all", quiet: bool = False,
          urls: tuple[str, ...] = OVERPASS_URLS) -> OsmVectors:
    """One Overpass call. Loud on purpose: this is the only step that leaves
    the machine, and a reader of the log should see the exact query."""
    query = build_query(bbox, filters)
    if not quiet:
        print(f"# OSM: no cached snapshot for this bbox -- querying Overpass\n"
              + "".join(f"#   {ln}\n" for ln in query.strip().splitlines()),
              file=sys.stderr, end="")
    data = urllib.parse.urlencode({"data": query}).encode()
    last: Exception | None = None
    # Overpass is a free service under permanent load: a 504 or a 500 on the
    # first try is routine and means "busy", not "wrong query" -- measured, this
    # exact bbox failed twice on both mirrors and then answered in 3 s. So every
    # mirror is tried, then the whole set again after a wait, before giving up.
    for attempt, delay in enumerate(RETRY_DELAYS_S):
        for url in urls:
            req = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT_S + 30) as resp:
                    raw = resp.read()
                doc = json.loads(raw)
                prov = {
                    "source": url,
                    "query": query,
                    "bbox": list(bbox),
                    "filters": filters,
                    "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n_elements": len(doc.get("elements", [])),
                    "attempts": attempt + 1,
                }
                v = load_json_doc(doc, provenance=prov)
                # The *query* box, not the data box: one landuse polygon
                # extending past the AOI must not redefine what this snapshot
                # covers, and the cache key is computed from the query box.
                v.bbox = tuple(bbox)
                return v
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    json.JSONDecodeError, OSError) as exc:
                last = exc
                if not quiet:
                    print(f"# OSM: {url} failed ({exc})", file=sys.stderr)
        if delay:
            if not quiet:
                print(f"# OSM: all mirrors busy; retrying in {delay}s "
                      f"({attempt + 1}/{len(RETRY_DELAYS_S)})", file=sys.stderr)
            time.sleep(delay)
    raise ConnectionError(
        f"every Overpass mirror failed (last error: {last}). Either this machine "
        f"is offline or Overpass is rate-limiting. Two ways forward: run the "
        f"query above at https://overpass-turbo.eu, export as JSON, and pass "
        f"`--osm <file.json>`; or wait and retry -- nothing else in the pipeline "
        f"needs the network."
    )


def save_snapshot(v: OsmVectors, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_overpass_json(v), ensure_ascii=False), encoding="utf-8")
    return p


# --- the one function callers use ----------------------------------------

def for_raster(raster, source: str | None = None, cache_dir=".segmap_cache",
               filters: str = "all", refresh: bool = False,
               allow_network: bool = True, bbox=None,
               quiet: bool = False) -> OsmVectors:
    """The vectors covering `raster`, from a file, the snapshot cache, or Overpass.

    `source` is an explicit path and short-circuits everything else. `bbox`
    overrides the raster's own footprint, for a raster with no georeference.
    """
    if source:
        v = load_file(source)
        if not quiet:
            print(f"# OSM: {len(v.ways)} ways from {source}", file=sys.stderr)
        return v

    box = tuple(bbox) if bbox else bbox_of_raster(raster)
    snap = snapshot_path(cache_dir, box, filters)
    if snap.exists() and not refresh:
        v = load_file(snap)
        if not quiet:
            when = v.provenance.get("fetched_utc", "unknown time")
            print(f"# OSM: {len(v.ways)} ways from cached snapshot {snap.name} "
                  f"(fetched {when})", file=sys.stderr)
        return v
    if not allow_network:
        raise ConnectionError(
            f"no cached OSM snapshot at {snap} and --osm-offline was given. "
            f"Fetch once without it, or pass --osm <file>."
        )
    v = fetch(box, filters=filters, quiet=quiet)
    v.bbox = box
    save_snapshot(v, snap)
    if not quiet:
        print(f"# OSM: {len(v.ways)} ways fetched and cached at {snap}",
              file=sys.stderr)
    return v
