"""The vector model: what an OSM way is once RT has finished with it.

Deliberately thin. This is not a GIS library and there is no shapely here -- a
way is its node ids, its lon/lat vertices and its tags, and everything else
(width, expected ST class, reliability) is derived on demand from `tags.py`.

Two things worth knowing about the shape of the data:

  A closed way is not necessarily an area. `highway=residential` that loops back
    on itself is a ring road, not a polygon. Only `is_area` decides, from the
    tags, and it is what separates a building (fill it) from a road (buffer it).
  Relations are skipped, and counted. A building mapped as a multipolygon
    relation -- courtyard blocks, anything with a hole -- has no `way` geometry
    of its own, so it is absent from this layer. `OsmVectors.skipped` says how
    many, because a silently missing building is exactly the kind of gap that
    manufactures a false "ST invented a house" finding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from . import tags as T


@dataclass
class OsmWay:
    id: int
    tags: dict
    lon: np.ndarray                  # (n,) float64, WGS84 degrees
    lat: np.ndarray
    nodes: tuple[int, ...] = ()      # OSM node ids, for the intersection graph

    @property
    def name(self) -> str:
        return self.tags.get("name") or self.tags.get("name:en") or ""

    @property
    def grade(self) -> str | None:
        return T.road_grade(self.tags)

    @property
    def surface(self) -> str:
        return str(self.tags.get("surface", ""))

    @property
    def kind(self) -> str:
        return T.feature_kind(self.tags)

    @property
    def half_width_m(self) -> float:
        return T.half_width_m(self.tags)

    @property
    def is_closed(self) -> bool:
        return len(self.nodes) > 3 and self.nodes[0] == self.nodes[-1]

    @property
    def is_area(self) -> bool:
        """Fill it, rather than buffer it. Tags decide, not geometry."""
        if not self.is_closed and len(self.lon) < 4:
            return False
        if T.is_building(self.tags) or self.tags.get("area") == "yes":
            return True
        if self.tags.get("natural") == "water" or self.tags.get("landuse"):
            return True
        if self.tags.get("water") or self.tags.get("amenity") or self.tags.get("leisure"):
            return self.is_closed
        return False

    @property
    def label(self) -> str:
        """What a human calls this way, in the order a human would try."""
        if self.name:
            return self.name
        if self.grade:
            return f"unnamed {self.grade}"
        return f"{self.kind}/{self.id}"

    def as_geojson(self) -> dict:
        coords = [[float(x), float(y)] for x, y in zip(self.lon, self.lat)]
        if self.is_area:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}
        return {"type": "LineString", "coordinates": coords}


@dataclass
class OsmVectors:
    ways: list[OsmWay]
    bbox: tuple[float, float, float, float]      # s, w, n, e (WGS84)
    provenance: dict = field(default_factory=dict)
    # Relation members and elements without geometry, by type. Not an error --
    # a number that has to be visible, because it bounds how complete this layer
    # can possibly be.
    skipped: dict = field(default_factory=dict)

    def by_kind(self, *kinds: str) -> list[OsmWay]:
        want = set(kinds)
        return [w for w in self.ways if w.kind in want]

    def roads(self, include_foot: bool = False) -> list[OsmWay]:
        kinds = ("road", "footway") if include_foot else ("road",)
        return self.by_kind(*kinds)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for w in self.ways:
            out[w.kind] = out.get(w.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def named_streets(self) -> set[str]:
        return {w.name for w in self.roads() if w.name}

    def fingerprint(self) -> str:
        """Identity of this snapshot, for the index cache key.

        Over the way ids and their tag *content*, not the fetch time: refetching
        the same unchanged ground must not invalidate a block partition, and a
        single retagged street must.
        """
        h = hashlib.sha256()
        h.update(f"{self.bbox}|{len(self.ways)}".encode())
        for w in sorted(self.ways, key=lambda x: x.id):
            h.update(f"{w.id}:{sorted(w.tags.items())}:{len(w.lon)}".encode())
        return h.hexdigest()[:16]

    def summary(self) -> str:
        src = self.provenance.get("source", "?")
        when = self.provenance.get("fetched_utc", "")
        counts = ", ".join(f"{k} {v}" for k, v in self.counts().items())
        sk = ", ".join(f"{k} {v}" for k, v in self.skipped.items())
        lines = [
            f"# OSM snapshot: {len(self.ways)} ways -- {counts or 'nothing'}",
            f"# bbox {self.bbox[0]:.5f},{self.bbox[1]:.5f} .. "
            f"{self.bbox[2]:.5f},{self.bbox[3]:.5f}  source: {src}"
            + (f"  fetched {when}" if when else ""),
        ]
        if sk:
            lines.append(f"# NOTE: skipped, no way geometry: {sk} -- features mapped as "
                         f"relations are absent from this layer")
        if self.named_streets:
            lines.append(f"# {len(self.named_streets)} named streets")
        return "\n".join(lines)


# --- parsing ---------------------------------------------------------------

def from_overpass_json(doc: dict, bbox=None, provenance=None) -> OsmVectors:
    ways: list[OsmWay] = []
    skipped: dict[str, int] = {}
    for el in doc.get("elements", []):
        et = el.get("type")
        geom = el.get("geometry")
        if et != "way" or not geom:
            skipped[et or "?"] = skipped.get(et or "?", 0) + 1
            continue
        pts = [(g["lon"], g["lat"]) for g in geom if g and g.get("lon") is not None]
        if len(pts) < 2:
            skipped["degenerate"] = skipped.get("degenerate", 0) + 1
            continue
        lon = np.array([p[0] for p in pts], dtype=np.float64)
        lat = np.array([p[1] for p in pts], dtype=np.float64)
        ways.append(OsmWay(id=int(el["id"]), tags=dict(el.get("tags") or {}),
                           lon=lon, lat=lat,
                           nodes=tuple(int(n) for n in el.get("nodes") or ())))
    if bbox is None:
        bbox = _bbox_of(ways)
    return OsmVectors(ways, bbox, provenance or {}, skipped)


def from_geojson(doc: dict, bbox=None, provenance=None) -> OsmVectors:
    """A GeoJSON export -- what you get from josm, osmium export, or QGIS.

    Ids and node ids are usually gone in this format, which costs the
    intersection graph its shared-node signal; `partition` falls back to
    geometric intersection when `nodes` is empty.
    """
    ways: list[OsmWay] = []
    skipped: dict[str, int] = {}
    feats = doc.get("features", []) if doc.get("type") == "FeatureCollection" else [doc]
    for i, f in enumerate(feats):
        g = f.get("geometry") or {}
        gt = g.get("type")
        props = dict(f.get("properties") or {})
        rings: list[list] = []
        if gt == "LineString":
            rings = [g["coordinates"]]
        elif gt == "MultiLineString":
            rings = list(g["coordinates"])
        elif gt == "Polygon":
            rings = [g["coordinates"][0]]
        elif gt == "MultiPolygon":
            rings = [p[0] for p in g["coordinates"]]
        else:
            skipped[str(gt)] = skipped.get(str(gt), 0) + 1
            continue
        wid = props.get("id") or props.get("osm_id") or f.get("id") or -(i + 1)
        try:
            wid = int(str(wid).split("/")[-1])
        except ValueError:
            wid = -(i + 1)
        for j, ring in enumerate(rings):
            pts = [(float(c[0]), float(c[1])) for c in ring if len(c) >= 2]
            if len(pts) < 2:
                continue
            ways.append(OsmWay(
                id=wid if j == 0 else wid * 1000 + j,
                tags={k: str(v) for k, v in props.items() if v is not None},
                lon=np.array([p[0] for p in pts]), lat=np.array([p[1] for p in pts]),
                nodes=(),
            ))
    if bbox is None:
        bbox = _bbox_of(ways)
    return OsmVectors(ways, bbox, provenance or {}, skipped)


def from_osm_xml(text: str, bbox=None, provenance=None) -> OsmVectors:
    """Overpass/JOSM XML. stdlib only -- no osmium, no lxml."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(text)
    nodes: dict[int, tuple[float, float]] = {}
    for n in root.iter("node"):
        try:
            nodes[int(n.attrib["id"])] = (float(n.attrib["lon"]), float(n.attrib["lat"]))
        except (KeyError, ValueError):
            continue
    ways: list[OsmWay] = []
    skipped: dict[str, int] = {}
    for w in root.iter("way"):
        refs = [int(nd.attrib["ref"]) for nd in w.findall("nd") if "ref" in nd.attrib]
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) < 2:
            skipped["way_without_nodes"] = skipped.get("way_without_nodes", 0) + 1
            continue
        tg = {t.attrib["k"]: t.attrib["v"] for t in w.findall("tag")
              if "k" in t.attrib and "v" in t.attrib}
        ways.append(OsmWay(id=int(w.attrib["id"]), tags=tg,
                           lon=np.array([p[0] for p in pts]),
                           lat=np.array([p[1] for p in pts]),
                           nodes=tuple(refs)))
    for r in root.iter("relation"):
        skipped["relation"] = skipped.get("relation", 0) + 1
    if bbox is None:
        bbox = _bbox_of(ways)
    return OsmVectors(ways, bbox, provenance or {}, skipped)


def _bbox_of(ways: list[OsmWay]) -> tuple[float, float, float, float]:
    if not ways:
        return (0.0, 0.0, 0.0, 0.0)
    s = min(float(w.lat.min()) for w in ways)
    n = max(float(w.lat.max()) for w in ways)
    west = min(float(w.lon.min()) for w in ways)
    e = max(float(w.lon.max()) for w in ways)
    return (s, west, n, e)


def to_overpass_json(v: OsmVectors) -> dict:
    """Round-trippable snapshot. Provenance rides along under a key Overpass
    itself does not use, so the file stays valid Overpass JSON."""
    return {
        "version": 0.6,
        "generator": "reasoning-terrain/osm",
        "rt_provenance": v.provenance,
        "rt_bbox": list(v.bbox),
        "elements": [
            {"type": "way", "id": w.id, "nodes": list(w.nodes), "tags": w.tags,
             "geometry": [{"lon": float(x), "lat": float(y)}
                          for x, y in zip(w.lon, w.lat)]}
            for w in v.ways
        ],
    }


def load_json_doc(doc: dict, provenance=None) -> OsmVectors:
    """Sniff Overpass JSON vs GeoJSON."""
    bbox = tuple(doc["rt_bbox"]) if doc.get("rt_bbox") else None
    prov = provenance or doc.get("rt_provenance") or {}
    if "elements" in doc:
        return from_overpass_json(doc, bbox, prov)
    if doc.get("type") in ("FeatureCollection", "Feature"):
        return from_geojson(doc, bbox, prov)
    raise ValueError("not an Overpass JSON or GeoJSON document: expected an "
                     "'elements' or 'features' key")
