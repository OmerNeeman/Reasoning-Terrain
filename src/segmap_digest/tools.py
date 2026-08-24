"""The tool surface: the S5 query verbs as Anthropic tool definitions.

S5's open question 1 was whether the model emits tool calls against these verbs
or writes code against the index. Decided: **tool calls** -- safe and auditable.
There is no code-execution path here and adding one is a separate decision.

Three things this module exists to guarantee:

  * **One place for the arithmetic.** A tool call is compiled into an `s5_query`
    query string and handed to `s5_query.query`. Nothing is recomputed here, so
    the tool answer and `segmap solve s5 --query ...` cannot drift apart.
  * **Evidence, always.** Every payload carries the region ids (or component /
    class ids) the answer came from, capped, with the total stated. A number a
    reviewer cannot go and check on the map is barely better than a guess.
  * **Refusal that reads as refusal.** "0 Houses here" and "the map does not
    know what a fence is" are different answers and are returned as different
    statuses. An empty table for the second one is a wrong answer, not a
    missing one.

Everything in this module is offline: building the definitions, dispatching a
call, and rendering the payload need no SDK and no network. `ask.ask_tools()`
is the only thing that talks to the API.
"""

from __future__ import annotations

from dataclasses import dataclass

from .index import RegionIndex
from .loader import LabelRaster
from .solutions import s4_products, s5_query
from .taxonomy import NAMES, cid

# --- limits (guesses -- see docs/solutions/S5-query.md) --------------------

# Evidence ids handed back per result. A reviewer spot-checks a handful on the
# map; the payload always states the total, so a sample cannot read as "all".
EVIDENCE_MAX_IDS = 25

# Table rows handed back to the model. Same rule: the payload says how many
# were dropped. A model that is told 40 of 900 rows is looking at 40 of 900.
TOOL_MAX_ROWS = 40

# Ceiling on one question's plan: tool calls before the runner gives up.
MAX_ITERATIONS = 12


# --- tool definitions ------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One verb, typed. `definition()` is the Anthropic tool-definition dict."""

    name: str
    description: str
    properties: dict[str, dict]
    required: tuple[str, ...] = ()

    def definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": dict(self.properties),
                "required": list(self.required),
                "additionalProperties": False,
            },
        }


def _class_arg(role: str) -> dict:
    return {
        "type": "string",
        "description": f"{role}. Exactly one of the 47 class names, spelled as in "
                       f"the legend (e.g. 'House', 'PavedRoad', 'TerraRosa'). "
                       f"Anything else is refused rather than guessed at.",
    }


# Shared by find/count/area: same filters, so the same query string is built and
# the three verbs cannot disagree about what "House over 40 m2" means.
_FILTERS: dict[str, dict] = {
    "min_area_m2": {
        "type": "number",
        "description": "drop regions smaller than this many square metres",
    },
    "slope_lt": {
        "type": "number",
        "description": "keep regions whose mean slope is below this many degrees. "
                       "Refused when no DEM is attached, because slope is then "
                       "unmeasured rather than zero.",
    },
    "slope_gt": {
        "type": "number",
        "description": "keep regions whose mean slope is above this many degrees. "
                       "Refused when no DEM is attached.",
    },
    "near_class": {
        "type": "string",
        "description": "keep only regions within near_m of this class. Requires "
                       "near_m. Measured centroid-to-nearest-pixel.",
    },
    "near_m": {
        "type": "number",
        "description": "distance threshold in metres for near_class",
    },
}

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="find",
        description=(
            "List the individual regions of one class, largest first, with area, "
            "mean slope and centroid pixel. Use this when the user wants to know "
            "*which* ones, or wants ids to check on the map."
        ),
        properties={"class_name": _class_arg("the class to list"), **_FILTERS},
        required=("class_name",),
    ),
    ToolSpec(
        name="count",
        description=(
            "Count the connected regions of one class, after any filters. Note "
            "this counts regions, not objects: two houses that share a wall are "
            "one region, and a region below the index's minimum area is not "
            "counted at all."
        ),
        properties={"class_name": _class_arg("the class to count"), **_FILTERS},
        required=("class_name",),
    ),
    ToolSpec(
        name="area",
        description=(
            "Total ground area in square metres covered by one class, after any "
            "filters. Computed from pixel counts and the raster's ground sample "
            "distance."
        ),
        properties={"class_name": _class_arg("the class to measure"), **_FILTERS},
        required=("class_name",),
    ),
    ToolSpec(
        name="distance",
        description=(
            "Minimum distance in metres between any pixel of one class and any "
            "pixel of another -- the closest approach, not a centroid distance. "
            "Returns the two pixels where the minimum occurs so it can be checked."
        ),
        properties={
            "class_a": _class_arg("first class"),
            "class_b": _class_arg("second class"),
        },
        required=("class_a", "class_b"),
    ),
    ToolSpec(
        name="corridor",
        description=(
            "Where can a vehicle actually get to, starting from a seed class? "
            "Thresholds the S4 trafficability raster, connected-components it, "
            "and reports the components that touch the seed class. Every number "
            "it returns inherits S4's guessed per-class trafficability scores."
        ),
        properties={
            "seed_class": _class_arg("the class movement starts from, e.g. 'PavedRoad'"),
            "vehicle": {
                "type": "string",
                "enum": sorted(s4_products.VEHICLES),
                "description": "vehicle class; defaults to wheeled",
            },
        },
        required=("seed_class",),
    ),
    ToolSpec(
        name="describe",
        description=(
            "Composition of the whole tile: every class present, its share of the "
            "classified area, its area in square metres, and how many regions it "
            "forms. Start here when the question is broad."
        ),
        properties={},
    ),
)

SPECS_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in SPECS}


def definitions() -> list[dict]:
    """The tool block of an Anthropic request. Stable order -- it sits in front
    of the cached prompt prefix, and reordering it would invalidate the cache."""
    return [s.definition() for s in SPECS]


# --- dispatch --------------------------------------------------------------

class ToolCallError(ValueError):
    """The call itself was malformed -- unknown tool, bad or missing argument."""


def _validate(spec: ToolSpec, args: dict) -> None:
    unknown = sorted(set(args) - set(spec.properties))
    if unknown:
        raise ToolCallError(
            f"{spec.name} has no argument(s) {', '.join(unknown)}; "
            f"accepted: {', '.join(sorted(spec.properties)) or '(none)'}"
        )
    missing = [k for k in spec.required if args.get(k) in (None, "")]
    if missing:
        raise ToolCallError(f"{spec.name} needs {', '.join(missing)}")
    if args.get("near_class") and args.get("near_m") is None:
        raise ToolCallError("near_class needs near_m (the distance in metres)")
    if args.get("slope_lt") is not None and args.get("slope_gt") is not None:
        raise ToolCallError("give at most one of slope_lt / slope_gt; the query "
                            "language supports a single slope bound")
    if spec.name == "distance" and args.get("class_a") == args.get("class_b"):
        raise ToolCallError("distance from a class to itself is 0 by definition; "
                            "give two different classes")


def query_string(name: str, args: dict) -> str:
    """Compile a tool call into an s5_query string.

    This is the whole integration: the tool layer plans, `s5_query` computes.
    The string also goes into the payload, so the user can rerun any answer with
    `segmap solve s5 --query '...'` and get the same number.
    """
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        raise ToolCallError(f"unknown tool {name!r}; available: "
                            f"{', '.join(SPECS_BY_NAME)}")
    _validate(spec, args)

    if name == "describe":
        return "describe"
    if name == "distance":
        return f"distance {args['class_a']} {args['class_b']}"
    if name == "corridor":
        return f"corridor {args['seed_class']} vehicle {args.get('vehicle') or 'wheeled'}"

    parts = [name, str(args["class_name"])]
    if args.get("min_area_m2") is not None:
        parts += ["minarea", _num(args["min_area_m2"])]
    if args.get("slope_lt") is not None:
        parts += ["slope", "<", _num(args["slope_lt"])]
    if args.get("slope_gt") is not None:
        parts += ["slope", ">", _num(args["slope_gt"])]
    if args.get("near_class"):
        parts += ["near", str(args["near_class"]), _num(args["near_m"])]
    return " ".join(parts)


def _num(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def _class_args(args: dict) -> list[str]:
    return [str(args[k]) for k in ("class_name", "class_a", "class_b", "seed_class",
                                   "near_class")
            if args.get(k)]


def call(name: str, args: dict, raster: LabelRaster, ridx: RegionIndex) -> dict:
    """Run one tool call. Never raises: a bad call is a payload, not a traceback,
    because the model has to be able to read the failure and correct itself.

    status is one of:
      ok           -- computed, with matches
      empty        -- computed, and the answer is genuinely nothing
      unanswerable -- the label raster cannot answer this at all
      error        -- the call was malformed
    """
    args = {k: v for k, v in (args or {}).items() if v is not None}
    base = {"tool": name, "arguments": args}

    try:
        q = query_string(name, args)
    except ToolCallError as exc:
        return {**base, "status": "error", "message": str(exc),
                "summary": f"malformed call: {exc}"}

    base["query"] = q
    try:
        res = s5_query.query(q, raster, ridx)
    except s5_query.Unanswerable as exc:
        return {**base, "status": "unanswerable", "message": str(exc),
                "summary": f"refused: {exc}",
                "guidance": "Say this to the user in words. Do not substitute a "
                            "near-miss class or report zero -- neither is the "
                            "answer to what was asked."}
    except s5_query.QueryError as exc:
        return {**base, "status": "error", "message": str(exc),
                "summary": f"query error: {exc}"}

    payload = {**base, "kind": res.kind}
    if res.scalar is not None:
        payload["value"] = round(float(res.scalar), 3)
        payload["unit"] = _unit(name)
    if res.note:
        payload["assumptions"] = res.note

    rows = [list(r) for r in res.rows[:TOOL_MAX_ROWS]]
    if res.rows:
        payload["header"] = list(res.header)
        payload["rows"] = rows
        payload["rows_omitted"] = max(len(res.rows) - TOOL_MAX_ROWS, 0)
        payload["rows_total"] = len(res.rows)

    payload["evidence"] = {
        "id_kind": res.id_kind,
        "ids": res.ids[:EVIDENCE_MAX_IDS],
        "ids_total": len(res.ids),
        "ids_omitted": max(len(res.ids) - EVIDENCE_MAX_IDS, 0),
        "how_to_check": _how_to_check(res.id_kind),
    }

    empty = not res.rows and not res.ids and not payload.get("value")
    if empty:
        present = [n for n in _class_args(args)
                   if n in NAMES and (raster.labels == cid(n)).any()]
        absent = [n for n in _class_args(args) if n in NAMES and n not in present]
        payload["status"] = "empty"
        payload["message"] = (
            "Computed, and the answer is zero. This is a measured absence, not a "
            "failed query"
            + (f": {', '.join(absent)} does not occur anywhere in this tile"
               if absent else " under these filters")
            + ". Report it as a finding, in words."
        )
    else:
        payload["status"] = "ok"

    payload["summary"] = _summary(payload, res)
    return payload


def _unit(name: str) -> str:
    return {"area": "m2", "distance": "m", "corridor": "m2 reachable",
            "count": "regions"}.get(name, "")


def _how_to_check(id_kind: str) -> str:
    if id_kind == "region":
        return ("region ids index the region table: `segmap digest l2` lists them "
                "with their bounding boxes")
    if id_kind == "component":
        return ("component ids are connected components of the thresholded "
                "trafficability raster; each row gives a centre pixel to look at")
    if id_kind == "class":
        return "class ids are taxonomy ids: `segmap legend --full`"
    return "no ids apply to this result"


def _summary(payload: dict, res: s5_query.QueryResult) -> str:
    bits = []
    if "value" in payload:
        bits.append(f"{payload['value']:,g} {payload['unit']}".strip())
    if payload.get("rows_total"):
        bits.append(f"{payload['rows_total']} rows")
    if res.ids:
        bits.append(f"{len(res.ids)} {res.id_kind} id(s)")
    return ", ".join(bits) or payload.get("message", "no result")


# --- rendering -------------------------------------------------------------

def render(payload: dict) -> str:
    """The tool result as the model sees it. TSV, like every other digest here:
    fewer tokens than JSON for the same content."""
    lines = [f"# {payload['tool']}: {payload.get('query', payload['arguments'])}",
             f"# status: {payload['status']}"]
    if payload.get("message"):
        lines.append(f"# {payload['message']}")
    if payload.get("guidance"):
        lines.append(f"# {payload['guidance']}")
    if payload.get("assumptions"):
        lines.append(f"# assumptions: {payload['assumptions']}")
    if "value" in payload:
        lines.append(f"= {payload['value']:,g} {payload['unit']}".rstrip())
    if payload.get("rows"):
        lines.append("\t".join(payload["header"]))
        lines += ["\t".join(str(x) for x in row) for row in payload["rows"]]
        if payload["rows_omitted"]:
            lines.append(f"# NOTE: {payload['rows_omitted']} of "
                         f"{payload['rows_total']} rows omitted by "
                         f"limit={TOOL_MAX_ROWS}")
    ev = payload.get("evidence")
    if ev and ev["ids"]:
        tail = f" (+{ev['ids_omitted']} more, {ev['ids_total']} total)" if ev["ids_omitted"] else ""
        lines.append(f"# evidence: {ev['id_kind']} ids "
                     f"{', '.join(str(i) for i in ev['ids'])}{tail}")
        lines.append(f"# {ev['how_to_check']}")
    return "\n".join(lines)
