"""The tool layer, exercised with no SDK installed and no network.

What these lock down is the part that is not the model's job: the tool
definitions, the dispatch from tool-name + args into `s5_query`, the evidence
payload, and the difference between "no matches" and "the map cannot answer
that". The live API path (`ask.ask_tools`) is not exercised here -- only that it
fails with a usable message when the SDK is absent.
"""

import importlib.util

import pytest

from segmap_digest import ask, synth, tools
from segmap_digest.index import build_regions
from segmap_digest.solutions import s5_query
from segmap_digest.taxonomy import NAMES, cid


@pytest.fixture(scope="module")
def tile():
    return synth.generate(size=384, seed=5)


@pytest.fixture(scope="module")
def ridx(tile):
    return build_regions(tile)


@pytest.fixture(scope="module")
def flat():
    """No DEM: every slope is 0.0, which is not the same as flat ground."""
    r = synth.generate(size=192, seed=8, with_dem=False)
    return r, build_regions(r)


# --- the definitions -------------------------------------------------------

def test_definitions_are_well_typed():
    defs = tools.definitions()
    assert len(defs) == len(tools.SPECS) == len({d["name"] for d in defs})
    for d in defs:
        schema = d["input_schema"]
        assert d["description"].strip()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False, \
            "an unconstrained schema lets the model invent arguments"
        assert set(schema["required"]) <= set(schema["properties"])
        for name, prop in schema["properties"].items():
            assert prop["type"] in ("string", "number", "integer", "boolean"), name
            assert prop["description"].strip(), name


def test_every_tool_compiles_to_a_real_query(tile, ridx):
    """The tools are a front end for s5_query, not a second implementation."""
    sample = {
        "find": {"class_name": "House"},
        "count": {"class_name": "House"},
        "area": {"class_name": "House"},
        "distance": {"class_a": "House", "class_b": "Water"},
        "corridor": {"seed_class": "PavedRoad"},
        "describe": {},
    }
    assert set(sample) == set(tools.SPECS_BY_NAME)
    for name, args in sample.items():
        q = tools.query_string(name, args)
        assert s5_query.query(q, tile, ridx) is not None, q


def test_numbers_match_the_query_layer_exactly(tile, ridx):
    """One place for the arithmetic: the tool and the CLI must not drift."""
    for name, args, q in (
        ("count", {"class_name": "House"}, "count House"),
        ("area", {"class_name": "Batha", "min_area_m2": 50}, "area Batha minarea 50"),
        ("distance", {"class_a": "House", "class_b": "Water"}, "distance House Water"),
    ):
        payload = tools.call(name, args, tile, ridx)
        assert payload["query"] == q
        assert payload["value"] == pytest.approx(
            s5_query.query(q, tile, ridx).scalar, rel=1e-3)


# --- evidence --------------------------------------------------------------

def test_every_answer_carries_checkable_evidence(tile, ridx):
    known = {r.id for r in ridx.regions}
    for name, args in (("find", {"class_name": "House"}),
                       ("count", {"class_name": "House"}),
                       ("area", {"class_name": "House"}),
                       ("describe", {}),
                       ("corridor", {"seed_class": "PavedRoad"})):
        ev = tools.call(name, args, tile, ridx)["evidence"]
        assert ev["ids"], f"{name} answered with no provenance"
        assert ev["ids_total"] >= len(ev["ids"])
        assert ev["how_to_check"]
        if ev["id_kind"] == "region":
            assert set(ev["ids"]) <= known


def test_evidence_sample_states_what_it_dropped(tile, ridx):
    payload = tools.call("count", {"class_name": "Batha"}, tile, ridx)
    ev = payload["evidence"]
    assert ev["ids_total"] > tools.EVIDENCE_MAX_IDS, "fixture should have many Batha"
    assert len(ev["ids"]) == tools.EVIDENCE_MAX_IDS
    assert ev["ids_omitted"] == ev["ids_total"] - tools.EVIDENCE_MAX_IDS
    assert f"{ev['ids_total']} total" in tools.render(payload), \
        "a capped sample that does not say so reads as the whole set"


def test_rows_are_never_silently_truncated(tile, ridx):
    payload = tools.call("find", {"class_name": "Batha"}, tile, ridx)
    assert payload["rows_total"] > tools.TOOL_MAX_ROWS
    assert len(payload["rows"]) == tools.TOOL_MAX_ROWS
    assert payload["rows_omitted"] == payload["rows_total"] - tools.TOOL_MAX_ROWS
    assert "rows omitted" in tools.render(payload)


def test_payload_reports_its_thresholds(tile, ridx):
    payload = tools.call("corridor", {"seed_class": "PavedRoad"}, tile, ridx)
    assert str(s5_query.CORRIDOR_MIN_TRAFFIC) in payload["assumptions"], \
        "a corridor answer that hides its threshold is hiding its main assumption"


# --- refusal, and the difference from a real zero --------------------------

def test_absent_class_is_an_empty_answer_not_a_failure(tile, ridx):
    absent = next(n for n in NAMES if not (tile.labels == cid(n)).any())
    payload = tools.call("count", {"class_name": absent}, tile, ridx)
    assert payload["status"] == "empty"
    assert "measured absence" in payload["message"]
    assert payload["value"] == 0


def test_out_of_vocabulary_is_refused_in_words(tile, ridx):
    for term in ("Fence", "Truck", "SchoolBuilding", "RoofColour"):
        payload = tools.call("find", {"class_name": term}, tile, ridx)
        assert payload["status"] == "unanswerable", term
        assert "vocabulary" in payload["message"]
        assert "rows" not in payload, "a refusal must not render as an empty table"
        assert "unanswerable" in tools.render(payload)


def test_refusal_and_empty_are_distinguishable(tile, ridx):
    refused = tools.call("find", {"class_name": "Fence"}, tile, ridx)
    empty = tools.call("find", {"class_name": "Car", "min_area_m2": 1e9},
                       tile, ridx)
    assert refused["status"] != empty["status"]
    assert empty["status"] == "empty"


def test_slope_filter_without_a_dem_abstains(flat):
    raster, ridx = flat
    assert not ridx.has_terrain
    payload = tools.call("find", {"class_name": "House", "slope_gt": 10.0},
                         raster, ridx)
    assert payload["status"] == "unanswerable"
    assert "DEM" in payload["message"]


# --- malformed calls -------------------------------------------------------

def test_bad_calls_are_payloads_not_tracebacks(tile, ridx):
    bad = [
        ("nosuchtool", {}),
        ("find", {"class_name": "House", "colour": "red"}),
        ("distance", {"class_a": "House"}),
        ("find", {"class_name": "House", "near_class": "Water"}),
        ("find", {"class_name": "House", "slope_lt": 5, "slope_gt": 10}),
        ("distance", {"class_a": "House", "class_b": "House"}),
    ]
    for name, args in bad:
        payload = tools.call(name, args, tile, ridx)
        assert payload["status"] == "error", (name, args)
        assert payload["message"], "the model has to be able to read the failure"


# --- the live path ---------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("anthropic") is not None,
                    reason="SDK installed; this asserts the no-SDK failure mode")
def test_ask_tools_fails_with_an_actionable_message(tile, ridx):
    with pytest.raises(SystemExit, match="Anthropic SDK"):
        ask.ask_tools("how many houses?", tile, ridx, legend="")


def test_tool_guidance_is_in_the_cached_system_block():
    block = ask.build_system("LEGEND", "PRIORS", extra=ask.TOOL_ROLE)[0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert "LEGEND" in block["text"] and "unanswerable" in block["text"]
