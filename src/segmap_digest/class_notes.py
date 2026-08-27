"""What the analyst knows that the class name does not say.

`TerraRosa` is one token. What an analyst means by it is a paragraph, and the
paragraph is the part a reasoning model needs -- the class names are the entire
interface between a segmenter that learned integers and a model that knows
geology. `taxonomy.py` carries one sentence per class, written by a programmer
from a textbook, and the file says in its own header that it needs expert review.

This module is where that review lands. Three properties it was designed for:

  **A domain expert can edit it without asking anyone.** It is Markdown, one
    section per class, `field: value`. Not Python -- a taxonomy that needs a pull
    request to correct is a taxonomy that stays wrong. Not JSON -- nobody writes
    three paragraphs of prose inside a quoted string.
  **Four fields are machine-read.** That is what separates this from a glossary:
    `confused_with` feeds S2's candidate shortlist, `season` gates S6's
    phenology reasoning, `never` is a candidate prior, `scale` can replace S2's
    guessed area bands. The rest is prose for the model.
  **Nothing is mandatory and nothing is silently dropped.** A missing file is
    today's behaviour exactly. A field nobody filled reads as unfilled rather
    than as empty. A field name the parser does not know is kept and shown, not
    discarded -- if an expert writes something down, losing it is the one
    unacceptable outcome.

On `source` and `confidence`, which look like bureaucracy and are not: see
`taxonomy.RETIRED_PRIORS`. Four priors in this repo came from a soil-genesis
textbook, were never attributed, were wrong about this model, and produced 9,793
false findings on one AOI before anyone could work out where they had come from.
An attributed claim can be checked with its author. An unattributed one has to be
re-derived from scratch or thrown away.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import BY_NAME, CLASSES, NAMES

# Where the notes live unless told otherwise. A repo-relative default, so a
# checkout carries its own class definitions.
DEFAULT_PATH = Path(
    os.environ.get("SEGMAP_CLASS_NOTES",
                   Path(__file__).resolve().parents[2] / "docs" / "class_notes.md")
)

# The fields this module understands. Order is the order they are written and
# printed in, and it is the order an expert should fill them: what it is, how you
# recognise it, how you tell it from its neighbours, what it means operationally.
FIELDS: tuple[str, ...] = (
    "aka",
    "what_it_is",
    "looks_like",
    "confused_with",
    "implies",
    "scale",
    "season",
    "never",
    "source",
    "confidence",
)

# The subset that code reads rather than the model. Changing this set changes
# behaviour, so it is stated once, here.
MACHINE_READ: tuple[str, ...] = ("confused_with", "season", "never", "scale")

CONFIDENCE_VALUES = ("high", "medium", "low")

# A value that is still the template's placeholder counts as unfilled. Without
# this, a coverage report reads 100% complete the moment the template is
# committed, which is worse than reading 0%.
PLACEHOLDERS = ("", "-", "--", "?", "tbd", "todo", "n/a", "na", "none yet")

# The template seeds `what_it_is` with the one-line definition from taxonomy.py
# so an expert has something to correct rather than a blank page. A seeded value
# is NOT filled -- without this, a fresh template reports 47 of 47 classes
# documented the moment it is committed, which is a worse lie than reporting
# zero. The marker is removed by whoever rewrites the line.
SEED_MARKER = "[from taxonomy.py"

_SECTION = re.compile(r"^##\s+(\S+)\s*$")
_FIELD = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_ ]*?)\s*:\s?(?P<val>.*)$")
# `Name — how you tell them apart`, with any of the three dashes people type.
_CONFUSION = re.compile(r"^\s*(?P<name>[A-Za-z][A-Za-z0-9]*)\s*(?:[-–—]\s*"
                        r"(?P<why>.*))?$")


@dataclass
class ClassNote:
    """One class's entry. Every field is a string; absent means never filled."""
    class_name: str
    fields: dict[str, str] = field(default_factory=dict)
    # Field names the parser does not know. Kept, shown, never machine-read.
    extra: dict[str, str] = field(default_factory=dict)
    line: int = 0

    def get(self, key: str) -> str:
        return self.fields.get(key, "")

    def filled(self, key: str) -> bool:
        v = self.get(key).strip().lower()
        return bool(v) and v not in PLACEHOLDERS and not v.startswith(SEED_MARKER)

    def seeded(self, key: str) -> bool:
        """Still carrying the taxonomy's placeholder text."""
        return self.get(key).strip().lower().startswith(SEED_MARKER)

    @property
    def filled_fields(self) -> list[str]:
        return [f for f in FIELDS if self.filled(f)]

    @property
    def is_empty(self) -> bool:
        return not self.filled_fields and not self.extra

    @property
    def confused_with(self) -> list[tuple[str, str]]:
        """[(class name, how to tell them apart)]. The S2-facing field.

        Unknown class names are dropped HERE and reported by `check()`: a
        shortlist silently containing a class that does not exist would fail
        somewhere far away from the typo that caused it.
        """
        out: list[tuple[str, str]] = []
        for part in re.split(r"[;\n]", self.get("confused_with")):
            m = _CONFUSION.match(part)
            if not m:
                continue
            name = m.group("name")
            if name in BY_NAME:
                out.append((name, (m.group("why") or "").strip()))
        return out

    @property
    def unknown_confusions(self) -> list[str]:
        out = []
        for part in re.split(r"[;\n]", self.get("confused_with")):
            m = _CONFUSION.match(part)
            if m and m.group("name") not in BY_NAME:
                out.append(m.group("name"))
        return out

    @property
    def confidence(self) -> str:
        v = self.get("confidence").strip().lower()
        return v if v in CONFIDENCE_VALUES else ""

    def render(self, fields: tuple[str, ...] = FIELDS) -> str:
        """This class's note, as the model should see it."""
        lines = [f"## {self.class_name}"]
        for f in fields:
            if self.filled(f):
                lines.append(f"{f}: {self.get(f)}")
        for k, v in self.extra.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


@dataclass
class NoteSet:
    notes: dict[str, ClassNote] = field(default_factory=dict)
    path: Path | None = None
    # Section headings that are not class names. Reported, never guessed at.
    unknown_sections: list[tuple[str, int]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return any(not n.is_empty for n in self.notes.values())

    def get(self, name: str) -> ClassNote | None:
        n = self.notes.get(name)
        return n if n is not None and not n.is_empty else None

    @property
    def documented(self) -> list[str]:
        return [n for n in NAMES if self.get(n)]

    def render(self, names=None, fields: tuple[str, ...] = FIELDS) -> str:
        """Notes for the classes in play -- not all 47.

        This is the token discipline the rest of the repo already applies to
        digests: a full note set is several thousand tokens, and a question about
        three classes should pay for three.
        """
        want = list(names) if names is not None else self.documented
        out = [self.get(n).render(fields) for n in want if self.get(n)]
        missing = [n for n in want if not self.get(n)]
        text = "\n\n".join(out)
        if missing:
            text += (f"\n\n# NOTE: no analyst notes on file for "
                     f"{', '.join(missing)} -- the taxonomy's one-line "
                     f"definition is all there is for {'them' if len(missing) > 1 else 'it'}")
        return text


# --- parsing ---------------------------------------------------------------

def parse(text: str, path: Path | None = None) -> NoteSet:
    notes: dict[str, ClassNote] = {}
    unknown: list[tuple[str, int]] = []
    current: ClassNote | None = None
    key: str | None = None
    is_extra = False

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if line.startswith(">") or line.startswith("<!--"):
            continue                       # template instructions, not content
        m = _SECTION.match(line)
        if m:
            name = m.group(1)
            current = ClassNote(class_name=name, line=lineno)
            key, is_extra = None, False
            if name in BY_NAME:
                notes[name] = current
            else:
                unknown.append((name, lineno))
                current = None
            continue
        if current is None:
            continue
        if not line.strip():
            if key:
                target = current.extra if is_extra else current.fields
                if target.get(key):
                    target[key] = target[key] + "\n"
            continue
        fm = _FIELD.match(line)
        if fm and not line.startswith((" ", "\t")):
            key = fm.group("key").strip().lower().replace(" ", "_")
            val = fm.group("val").strip()
            is_extra = key not in FIELDS
            (current.extra if is_extra else current.fields)[key] = val
        elif key:
            target = current.extra if is_extra else current.fields
            target[key] = (target[key] + " " + line.strip()).strip()

    for n in NAMES:
        notes.setdefault(n, ClassNote(class_name=n))
    return NoteSet(notes, path, unknown)


def load(path=None, required: bool = False) -> NoteSet:
    """The note set, or an empty one. A missing file is not an error.

    Deliberately: every command that can use notes calls this, and the repo has
    to behave exactly as it did before this module existed until somebody writes
    the content.
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        if required:
            raise FileNotFoundError(
                f"no class notes at {p}. `segmap notes --template > {p}` writes "
                f"an empty one with all {len(NAMES)} classes stubbed."
            )
        return NoteSet({n: ClassNote(class_name=n) for n in NAMES}, None)
    return parse(p.read_text(encoding="utf-8"), p)


_DEFAULT: NoteSet | None = None


def default() -> NoteSet:
    """The note set from the default path, parsed once per process.

    Every solution calls this on every region it adjudicates; re-reading and
    re-parsing a 600-line Markdown file per region would be the slowest thing in
    S2 by an order of magnitude.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load()
    return _DEFAULT


def reset_cache() -> None:
    """For tests, and for `--notes <path>` overriding the default mid-process."""
    global _DEFAULT
    _DEFAULT = None


def legend_with_notes(ns: NoteSet, names=None) -> str:
    """The class legend, with the analyst notes folded in where they exist.

    Separate from `taxonomy.legend` on purpose: `taxonomy` is imported by
    everything and must not depend on a file that may not exist.
    """
    from .taxonomy import BY_NAME as _BY_NAME

    want = list(names) if names is not None else NAMES
    out = []
    for n in want:
        c = _BY_NAME[n]
        extra = f" [{c.lithology}/{c.morphology}]" if c.lithology else ""
        out.append(f"{c.id}\t{c.name}\t{c.group}{extra}\t{c.definition}")
        note = ns.get(n)
        if note:
            for f in FIELDS:
                if note.filled(f):
                    out.append(f"\t\t{f}: {note.get(f)}")
    return "\n".join(out)


# --- reporting -------------------------------------------------------------

def check(ns: NoteSet) -> list[str]:
    """Problems worth fixing before anything reasons over this file."""
    out: list[str] = []
    for name, lineno in ns.unknown_sections:
        out.append(f"line {lineno}: `## {name}` is not one of the {len(NAMES)} "
                   f"class names -- the section is ignored")
    for n in NAMES:
        note = ns.notes.get(n)
        if note is None or note.is_empty:
            continue
        for bad in note.unknown_confusions:
            out.append(f"{n}: `confused_with` names `{bad}`, which is not a "
                       f"class -- that pair will not reach S2's shortlist")
        if note.filled("confidence") and not note.confidence:
            out.append(f"{n}: confidence `{note.get('confidence')}` is not one "
                       f"of {', '.join(CONFIDENCE_VALUES)}")
        if note.filled("what_it_is") and not note.filled("source"):
            out.append(f"{n}: has content but no `source`. See "
                       f"taxonomy.RETIRED_PRIORS for what an unattributed claim "
                       f"cost this repo")
        for k in note.extra:
            out.append(f"{n}: field `{k}` is not one of {', '.join(FIELDS)} -- "
                       f"kept and shown to the model, but no code reads it")
    return out


def coverage(ns: NoteSet, class_area: dict[int, float] | None = None,
             limit: int | None = None) -> str:
    """Which classes are documented, ranked by how much of THIS map they cover.

    The ranking is the point. On the sinai mosaic `Rendzina` alone is 64.9% of
    classified pixels; one good paragraph there is worth twenty on classes that
    never appear in the AOI. Without an area column, this is an alphabetical
    to-do list that gets worked in the wrong order.
    """
    total = sum(class_area.values()) if class_area else 0.0
    rows = []
    for c in CLASSES:
        note = ns.notes.get(c.name)
        area = (class_area or {}).get(c.id, 0.0)
        rows.append((area, c, note))
    rows.sort(key=lambda t: (-t[0], t[1].id))
    if class_area is None:
        rows.sort(key=lambda t: t[1].id)

    n_doc = len(ns.documented)
    n_seed = sum(1 for n in NAMES
                 if ns.notes.get(n) and ns.notes[n].seeded("what_it_is"))
    documented_area = sum(a for a, c, n in rows if n and not n.is_empty)
    head = [
        f"# class notes: {n_doc} of {len(NAMES)} classes have at least one field "
        f"filled" + (f" ({ns.path})" if ns.path else " (no notes file loaded)"),
    ]
    if n_seed:
        head.append(f"# {n_seed} classes still carry the seeded taxonomy.py "
                    f"definition, which counts as UNFILLED -- it is the text "
                    f"this file exists to replace")
    if class_area:
        head.append(f"# those classes cover {documented_area / max(total, 1e-9):.1%} "
                    f"of classified area on this map")
        head.append("# ranked by area on THIS map, so effort goes where the map is")
    head.append("# machine-read fields: " + ", ".join(MACHINE_READ)
                + " -- the rest is prose the model reads")
    head.append("class\tpct_map\tfilled\tmissing")
    shown = rows if limit is None else rows[:limit]
    for area, c, note in shown:
        filled = note.filled_fields if note else []
        missing = [f for f in FIELDS if f not in filled]
        pct = f"{area / max(total, 1e-9):.2%}" if class_area else "-"
        head.append(f"{c.name}\t{pct}\t{','.join(filled) or 'NONE'}\t"
                    f"{','.join(missing) if missing else '-'}")
    if limit is not None and len(rows) > limit:
        head.append(f"# NOTE: {len(rows) - limit} classes omitted by limit={limit}")
    return "\n".join(head)


def template(seed_definitions: bool = True) -> str:
    """An empty notes file, all 47 classes stubbed, ready to fill."""
    out = [
        "# Class notes — what the analyst knows that the name does not say",
        "",
        "> One section per class. Fill what you know, skip what you do not; a",
        "> blank field reads as unfilled, not as empty. Lines starting with `>`",
        "> are instructions and are ignored by the parser.",
        ">",
        "> Four fields are read by code, not just by the model:",
        ">",
        "> - `confused_with` — `OtherClass — how you tell them apart; ...`",
        ">   Feeds S2's candidate shortlist. A measured confusion beats the",
        ">   taxonomy's distance metric every time.",
        "> - `season` — does this label depend on the date the imagery was taken?",
        ">   Feeds S6's phenology gate.",
        "> - `never` — what this can never be adjacent to, or never look like.",
        ">   A candidate prior. Admitted ONLY if it is a statement about",
        ">   geometry, position or physics — a statement about how the thing came",
        ">   to exist is a statement about the world, not about the label. See",
        ">   `taxonomy.RETIRED_PRIORS` for what happened last time that line was",
        ">   crossed.",
        "> - `scale` — typical size and shape as a polygon. Replaces the guessed",
        ">   area and elongation bands in S2.",
        ">",
        "> `source` and `confidence` are not bureaucracy: an attributed claim can",
        "> be checked with its author, an unattributed one has to be re-derived or",
        "> thrown away.",
        "",
        "> `what_it_is` is seeded below with the one-line definition currently in",
        "> `taxonomy.py`. Those were written by a programmer from a textbook and",
        "> are exactly what needs replacing — overwrite them.",
        "",
    ]
    for c in CLASSES:
        out.append(f"## {c.name}")
        out.append("aka:")
        if seed_definitions:
            out.append(f"what_it_is: [FROM taxonomy.py, NEEDS REVIEW] {c.definition}")
        else:
            out.append("what_it_is:")
        for f in FIELDS:
            if f in ("aka", "what_it_is"):
                continue
            out.append(f"{f}:")
        out.append("")
    return "\n".join(out)
