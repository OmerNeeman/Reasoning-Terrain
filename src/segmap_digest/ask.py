"""Optional: ask Claude a question about a tile.

Two paths, and the difference matters:

  `ask_tools()`  -- the model plans, `tools.py` computes. Every number in the
      answer came out of `s5_query`, with the region ids that produced it. This
      is the default and it is what `segmap ask` runs.
  `ask()`        -- the model gets a digest and answers from it. For the
      interpretive questions no verb covers ("which of these look
      misclassified"). The model is reading a table here, so treat any
      arithmetic in the answer as an estimate.

The class legend and priors go in the system prompt behind a cache breakpoint,
so iterating on questions against the same tile reads the legend from cache
instead of re-billing it each time. Tool definitions render *before* the system
prompt, so they sit inside the cached prefix too -- which is why `tools.SPECS`
is a fixed tuple and not built per question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL = "claude-opus-5"

SYSTEM_ROLE = """\
You are a terrain analyst reasoning over the output of a 47-class semantic \
segmentation model ("Smart Terrain") applied to nadir aerial imagery of \
Mediterranean/Levantine terrain. You do not see the imagery -- you see a \
symbolic digest of the label raster.

How to work:
- The segmentation is the perception authority. You supply the world knowledge \
it lacks: the class names carry geological and ecological meaning the model \
never used, because it learned each class as an arbitrary integer.
- Anything numeric -- areas, counts, percentages, distances -- comes from the \
digest. Do not estimate quantities that are not in it; say what is missing.
- When you flag a suspected misclassification, name the specific evidence \
(neighbouring classes, slope, aspect coherence, geometry) and say what \
ancillary data would settle it.
- Confusions inside an ordinal series (DryGrassland/Batha/Garigue/Maquis, or \
the road grades) are minor. Confusions across superclasses are not. \
Limestone/Dolomite/Nari are not separable in RGB at all -- do not present a \
guess between them as a finding.
"""


TOOL_ROLE = """\
You have tools that query the map. They are the only source of numbers here.

- Never state a count, area, distance, or fraction that did not come back from a \
tool call in this conversation. If you want a number, call for it.
- Quote the region ids a tool returned when you make a claim about specific \
places. The user checks those ids on the map; an answer without them cannot be \
audited.
- `status: unanswerable` means the label raster cannot answer that question at \
all -- say so plainly and say why. Do not substitute a near-miss class, and do \
not report zero. `status: empty` is the opposite: a real, measured zero, and a \
legitimate answer.
- The taxonomy has one vehicle class (Car) and one building class (House). \
Vehicle type, building function, fences, and anything about colour or texture \
are not in the map and never will be -- no tool call will get at them.
- Say which thresholds an answer rests on when a tool reports them (corridor \
inherits every guessed trafficability score in S4).
- Start broad with `describe` when the question is broad; descend to `find` \
when it is about specific places.
"""


def client():
    """An Anthropic client, or an actionable exit.

    The SDK raises a bare `TypeError` with a paragraph of header prose when no
    credentials are configured. Someone ten minutes into this repo reads that as
    a bug in the repo, and the two offline paths that would have answered their
    question go undiscovered.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised by test, not by CI
        raise SystemExit(
            "`segmap ask` needs the Anthropic SDK: pip install -e '.[llm]' "
            "(and export ANTHROPIC_API_KEY). The tool layer itself runs offline: "
            "`segmap tools` prints the definitions and `segmap solve s5 --query` "
            "runs the verbs."
        ) from exc
    c = anthropic.Anthropic()
    # Construction succeeds without credentials; the failure surfaces mid-request
    # as a TypeError about HTTP headers. Check here instead, before anything has
    # been built or billed.
    if not any(getattr(c, a, None) for a in ("api_key", "auth_token", "credentials")):
        raise SystemExit(
            "`segmap ask` found no API credentials: export ANTHROPIC_API_KEY and "
            "try again. Everything except the model call runs offline -- "
            "`segmap solve s5 --query '...'` computes the same numbers, "
            "`segmap tools` prints the surface the model is given, and "
            "`segmap report` writes the whole page."
        )
    return c


def build_system(legend: str, priors: str, extra: str = "") -> list[dict]:
    """One cached block. Everything in it is stable across questions about the
    same tile, which is the whole point of the breakpoint."""
    head = SYSTEM_ROLE + (f"\n\n## Using the tools\n{extra}" if extra else "")
    return [{
        "type": "text",
        "text": f"{head}\n\n## Class legend\n{legend}\n\n## Priors\n{priors}",
        "cache_control": {"type": "ephemeral"},
    }]


def priors_text() -> str:
    from .taxonomy import PRIORS

    lines = []
    for p in PRIORS:
        bits = [f"{p.subject} {p.kind}"]
        if p.others:
            bits.append("{" + ", ".join(p.others) + "}")
        if p.slope_deg:
            bits.append(f"slope {p.slope_deg[0]}-{p.slope_deg[1]} deg")
        if p.aspect_variance_max is not None:
            bits.append(f"aspect_circvar <= {p.aspect_variance_max}")
        lines.append(" ".join(bits) + f" -- {p.why}")
    return "\n".join(lines)


def ask(
    digest: str,
    question: str,
    *,
    legend: str,
    model: str = MODEL,
    effort: str = "high",
    max_tokens: int = 16000,
    show_thinking: bool = False,
) -> str:
    api = client()

    thinking: dict = {"type": "adaptive"}
    if show_thinking:
        thinking["display"] = "summarized"

    with api.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking=thinking,
        output_config={"effort": effort},
        system=build_system(legend, priors_text()),
        messages=[{
            "role": "user",
            "content": (
                f"## Digest of the segmentation map\n```\n{digest}\n```\n\n"
                f"## Question\n{question}"
            ),
        }],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        cat = getattr(message.stop_details, "category", None)
        return f"[refused by safety classifiers; category={cat}]"

    parts = []
    for block in message.content:
        if block.type == "thinking" and show_thinking and block.thinking:
            parts.append(f"[thinking]\n{block.thinking}\n")
        elif block.type == "text":
            parts.append(block.text)
    usage = message.usage
    parts.append(
        f"\n---\ntokens: in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}"
    )
    return "\n".join(parts)


# --- the tool path ---------------------------------------------------------

@dataclass
class ToolAnswer:
    text: str
    # Every tool payload, in the order the model asked for them. This is the
    # audit trail: each one carries the query string that produced it, so any
    # number in `text` can be reproduced with `segmap solve s5 --query '...'`.
    calls: list[dict] = field(default_factory=list)
    usage: str = ""

    def audit(self) -> str:
        lines = [f"# {len(self.calls)} tool call(s); every number above was "
                 f"computed by one of them"]
        for c in self.calls:
            lines.append(f"#   {c['status']:<12} {c.get('query', c['arguments'])}"
                         f"  ->  {c['summary']}")
        return "\n".join(lines)


def _runnable_tools(raster, ridx, log: list[dict]) -> list:
    """Wrap each ToolSpec in something the SDK's tool runner can execute.

    The schema comes from `tools.py`, not from these wrappers' signatures --
    one definition of the tool surface, and it stays unit-testable with no SDK
    installed.
    """
    from anthropic import beta_tool

    from . import tools

    def make(spec):
        # Nothing but **kwargs in the signature: the schema is passed in
        # explicitly, so there is nothing here for the SDK to infer from.
        def run(**kwargs):
            payload = tools.call(spec.name, kwargs, raster, ridx)
            log.append(payload)
            return tools.render(payload)

        run.__name__ = spec.name
        run.__doc__ = spec.description
        d = spec.definition()
        return beta_tool(name=d["name"], description=d["description"],
                         input_schema=d["input_schema"])(run)

    return [make(spec) for spec in tools.SPECS]


def ask_tools(
    question: str,
    raster,
    ridx,
    *,
    legend: str,
    model: str = MODEL,
    effort: str = "high",
    max_tokens: int = 16000,
    show_thinking: bool = False,
) -> ToolAnswer:
    """Answer a natural-language question by letting the model call the S5 verbs.

    The model plans; the code computes. It cannot return a count, area or
    distance that did not come out of `s5_query`, because it has no other way to
    see the raster.
    """
    from . import tools

    api = client()
    calls: list[dict] = []

    thinking: dict = {"type": "adaptive"}
    if show_thinking:
        thinking["display"] = "summarized"

    runner = api.beta.messages.tool_runner(
        model=model,
        max_tokens=max_tokens,
        thinking=thinking,
        output_config={"effort": effort},
        system=build_system(legend, priors_text(), extra=TOOL_ROLE),
        tools=_runnable_tools(raster, ridx, calls),
        messages=[{"role": "user", "content": question}],
        max_iterations=tools.MAX_ITERATIONS,
    )
    message = runner.until_done()

    if message.stop_reason == "refusal":
        cat = getattr(message.stop_details, "category", None)
        return ToolAnswer(f"[refused by safety classifiers; category={cat}]", calls)

    parts = []
    for block in message.content:
        if block.type == "thinking" and show_thinking and block.thinking:
            parts.append(f"[thinking]\n{block.thinking}\n")
        elif block.type == "text":
            parts.append(block.text)
    u = message.usage
    return ToolAnswer(
        "\n".join(parts).strip(), calls,
        # Final turn only -- the runner's earlier turns are billed too.
        f"tokens (final turn): in={u.input_tokens} out={u.output_tokens} "
        f"cache_read={getattr(u, 'cache_read_input_tokens', 0)}",
    )


def count_tokens(text: str, model: str = MODEL) -> int:
    """Exact token count from the API. Never use tiktoken for Claude."""
    return client().messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens
