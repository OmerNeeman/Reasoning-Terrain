"""Optional: send a digest + question to Claude and print the answer.

The point of the POC is the digest, not this call -- but being able to actually
ask a question of a given rung is how you find out which rung is enough.

The class legend and priors go in the system prompt behind a cache breakpoint,
so iterating on questions against the same tile reads the legend from cache
instead of re-billing it each time.
"""

from __future__ import annotations

MODEL = "claude-opus-5"

SYSTEM_ROLE = """\
You are a terrain analyst reasoning over the output of a 45-class semantic \
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


def build_system(legend: str, priors: str) -> list[dict]:
    return [{
        "type": "text",
        "text": f"{SYSTEM_ROLE}\n\n## Class legend\n{legend}\n\n## Priors\n{priors}",
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
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "the `ask` command needs the Anthropic SDK: pip install 'anthropic'"
        ) from exc

    client = anthropic.Anthropic()

    thinking: dict = {"type": "adaptive"}
    if show_thinking:
        thinking["display"] = "summarized"

    with client.messages.stream(
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


def count_tokens(text: str, model: str = MODEL) -> int:
    """Exact token count from the API. Never use tiktoken for Claude."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--count-tokens needs: pip install 'anthropic'") from exc
    client = anthropic.Anthropic()
    return client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens
