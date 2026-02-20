from __future__ import annotations

import json
import os
from typing import List

try:
    from openai import OpenAI
except Exception:  # optional dependency
    OpenAI = None

from .models import ScoredCandidate


def _template_ideas(scored: ScoredCandidate) -> List[str]:
    c = scored.candidate
    return [
        f"Strong agree + add one sharp datapoint. Mention: '{c.text[:90]}...' and add your POV in 1 sentence.",
        "Ask a probing follow-up question that opens discussion, then share a tiny actionable framework.",
        "Offer a contrarian but respectful angle, then end with a practical takeaway people can test today.",
    ]


def _llm_ideas(scored: ScoredCandidate, n: int = 3) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _template_ideas(scored)

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    c = scored.candidate
    prompt = (
        "You are generating short Twitter reply ideas for an AI brand account. "
        "Return JSON array of concise reply drafts. Each draft must be <= 240 chars, specific, non-generic, and useful. "
        f"Generate {n} ideas for this target post:\n"
        f"Source: {c.source}\n"
        f"Text: {c.text}\n"
        f"Context score: {scored.score:.2f}\n"
    )

    res = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.7,
    )

    text = res.output_text.strip()
    try:
        ideas = json.loads(text)
        if isinstance(ideas, list) and all(isinstance(x, str) for x in ideas):
            return ideas[:n]
    except json.JSONDecodeError:
        pass

    lines = [ln.strip("- ") for ln in text.splitlines() if ln.strip()]
    return lines[:n] if lines else _template_ideas(scored)


def generate_markdown(scored_candidates: List[ScoredCandidate], top: int = 20) -> str:
    selected = scored_candidates[:top]
    out: List[str] = ["# Reply Ideas\n"]

    for idx, scored in enumerate(selected, start=1):
        c = scored.candidate
        out.append(f"## {idx}. [{c.source}] {c.text[:100]}")
        out.append(f"- Author: @{c.author}")
        out.append(f"- URL: {c.url}")
        out.append(f"- Score: {scored.score:.2f}")
        out.append(f"- Why: {', '.join(scored.reasons)}")
        out.append("- Reply ideas:")
        for idea in _llm_ideas(scored, n=3):
            out.append(f"  - {idea}")
        out.append("")

    return "\n".join(out)
