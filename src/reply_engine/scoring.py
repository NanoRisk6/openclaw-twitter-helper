from __future__ import annotations

import math
import time
from typing import Iterable, List

from .models import Candidate, ScoredCandidate


def _freshness_score(created_utc: int) -> float:
    age_hours = max((time.time() - created_utc) / 3600, 0)
    return math.exp(-age_hours / 24)


def _keyword_match_score(text: str, keywords: Iterable[str]) -> float:
    text_l = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_l)
    return min(hits / max(len(list(keywords)) or 1, 1), 1.0)


def _engagement_score(c: Candidate) -> float:
    vals = c.engagement.values()
    total = sum(float(v) for v in vals if isinstance(v, (int, float)))
    return min(math.log1p(max(total, 0)) / 8, 1.0)


def score_candidate(c: Candidate, keywords: List[str]) -> ScoredCandidate:
    freshness = _freshness_score(c.created_utc)
    keyword = _keyword_match_score(c.text, keywords)
    engagement = _engagement_score(c)

    question_bonus = 0.1 if "?" in c.text else 0.0
    length_penalty = 0.1 if len(c.text) < 20 else 0.0

    score = (
        0.45 * freshness
        + 0.25 * keyword
        + 0.25 * engagement
        + question_bonus
        - length_penalty
    )
    score = max(min(score, 1.0), 0.0)

    reasons = [
        f"freshness={freshness:.2f}",
        f"keyword_match={keyword:.2f}",
        f"engagement={engagement:.2f}",
    ]
    if question_bonus:
        reasons.append("has_question")
    if length_penalty:
        reasons.append("very_short_text")

    return ScoredCandidate(candidate=c, score=score, reasons=reasons)


def rank_candidates(
    candidates: List[Candidate],
    keywords: List[str],
    include_weak: bool = False,
) -> List[ScoredCandidate]:
    scored = []
    for c in candidates:
        k = _keyword_match_score(c.text, keywords)
        if not include_weak and k <= 0:
            continue
        scored.append(score_candidate(c, keywords))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored
