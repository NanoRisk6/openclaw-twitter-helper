from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .discovery import discover_all
from .generation import generate_markdown
from .models import Candidate, ScoredCandidate
from .scoring import rank_candidates


def run_discovery(keywords: List[str], limit: int, local_input: Optional[str]) -> List[Candidate]:
    return discover_all(keywords, limit=limit, local_input=local_input)


def save_candidates(candidates: List[Candidate], path: str) -> None:
    Path(path).write_text(
        json.dumps([c.to_dict() for c in candidates], indent=2),
        encoding="utf-8",
    )


def load_candidates(path: str) -> List[Candidate]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Candidate(**row) for row in raw]


def save_scored(scored: List[ScoredCandidate], path: str) -> None:
    Path(path).write_text(
        json.dumps([s.to_dict() for s in scored], indent=2),
        encoding="utf-8",
    )


def load_scored(path: str) -> List[ScoredCandidate]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scored: List[ScoredCandidate] = []
    for row in raw:
        scored.append(
            ScoredCandidate(
                candidate=Candidate(**row["candidate"]),
                score=float(row["score"]),
                reasons=list(row.get("reasons") or []),
            )
        )
    return scored


def run_rank(
    candidates: List[Candidate],
    keywords: List[str],
    include_weak: bool = False,
) -> List[ScoredCandidate]:
    return rank_candidates(candidates, keywords, include_weak=include_weak)


def run_ideas(scored: List[ScoredCandidate], top: int, out_path: str) -> None:
    md = generate_markdown(scored, top=top)
    Path(out_path).write_text(md, encoding="utf-8")
