from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Candidate:
    source: str
    author: str
    text: str
    url: str
    created_utc: int
    engagement: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score,
            "reasons": self.reasons,
        }
