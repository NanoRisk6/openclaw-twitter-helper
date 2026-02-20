from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Candidate

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


def _is_relevant(text: str, keyword: str) -> bool:
    text_l = text.lower()
    tokens = [t for t in keyword.lower().split() if t and t not in STOPWORDS and len(t) > 2]
    if not tokens:
        return keyword.lower() in text_l
    hits = sum(1 for tok in tokens if tok in text_l)
    if len(tokens) == 1:
        return hits == 1
    if len(tokens) == 2:
        return hits >= 1
    return hits >= 2


def _http_get_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "openclaw-reply-helper/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def discover_hn(keywords: Iterable[str], limit: int = 30) -> List[Candidate]:
    results: List[Candidate] = []
    for kw in keywords:
        query = urllib.parse.quote(kw)
        url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage={limit}"
        payload = _http_get_json(url)
        for hit in payload.get("hits", []):
            text = hit.get("title") or ""
            if not _is_relevant(text, kw):
                continue
            story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            results.append(
                Candidate(
                    source="hackernews",
                    author=hit.get("author", "unknown"),
                    text=text,
                    url=story_url,
                    created_utc=int(hit.get("created_at_i") or time.time()),
                    engagement={
                        "points": float(hit.get("points") or 0),
                        "comments": float(hit.get("num_comments") or 0),
                    },
                    metadata={"keyword": kw},
                )
            )
    return results


def discover_reddit(keywords: Iterable[str], limit: int = 30) -> List[Candidate]:
    results: List[Candidate] = []
    for kw in keywords:
        query = urllib.parse.quote(kw)
        url = (
            "https://www.reddit.com/search.json"
            f"?q={query}&sort=new&t=week&limit={limit}"
        )
        payload = _http_get_json(url)
        children = payload.get("data", {}).get("children", [])
        for child in children:
            data = child.get("data", {})
            text = data.get("title") or ""
            if not _is_relevant(text, kw):
                continue
            permalink = data.get("permalink") or ""
            post_url = f"https://reddit.com{permalink}" if permalink else data.get("url", "")
            results.append(
                Candidate(
                    source="reddit",
                    author=data.get("author", "unknown"),
                    text=text,
                    url=post_url,
                    created_utc=int(data.get("created_utc") or time.time()),
                    engagement={
                        "score": float(data.get("score") or 0),
                        "comments": float(data.get("num_comments") or 0),
                    },
                    metadata={
                        "subreddit": data.get("subreddit"),
                        "keyword": kw,
                    },
                )
            )
    return results


def discover_local(path: Optional[str]) -> List[Candidate]:
    if not path:
        return []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"local input not found: {path}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("local input must be a JSON list")

    candidates: List[Candidate] = []
    for row in raw:
        candidates.append(
            Candidate(
                source=row.get("source", "local"),
                author=row.get("author", "unknown"),
                text=row.get("text", ""),
                url=row.get("url", ""),
                created_utc=int(row.get("created_utc") or time.time()),
                engagement=row.get("engagement") or {},
                metadata=row.get("metadata") or {},
            )
        )
    return candidates


def dedupe(candidates: List[Candidate]) -> List[Candidate]:
    seen = set()
    out: List[Candidate] = []
    for c in candidates:
        key = (c.source, c.url, c.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def discover_all(
    keywords: Iterable[str],
    limit: int = 30,
    local_input: Optional[str] = None,
) -> List[Candidate]:
    combined = []
    combined.extend(discover_hn(keywords, limit=limit))
    combined.extend(discover_reddit(keywords, limit=limit))
    combined.extend(discover_local(local_input))
    return dedupe(combined)
