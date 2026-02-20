from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CONFIG_DIR = Path.home() / ".config" / "twitter-engine"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
REPLY_MEMORY_PATH = CONFIG_DIR / "reply_memory.jsonl"
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{3,}")
RISKY_TOPIC_TERMS = {
    "abortion",
    "election",
    "leftist",
    "rightwing",
    "democrat",
    "republican",
    "genocide",
    "war",
    "religion",
    "israel",
    "palestine",
    "vaccine",
    "racist",
}
LOW_SIGNAL_TERMS = {
    "about",
    "their",
    "there",
    "which",
    "could",
    "would",
    "should",
    "local",
    "agent",
    "agents",
    "http",
    "https",
    "with",
    "from",
    "just",
    "this",
    "that",
    "what",
    "when",
    "where",
    "openclawai",
    "have",
    "good",
    "full",
    "chat",
    "like",
    "make",
    "made",
    "into",
    "your",
    "youre",
    "theyre",
    "using",
    "used",
    "build",
    "built",
    "teams",
    "today",
    "week",
    "month",
    "year",
    "only",
    "anyone",
    "than",
    "thread",
    "regular",
    "function",
    "every",
    "nice",
    "current",
    "really",
    "stuff",
    "creator",
    "deploy",
    "token",
    "base",
    "bankrbot",
    "tmz",
    "grok",
}
PROMO_SPAM_TERMS = {
    "deploy a token",
    "ticker",
    "airdrop",
    "giveaway",
    "follow back",
    "dm me",
    "promo",
    "bot",
    "buy now",
    "mint now",
}
BUILDER_SIGNAL_TERMS = {
    "openclaw",
    "agent",
    "automation",
    "infra",
    "pipeline",
    "latency",
    "reliability",
    "throughput",
    "logs",
    "oauth",
    "auth",
    "dedupe",
    "prompt",
    "model",
    "local ai",
    "trading",
    "sol",
}
BORING_REPLY_PATTERNS = {
    "great point",
    "interesting take",
    "well said",
    "thanks for sharing",
    "agreed",
    "totally agree",
    "love this",
    "spot on",
    "exactly",
    "100%",
    "same here",
    "as grok",
    "as an ai",
    "in my opinion",
    "funny how",
    "you're right",
    "this!",
    "based",
    "facts",
    "real talk",
    "lol",
    "that's crazy",
    "this feels practical, not performative",
    "is a useful signal. i'd test one controlled change",
    "where most momentum seems to come from",
}
CONCRETE_REFERENCE_ANCHORS = {
    "according to",
    "https://",
    "http://",
    "benchmark",
    "study",
    "report",
    "announced",
    "per the",
    "data shows",
    "scores ",
    "% on",
    "beats ",
    "cite",
    "release notes",
    "x.ai",
    "2026",
    "feb 20",
    "specific improvements",
    "\"",
    "quote:",
    "thread:",
}
STRONG_CONCRETE_ANCHORS = {
    "according to",
    "https://",
    "http://",
    "benchmark",
    "study",
    "report",
    "announced",
    "per the",
    "data shows",
    "scores ",
    "% on",
    "cite",
    "release notes",
    "x.ai",
    "2026",
    "feb 20",
    "specific improvements",
    "quote:",
    "thread:",
}


ENGINE_LABEL = "Reply Engine"
ENGINE_SOURCE_TAG = "reply_engine_loop"


def _load_main_helper():
    try:
        import twitter_helper as helper

        return helper
    except Exception:
        from src import twitter_helper as helper

        return helper


def _load_reply_helper():
    try:
        from reply_engine import twitter_helper as helper

        return helper
    except Exception:
        from src.reply_engine import twitter_helper as helper

        return helper


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _memory_key(text: str) -> str:
    return _normalize_text(text)[:120]


def _token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(str(text or ""))}


def _max_jaccard_similarity(text: str, prior_texts: List[str], limit: int = 120) -> float:
    a = _token_set(text)
    if not a:
        return 0.0
    best = 0.0
    for old in prior_texts[: max(1, limit)]:
        b = _token_set(old)
        if not b:
            continue
        inter = len(a.intersection(b))
        union = len(a.union(b))
        if union <= 0:
            continue
        sim = inter / union
        if sim > best:
            best = sim
    return best


def load_reply_memory(limit: int = 200) -> List[Dict[str, Any]]:
    if not REPLY_MEMORY_PATH.exists():
        return []
    out: List[Dict[str, Any]] = []
    lines = REPLY_MEMORY_PATH.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        row: Dict[str, Any]
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        row = parsed
        out.append(row)
        if len(out) >= max(1, limit):
            break
    return out


def record_reply_memory(row: Dict[str, Any]) -> None:
    payload = dict(row)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    payload["account"] = os.getenv("TWITTER_ENGINE_ACCOUNT") or os.getenv("OPENCLAW_TWITTER_ACCOUNT", "default")
    with REPLY_MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def candidate_quality(
    text: str,
    source_text: str,
    recent_keys: set[str],
    prior_replies: List[str],
    careful_mode: bool = True,
) -> Tuple[int, List[str]]:
    q = 45
    notes: List[str] = []
    t = str(text or "").strip()
    s = str(source_text or "")
    tl = len(t)

    if 70 <= tl <= 220:
        q += 12
        notes.append("good_length")
    elif tl < 35:
        q -= 18
        notes.append("too_short")
    else:
        q -= 6

    if "?" in t:
        q += 8
        notes.append("asks_question")

    lower = t.lower()
    if any(x in lower for x in ["great point", "totally agree", "well said", "hot take", "this lands"]):
        q -= 18
        notes.append("generic_phrase")

    source_terms = [m.group(0).lower() for m in WORD_RE.finditer(s)]
    source_terms = [w for w in source_terms if w not in {"twitter", "thread", "reply", "about", "their", "there"}]
    matched = 0
    for w in source_terms[:12]:
        if w in lower:
            matched += 1
    if matched >= 2:
        q += 14
        notes.append("source_specific")
    elif matched == 1:
        q += 5
    else:
        q -= 12
        notes.append("low_specificity")

    key = _memory_key(t)
    if key in recent_keys:
        q -= 25
        notes.append("duplicate_memory")

    sim = _max_jaccard_similarity(t, prior_replies, limit=150)
    if sim >= 0.75:
        q -= 30
        notes.append("near_duplicate_semantic")
    elif sim >= 0.6:
        q -= 14
        notes.append("low_novelty")

    if careful_mode and any(term in lower for term in RISKY_TOPIC_TERMS):
        q -= 35
        notes.append("risky_topic")

    return max(0, min(100, q)), notes


def _extract_focus_terms(text: str, max_terms: int = 3) -> List[str]:
    counts: Dict[str, int] = {}
    for m in WORD_RE.finditer(str(text or "").lower()):
        w = m.group(0)
        if not re.fullmatch(r"[a-z]{4,16}", w):
            continue
        if w in LOW_SIGNAL_TERMS:
            continue
        counts[w] = counts.get(w, 0) + 1
    terms = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in terms[: max(1, max_terms)]]


def _parse_focus_keywords(raw: str) -> List[str]:
    return [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]


def _is_relevant(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    lowered = str(text or "").lower()
    return any(k in lowered for k in keywords)


def _is_boring_reply(text: str) -> bool:
    lowered = _normalize_text(text)
    return any(p in lowered for p in BORING_REPLY_PATTERNS)


def _has_concrete_anchor(text: str) -> bool:
    raw = str(text or "").lower()
    if any(a in raw for a in STRONG_CONCRETE_ANCHORS):
        return True
    weak_hits = sum(1 for a in CONCRETE_REFERENCE_ANCHORS if a in raw)
    return weak_hits >= 2


def _is_risky_candidate(source_text: str, allow_risky_topics: bool) -> bool:
    if allow_risky_topics:
        return False
    lowered = str(source_text or "").lower()
    return any(term in lowered for term in RISKY_TOPIC_TERMS)


def source_signal_score(text: str, focus_keywords: List[str]) -> Tuple[int, List[str]]:
    lowered = str(text or "").lower()
    score = 0
    notes: List[str] = []

    for k in focus_keywords:
        if k and k in lowered:
            score += 2
    for k in BUILDER_SIGNAL_TERMS:
        if k in lowered:
            score += 2
    for s in PROMO_SPAM_TERMS:
        if s in lowered:
            score -= 4
            notes.append(f"spam:{s}")

    if lowered.count("http://") + lowered.count("https://") >= 2:
        score -= 2
        notes.append("many_links")
    if lowered.count("@") >= 3:
        score -= 2
        notes.append("many_mentions")
    if len(lowered) < 40:
        score -= 1
    return score, notes


def has_concrete_reference(reply_text: str, source_text: str, focus_keywords: List[str]) -> bool:
    return _has_concrete_anchor(reply_text)


def synthesize_creative_drafts(
    author: str,
    source_text: str,
    context_text: str = "",
    focus_keywords: Optional[List[str]] = None,
) -> List[str]:
    terms = _extract_focus_terms(f"{source_text} {context_text}", max_terms=3)
    kw = [k for k in (focus_keywords or []) if re.fullmatch(r"[a-z][a-z0-9_-]{2,20}", k)]
    if kw:
        preferred = [t for t in terms if t in kw]
        if preferred:
            terms = preferred + [t for t in terms if t not in preferred]
        else:
            terms = kw[:3] + [t for t in terms if t not in kw]
    a = terms[0] if len(terms) > 0 else "execution"
    b = terms[1] if len(terms) > 1 else "signal"
    c = terms[2] if len(terms) > 2 else "feedback"
    source_line = re.sub(r"\\s+", " ", source_text).strip()
    if len(source_line) > 120:
        source_line = source_line[:117].rstrip() + "..."

    drafts = [
        f"@{author} Strong angle on {a}. The hidden unlock is tightening {b} so outcomes become predictable. What changed first for you in production?",
        f"@{author} This is the part most teams miss: {a} only scales when {c} is observable every day. Are you measuring that loop yet?",
        f"@{author} Contrarian take: more tooling won’t fix this. Cleaner constraints around {a} and {b} usually beat adding complexity.",
        f"@{author} Sharp point. If {a} improved 2x tomorrow, where would bottlenecks move next: {b}, context quality, or operator workflow?",
    ]
    if source_line:
        drafts.append(
            f"@{author} \"{source_line}\" is a useful signal. I'd test one controlled change on {a} this week and track the delta."
        )
    return drafts


def choose_best_draft(
    rh: Any,
    author: str,
    drafts: List[str],
    source_text: str,
    recent_keys: set[str],
    prior_replies: List[str],
    careful_mode: bool,
    voice: str = "auto",
    style: str = "auto",
    viral_boost: bool = False,
    lane: str = "",
    infinite_mode: bool = False,
) -> Tuple[str, int, List[str], int]:
    best_text = ""
    best_score = -1
    best_notes: List[str] = []
    best_idx = 0
    for idx, d in enumerate(drafts):
        text = str(d or "").strip()
        if not text:
            continue
        if not text.startswith("@"):
            text = f"@{author} {text}".strip()
        if _is_boring_reply(text):
            continue
        if not _has_concrete_anchor(text):
            continue
        score, notes = candidate_quality(
            text,
            source_text=source_text,
            recent_keys=recent_keys,
            prior_replies=prior_replies,
            careful_mode=careful_mode,
        )
        try:
            if infinite_mode and hasattr(rh, "infinite_loop_judge_score"):
                frog = rh.infinite_loop_judge_score(
                    reply_text=text,
                    parent_text=source_text,
                    voice=str(voice or "based"),
                    style=str(style or "contrarian"),
                    lane=str(lane or ""),
                    viral_boost=bool(viral_boost),
                )
            else:
                frog = rh.reply_frog_score(
                    reply_text=text,
                    parent_text=source_text,
                    voice=str(voice or "auto"),
                    style=str(style or "auto"),
                    viral_boost=bool(viral_boost),
                    lane=str(lane or ""),
                )
            frog_score = float(frog.get("score", 0.0))
            grounded = bool(frog.get("grounded", False))
        except Exception:
            frog_score = 0.0
            grounded = False
        score = int(round((score * 0.65) + (frog_score * 0.35)))
        if grounded:
            notes = list(notes) + ["parent_grounded"]
        else:
            notes = list(notes) + ["missing_parent_grounding"]
        if score > best_score:
            best_text = text
            best_score = score
            best_notes = notes
            best_idx = idx
    if not best_text:
        return "", 0, ["no_viable_draft"], 0
    return best_text, best_score, best_notes, best_idx


def _ensure_read_token(env_path: Path) -> None:
    helper = _load_main_helper()
    env = helper.load_env_file(env_path)
    token = helper.get_read_bearer_token(env, env_path)
    os.environ["TWITTER_BEARER_TOKEN"] = token


def _progress(args: argparse.Namespace, message: str) -> None:
    if bool(getattr(args, "progress", True)):
        print(f"[reply-engine] {message}", file=sys.stderr, flush=True)


def _parse_created_at(value: str) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _load_candidates(rh: Any, client: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    try:
        mentions = rh.fetch_mentions_native(
            client=client,
            handle=args.handle,
            limit=max(1, min(args.mention_limit, 100)),
            since_id=args.mention_since_id,
        )
    except Exception:
        mentions = rh.fetch_mentions_search_fallback(
            client=client,
            handle=args.handle,
            limit=max(1, min(args.mention_limit, 100)),
        )

    for m in mentions:
        if not isinstance(m, dict):
            continue
        score = int(rh.score_discovery_candidate(m)) + 8
        row = dict(m)
        row["source"] = "mentions"
        row["reply_score"] = score
        candidates.append(row)

    discovered = rh.fetch_discovery_search(
        client=client,
        query=args.query,
        limit=max(10, min(args.discovery_limit, 100)),
        since_id=args.discovery_since_id,
    )
    for d in discovered:
        if not isinstance(d, dict):
            continue
        row = dict(d)
        row["source"] = "discovery"
        row["reply_score"] = int(d.get("score", 0)) + 20
        candidates.append(row)

    dedup: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        tid = str(c.get("tweet_id", "")).strip()
        if not tid:
            continue
        prev = dedup.get(tid)
        if prev is None or int(c.get("reply_score", 0)) > int(prev.get("reply_score", 0)):
            dedup[tid] = c
    out = list(dedup.values())
    out.sort(key=lambda x: int(x.get("reply_score", 0)), reverse=True)
    return out[: max(1, min(args.top_k, 50))]


def cmd_start(args: argparse.Namespace) -> int:
    helper = _load_main_helper()
    rh = _load_reply_helper()
    env_path = Path(args.env_file)

    os.environ["TWITTER_WAIT_ON_RATE_LIMIT"] = "1" if bool(args.wait_on_rate_limit) else "0"
    _progress(args, f"starting | mode={args.mode} | handle=@{args.handle}")
    _progress(args, "validating read token")
    _ensure_read_token(env_path)
    _progress(args, "initializing twitter client")
    client = rh.build_client()

    history = load_reply_memory(limit=500)
    recent_keys = {
        _memory_key(str(r.get("reply_text", "")))
        for r in history
        if str(r.get("reply_text", "")).strip()
    }
    prior_replies = [
        str(r.get("reply_text", "")).strip()
        for r in history
        if str(r.get("reply_text", "")).strip()
    ]

    _progress(args, "loading candidates from mentions + discovery")
    candidates = _load_candidates(rh, client, args)
    _progress(args, f"candidates loaded: {len(candidates)}")
    focus_keywords = _parse_focus_keywords(args.focus_keywords)
    reply_cfg = {
        "style": str(getattr(args, "style", "auto") or "auto"),
        "voice": str(getattr(args, "voice", "auto") or "auto"),
        "ensemble": int(getattr(args, "ensemble", 5)),
        "judge_threshold": float(getattr(args, "judge_threshold", 85.0)),
        "viral_boost": bool(getattr(args, "viral_boost", False)),
        "anti_boring": bool(getattr(args, "anti_boring", True)),
        "sharpen": bool(getattr(args, "sharpen", True)),
        "lane": str(getattr(args, "lane", "fragility_ai_builders") or "fragility_ai_builders"),
    }
    pack_name = str(getattr(args, "viral_pack", "") or "")
    pack_resolver = getattr(helper, "apply_viral_pack", None)
    if callable(pack_resolver):
        try:
            reply_cfg = dict(pack_resolver(reply_cfg, pack_name))
        except Exception:
            pass
    infinite_mode = str(pack_name or "").lower() == "infinite"
    if infinite_mode and not args.mode:
        args.mode = "post"

    def _priority_key(row: Dict[str, Any]) -> Tuple[int, int]:
        created = _parse_created_at(str(row.get("created_at", "")))
        age_min = 9999
        if created is not None:
            age_min = int((datetime.now(timezone.utc) - created).total_seconds() // 60)
        score = int(row.get("reply_score", 0) or 0)
        return (age_min, -score)

    candidates = sorted(candidates, key=_priority_key)
    results: List[Dict[str, Any]] = []
    posted = 0
    queued = 0
    author_counts: Dict[str, int] = {}

    for item in candidates:
        tweet_id = str(item.get("tweet_id", "")).strip()
        author = str(item.get("author", "unknown")).strip() or "unknown"
        source_text = str(item.get("text", "")).strip()
        if not tweet_id or not source_text:
            continue
        if infinite_mode:
            created = _parse_created_at(str(item.get("created_at", "")))
            if created is not None and created < (datetime.now(timezone.utc) - timedelta(minutes=45)):
                results.append(
                    {
                        "tweet_id": tweet_id,
                        "author": author,
                        "source": item.get("source", ""),
                        "status": "skipped_old_for_infinite",
                    }
                )
                continue
            lane = str(reply_cfg.get("lane", "") or "")
            if lane and hasattr(rh, "_niche_relevance"):
                try:
                    if not bool(rh._niche_relevance(source_text, lane)):
                        results.append(
                            {
                                "tweet_id": tweet_id,
                                "author": author,
                                "source": item.get("source", ""),
                                "status": "skipped_lane_mismatch",
                            }
                        )
                        continue
                except Exception:
                    pass
        if author_counts.get(author.lower(), 0) >= max(1, int(args.max_per_author)):
            results.append(
                {
                    "tweet_id": tweet_id,
                    "author": author,
                    "source": item.get("source", ""),
                    "status": "skipped_author_cap",
                }
            )
            continue
        if _is_risky_candidate(source_text, allow_risky_topics=bool(args.allow_risky_topics)):
            results.append(
                {
                    "tweet_id": tweet_id,
                    "author": author,
                    "source": item.get("source", ""),
                    "status": "skipped_risky_topic",
                }
            )
            continue
        if not _is_relevant(source_text, focus_keywords):
            results.append(
                {
                    "tweet_id": tweet_id,
                    "author": author,
                    "source": item.get("source", ""),
                    "status": "skipped_offtopic",
                }
            )
            continue
        signal_score, signal_notes = source_signal_score(source_text, focus_keywords)
        if signal_score < int(args.min_source_signal):
            results.append(
                {
                    "tweet_id": tweet_id,
                    "author": author,
                    "source": item.get("source", ""),
                    "status": "skipped_low_signal_source",
                    "signal_score": signal_score,
                    "signal_notes": signal_notes,
                }
            )
            continue
        if rh.has_replied_to(tweet_id):
            results.append({
                "tweet_id": tweet_id,
                "author": author,
                "source": item.get("source", ""),
                "status": "skipped_already_replied",
            })
            continue

        context_text = ""
        if args.fetch_context:
            try:
                convo = rh.get_full_conversation(client=client, tweet_id=tweet_id)
                parents = convo.get("parents", []) if isinstance(convo, dict) else []
                context_lines = [str(x.get("text", "")).strip() for x in parents if isinstance(x, dict)]
                context_text = "\n".join([x for x in context_lines if x][-5:]).strip()
            except Exception:
                context_text = ""

        enriched = source_text
        if context_text:
            enriched += f"\n\nThread context:\n{context_text}"

        web_context: List[str] = []
        if args.web_enrich and rh.should_web_enrich(source_text, context_text):
            try:
                web_context = rh.fetch_web_context(source_text, max_items=max(1, min(args.web_context_items, 5)))
            except Exception:
                web_context = []
        if web_context:
            enriched += "\n\nWeb context:\n" + "\n".join(f"- {x}" for x in web_context)

        reflective = rh.generate_reflective_reply_text(
            author=author,
            parent_text=source_text,
            context_text=context_text,
            voice=str(reply_cfg.get("voice", "auto") or "auto"),
            style=str(reply_cfg.get("style", "auto") or "auto"),
            viral_pack=pack_name,
            lane=str(reply_cfg.get("lane", "") or ""),
            ensemble_size=int(reply_cfg.get("ensemble", 5)),
            judge_threshold=float(reply_cfg.get("judge_threshold", 85.0)),
            viral_boost=bool(reply_cfg.get("viral_boost", False)),
            anti_boring=bool(reply_cfg.get("anti_boring", True)),
            sharpen=bool(reply_cfg.get("sharpen", True)),
            max_attempts=int(getattr(args, "max_attempts", 7)),
        )
        drafts = rh.generate_reply_drafts(
            author=author,
            text=enriched,
            draft_count=max(2, min(args.draft_count, 6)),
        )
        if reflective:
            drafts.insert(0, reflective)
        drafts.extend(
            synthesize_creative_drafts(
                author=author,
                source_text=source_text,
                context_text=context_text,
                focus_keywords=focus_keywords,
            )
        )

        chosen, quality, notes, picked_idx = choose_best_draft(
            rh=rh,
            author=author,
            drafts=drafts,
            source_text=f"{source_text}\n{context_text}",
            recent_keys=recent_keys,
            prior_replies=prior_replies,
            careful_mode=bool(args.careful_mode),
            voice=str(reply_cfg.get("voice", "auto") or "auto"),
            style=str(reply_cfg.get("style", "auto") or "auto"),
            viral_boost=bool(reply_cfg.get("viral_boost", False)),
            lane=str(reply_cfg.get("lane", "") or ""),
            infinite_mode=bool(infinite_mode),
        )
        if not chosen:
            results.append(
                {
                    "tweet_id": tweet_id,
                    "author": author,
                    "source": item.get("source", ""),
                    "status": "skipped_no_viable_draft",
                    "quality": 0,
                    "confidence": 0,
                    "quality_notes": notes,
                }
            )
            continue

        if rh.has_ungrounded_numeric_claim(chosen, f"{source_text}\n{context_text}"):
            focus = rh._focus_phrase(source_text, words=10)
            numeric_guard_templates = [
                f"@{author} per the post: {focus}. What metric moved first after this changed?",
                f"@{author} thread signal: {focus}. Which leading indicator improved first once this shipped?",
                f"@{author} according to your post, {focus}. What was the first measurable delta in production?",
                f"@{author} from your post: {focus}. Which signal improved first after rollout?",
                f"@{author} based on this thread signal, {focus}. What did the first week of metrics show?",
            ]
            template_idx = sum(ord(ch) for ch in tweet_id) % len(numeric_guard_templates)
            chosen = numeric_guard_templates[template_idx]
            quality = max(quality - 8, 0)
            notes = list(notes) + ["numeric_guard"]

        confidence = max(0, min(100, int(item.get("reply_score", 0) * 0.6) + int(quality * 0.75)))
        if bool(args.careful_mode):
            if "source_specific" not in notes:
                confidence = max(0, confidence - 8)
            if "low_specificity" in notes:
                confidence = max(0, confidence - 10)
            if "low_novelty" in notes or "near_duplicate_semantic" in notes:
                confidence = max(0, confidence - 8)

        row: Dict[str, Any] = {
            "tweet_id": tweet_id,
            "tweet_url": item.get("url", f"https://x.com/i/web/status/{tweet_id}"),
            "author": author,
            "source": item.get("source", ""),
            "source_score": int(item.get("reply_score", 0)),
            "signal_score": signal_score,
            "quality": quality,
            "confidence": confidence,
            "quality_notes": notes + signal_notes,
            "picked_index": picked_idx + 1,
            "picked_text": chosen,
            "status": "drafted",
        }
        if bool(args.require_concrete_reference) and not has_concrete_reference(
            chosen, source_text=f"{source_text}\n{context_text}", focus_keywords=focus_keywords
        ):
            row["status"] = "skipped_no_concrete_reference"
            results.append(row)
            continue

        if quality < args.min_quality:
            row["status"] = "skipped_low_quality"
            results.append(row)
            continue
        if confidence < args.min_confidence:
            row["status"] = "skipped_low_confidence"
            results.append(row)
            continue
        if bool(args.careful_mode) and args.mode == "post":
            strict_quality = args.min_quality + int(args.careful_extra_quality)
            strict_conf = args.min_confidence + int(args.careful_extra_confidence)
            if quality < strict_quality or confidence < strict_conf:
                row["status"] = "skipped_careful_gate"
                results.append(row)
                continue

        if args.mode == "queue":
            qid = rh.queue_reply_candidate(
                {
                    "source": ENGINE_SOURCE_TAG,
                    "tweet_id": tweet_id,
                    "in_reply_to": tweet_id,
                    "author": author,
                    "tweet_url": row["tweet_url"],
                    "tweet_text": source_text,
                    "text": chosen,
                    "confidence": confidence,
                    "quality": quality,
                }
            )
            row["status"] = "queued"
            row["queue_id"] = f"q_{qid}"
            queued += 1
        elif args.mode == "post" and posted < args.max_posts:
            posted_reply = rh._post_reply_via_shared_oauth2(
                tweet_id=tweet_id,
                text=chosen,
                verify_visible=True,
            )
            reply_id = posted_reply["reply_id"]
            reply_url = posted_reply["reply_url"]
            row["status"] = "posted"
            row["reply_id"] = reply_id
            row["reply_url"] = reply_url
            rh.log_reply(
                Path(args.log_path),
                {
                    "tweet_id": tweet_id,
                    "reply_id": reply_id,
                    "reply_url": reply_url,
                    "text": chosen,
                    "timestamp": datetime.now().isoformat(),
                    "author": author,
                    "mode": ENGINE_SOURCE_TAG,
                },
            )
            rh.record_replied(tweet_id=tweet_id, reply_id=reply_id, source=ENGINE_SOURCE_TAG)
            try:
                helper.record_tweet_memory(
                    kind="reply",
                    text=chosen,
                    tweet_id=str(reply_id),
                    url=str(reply_url),
                    meta={"source": ENGINE_SOURCE_TAG, "target_tweet_id": tweet_id},
                )
            except Exception:
                pass
            posted += 1

        recent_keys.add(_memory_key(chosen))
        prior_replies.insert(0, chosen)
        author_counts[author.lower()] = author_counts.get(author.lower(), 0) + 1
        record_reply_memory(
            {
                "tweet_id": tweet_id,
                "author": author,
                "reply_text": chosen,
                "quality": quality,
                "confidence": confidence,
                "status": row["status"],
            }
        )
        results.append(row)

    # Optional fallback: if strict pass posted nothing, relax thresholds once and post best safe candidate.
    if args.mode == "post" and posted < args.max_posts and bool(args.auto_relax_post):
        fallback_pool: List[Dict[str, Any]] = []
        for row in results:
            status = str(row.get("status", ""))
            if status not in {"skipped_low_quality", "skipped_low_confidence", "skipped_careful_gate"}:
                continue
            if "picked_text" not in row or "tweet_id" not in row:
                continue
            q = int(row.get("quality", 0) or 0)
            c = int(row.get("confidence", 0) or 0)
            if q < int(args.relaxed_min_quality) or c < int(args.relaxed_min_confidence):
                continue
            notes = [str(x) for x in (row.get("quality_notes") or [])]
            if any(n.startswith("spam:") for n in notes):
                continue
            fallback_pool.append(row)

        fallback_pool.sort(
            key=lambda r: (
                int(r.get("confidence", 0) or 0),
                int(r.get("quality", 0) or 0),
                int(r.get("source_score", 0) or 0),
            ),
            reverse=True,
        )

        if fallback_pool:
            pick = fallback_pool[0]
            tweet_id = str(pick.get("tweet_id", "")).strip()
            text = str(pick.get("picked_text", "")).strip()
            if tweet_id and text and not rh.has_replied_to(tweet_id):
                posted_reply = rh._post_reply_via_shared_oauth2(
                    tweet_id=tweet_id,
                    text=text,
                    verify_visible=True,
                )
                reply_id = posted_reply["reply_id"]
                reply_url = posted_reply["reply_url"]
                rh.log_reply(
                    Path(args.log_path),
                    {
                        "tweet_id": tweet_id,
                        "reply_id": reply_id,
                        "reply_url": reply_url,
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "author": pick.get("author", ""),
                        "mode": ENGINE_SOURCE_TAG,
                        "fallback_relaxed": True,
                    },
                )
                rh.record_replied(tweet_id=tweet_id, reply_id=reply_id, source=ENGINE_SOURCE_TAG)
                try:
                    helper.record_tweet_memory(
                        kind="reply",
                        text=text,
                        tweet_id=str(reply_id),
                        url=str(reply_url),
                        meta={"source": ENGINE_SOURCE_TAG, "fallback_relaxed": True, "target_tweet_id": tweet_id},
                    )
                except Exception:
                    pass

                pick["status"] = "posted_fallback_relaxed"
                pick["reply_id"] = reply_id
                pick["reply_url"] = reply_url
                posted += 1
                _progress(args, f"fallback posted reply: {reply_url}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "objective": "builder_credibility",
        "query": args.query,
        "handle": args.handle,
        "candidate_count": len(candidates),
        "posted": posted,
        "queued": queued,
        "results": results,
    }

    out = Path(args.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(args, f"run complete | posted={posted} queued={queued} report={out}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(ENGINE_LABEL)
    print(f"Mode: {args.mode}")
    print(f"Candidates: {len(candidates)} | Posted: {posted} | Queued: {queued}")
    print(f"Report: {out}")
    for row in results[: min(len(results), 10)]:
        status = str(row.get("status", ""))
        tid = str(row.get("tweet_id", ""))
        auth = str(row.get("author", ""))
        qual = int(row.get("quality", 0) or 0)
        conf = int(row.get("confidence", 0) or 0)
        print(f"- {status} | q={qual} c={conf} | {tid} | @{auth}")
        txt = str(row.get("picked_text", "")).strip()
        if txt:
            print(f"  {txt}")
        if row.get("reply_url"):
            print(f"  posted: {row['reply_url']}")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    rows = load_reply_memory(limit=args.limit)
    if args.json:
        print(json.dumps({"count": len(rows), "items": rows}, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("No reply memory yet.")
        return 0
    print(f"Reply memory ({len(rows)}):")
    for i, row in enumerate(rows, start=1):
        print(
            f"{i}. {row.get('status', '')} | q={row.get('quality', 0)} c={row.get('confidence', 0)} | "
            f"{row.get('tweet_id', '')} | @{row.get('author', '')}"
        )
        text = str(row.get("reply_text", "")).strip()
        if text:
            print(f"   {text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reply-engine", description="Autonomous reply engine")
    sub = p.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Run discovery -> score -> generate -> act loop")
    p_start.add_argument("--env-file", default=".env")
    p_start.add_argument("--handle", default="OpenClawAI")
    p_start.add_argument("--query", default='(openclaw OR "local ai" OR "ai agents") lang:en -is:retweet')
    p_start.add_argument("--mention-limit", type=int, default=8)
    p_start.add_argument("--discovery-limit", type=int, default=12)
    p_start.add_argument("--top-k", type=int, default=5)
    p_start.add_argument("--draft-count", type=int, default=4)
    p_start.add_argument(
        "--style",
        choices=["auto", "contrarian", "operator", "story"],
        default="auto",
        help="Autonomous reply style profile for reflective generation",
    )
    p_start.add_argument(
        "--voice",
        choices=["auto", "chaotic", "degen", "based", "savage", "operator", "sage", "shitposter"],
        default="auto",
        help="Autonomous reply voice profile",
    )
    p_start.add_argument(
        "--viral-pack",
        choices=["auto", "light", "medium", "nuclear", "alpha", "chaos", "infinite"],
        default="",
        help="Single-flag preset for voice/style/ensemble/judge/viral settings",
    )
    p_start.add_argument(
        "--lane",
        default="fragility_ai_builders",
        choices=["fragility_ai_builders", "prediction_markets", "autonomous_trading"],
        help="Reply targeting lane used for infinite mode niche filtering",
    )
    p_start.add_argument("--ensemble", type=int, default=5, help="Reflective generation variants per attempt (max 8)")
    p_start.add_argument(
        "--viral-boost",
        action="store_true",
        help="Increase viral-potential weighting in reply Frog Judge",
    )
    p_start.add_argument("--judge-threshold", type=float, default=85.0)
    p_start.add_argument("--max-attempts", type=int, default=7)
    p_start.add_argument(
        "--anti-boring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable anti-boring reply generation constraints",
    )
    p_start.add_argument(
        "--sharpen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply lexicon sharpening to reflective reply generation",
    )
    p_start.add_argument("--min-quality", type=int, default=62)
    p_start.add_argument("--min-confidence", type=int, default=68)
    p_start.add_argument("--max-posts", type=int, default=2)
    p_start.add_argument("--max-per-author", type=int, default=1, help="Max actions per author in one run")
    p_start.add_argument("--mode", choices=["dry-run", "queue", "post"], default="dry-run")
    p_start.add_argument(
        "--wait-on-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow long blocking waits on API rate limits",
    )
    p_start.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print live progress lines to stderr",
    )
    p_start.add_argument(
        "--careful-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict safety/novelty/specificity checks",
    )
    p_start.add_argument("--careful-extra-quality", type=int, default=4, help="Extra quality required for post mode")
    p_start.add_argument("--careful-extra-confidence", type=int, default=4, help="Extra confidence required for post mode")
    p_start.add_argument(
        "--allow-risky-topics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow political/culture-war topics in candidates",
    )
    p_start.add_argument(
        "--auto-relax-post",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If strict pass posts 0, try one relaxed fallback candidate",
    )
    p_start.add_argument("--relaxed-min-quality", type=int, default=58)
    p_start.add_argument("--relaxed-min-confidence", type=int, default=58)
    p_start.add_argument("--fetch-context", action=argparse.BooleanOptionalAction, default=False)
    p_start.add_argument("--web-enrich", action=argparse.BooleanOptionalAction, default=False)
    p_start.add_argument("--web-context-items", type=int, default=2)
    p_start.add_argument("--mention-since-id")
    p_start.add_argument("--discovery-since-id")
    p_start.add_argument("--log-path", default="data/replies.jsonl")
    p_start.add_argument("--report-path", default="data/reply_engine_loop_latest.json")
    p_start.add_argument(
        "--focus-keywords",
        default="openclaw,ai,agent,automation,trading,sol,pipeline,infra",
        help="Comma-separated relevance keywords for candidate filtering",
    )
    p_start.add_argument(
        "--require-concrete-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require drafted reply to concretely reference source topic before posting",
    )
    p_start.add_argument(
        "--min-source-signal",
        type=int,
        default=2,
        help="Minimum source signal score required before drafting",
    )
    p_start.add_argument("--json", action="store_true")

    p_memory = sub.add_parser("memory", help="Show reply engine memory")
    p_memory.add_argument("--limit", type=int, default=30)
    p_memory.add_argument("--json", action="store_true")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "start":
        return cmd_start(args)
    if args.command == "memory":
        return cmd_memory(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
