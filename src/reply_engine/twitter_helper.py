from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

try:
    from dotenv import load_dotenv
except Exception:  # optional dependency
    def load_dotenv() -> None:
        return None

try:
    import keyring
except Exception:  # optional dependency fallback
    keyring = None


TWEET_ID_RE = re.compile(r"(?:status/)?(\d{10,30})")
CONFIG_DIR = Path.home() / ".config" / "openclaw-twitter-helper"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
APPROVAL_DIR = CONFIG_DIR / "approval_queue"
APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = CONFIG_DIR / "reply_engine_runs.jsonl"
DEFAULT_REPLY_MODES = [
    "direct",
    "curious",
    "witty",
    "technical",
    "supportive",
    "question",
]
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "been",
    "being",
    "both",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "most",
    "only",
    "over",
    "really",
    "some",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "with",
    "your",
}
NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


def extract_tweet_id(tweet: str) -> str:
    m = TWEET_ID_RE.search(tweet)
    if not m:
        raise ValueError(f"could not extract tweet id from: {tweet}")
    return m.group(1)


def build_tweet_url(tweet_id: str, username: Optional[str] = None) -> str:
    clean_id = str(tweet_id).strip()
    if username:
        return f"https://x.com/{username}/status/{clean_id}"
    return f"https://x.com/i/web/status/{clean_id}"


def _required_env() -> Dict[str, Optional[str]]:
    load_dotenv()
    bearer = os.getenv("TWITTER_BEARER_TOKEN") or os.getenv("TWITTER_OAUTH2_ACCESS_TOKEN")
    if not bearer and keyring is not None:
        account = os.getenv("OPENCLAW_TWITTER_ACCOUNT", "default")
        service = f"openclaw-twitter-helper:{account}"
        try:
            raw = keyring.get_password(service, "oauth_tokens")
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    bearer = str(parsed.get("access_token", "")).strip() or bearer
        except Exception:
            pass
    return {"TWITTER_BEARER_TOKEN": bearer}


def _account_name() -> str:
    return os.getenv("OPENCLAW_TWITTER_ACCOUNT", "default")


def _last_mention_path(account: Optional[str] = None) -> Path:
    acct = account or _account_name()
    return CONFIG_DIR / f"last_mention_id_{acct}.txt"


def _replied_log_path(account: Optional[str] = None) -> Path:
    acct = account or _account_name()
    return CONFIG_DIR / f"replied_to_{acct}.jsonl"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any], ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=ensure_ascii) + "\n")


def _log_run_event(kind: str, payload: Dict[str, Any]) -> None:
    row = {
        "ts": datetime.now().isoformat(),
        "account": _account_name(),
        "kind": kind,
        "payload": payload,
    }
    _append_jsonl(RUN_LOG, row, ensure_ascii=False)


def load_last_mention_id(account: Optional[str] = None) -> Optional[str]:
    path = _last_mention_path(account)
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    return raw or None


def save_last_mention_id(tweet_id: str, account: Optional[str] = None) -> None:
    path = _last_mention_path(account)
    _atomic_write_text(path, str(tweet_id).strip())


def build_client() -> Any:
    try:
        import tweepy
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `tweepy`. Install reply-engine deps with: "
            "pip install -r requirements-reply-engine.txt"
        ) from exc

    env = _required_env()
    required = ["TWITTER_BEARER_TOKEN"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return tweepy.Client(
        bearer_token=env["TWITTER_BEARER_TOKEN"],
        wait_on_rate_limit=True,
    )


def _load_main_helper():
    try:
        import twitter_helper as main_helper  # installed script/module path
        return main_helper
    except Exception:
        pass
    try:
        from src import twitter_helper as main_helper  # repo-local module path
        return main_helper
    except Exception as exc:
        raise RuntimeError(
            "Could not import shared twitter_helper module for OAuth2 posting path."
        ) from exc


def _post_reply_via_shared_oauth2(
    tweet_id: str,
    text: str,
    verify_visible: bool = True,
) -> Dict[str, str]:
    main_helper = _load_main_helper()
    env_path = Path(os.getenv("TWITTER_HELPER_ENV_FILE", ".env"))
    cfg, env_values = main_helper.resolve_config(env_path)
    fresh, (status, body) = main_helper.post_with_retry(
        cfg=cfg,
        env_path=env_path,
        env_values=env_values,
        text=text,
        reply_to_id=tweet_id,
        unique_on_duplicate=True,
    )
    if status < 200 or status >= 300:
        raise RuntimeError(f"OAuth2 post failed ({status}): {json.dumps(body, ensure_ascii=False)}")
    data = body.get("data") if isinstance(body, dict) else None
    reply_id = str(data.get("id", "")) if isinstance(data, dict) else ""
    if not reply_id:
        raise RuntimeError("OAuth2 post succeeded but returned no reply id")
    if verify_visible:
        _, url = main_helper.verify_post_visible(fresh, reply_id)
    else:
        url = f"https://x.com/i/web/status/{reply_id}"
    return {"reply_id": reply_id, "reply_url": url}


def fetch_tweet_context(client: Any, tweet_id: str) -> Dict[str, Any]:
    res = client.get_tweet(
        id=tweet_id,
        expansions=["author_id"],
        tweet_fields=["created_at", "public_metrics", "text"],
        user_fields=["username", "name"],
    )
    if not res or not res.data:
        raise RuntimeError(f"tweet not found: {tweet_id}")

    user_by_id = {u.id: u for u in (res.includes.get("users", []) if res.includes else [])}
    author = user_by_id.get(res.data.author_id)
    public_metrics = getattr(res.data, "public_metrics", {}) or {}

    return {
        "tweet_id": tweet_id,
        "author": getattr(author, "username", "unknown"),
        "text": res.data.text,
        "metrics": public_metrics,
        "created_at": str(getattr(res.data, "created_at", "")),
    }


def _templates(author: str, text: str) -> List[str]:
    return [
        f"@{author} This lands. I keep noticing progress compounds when we stay with one honest signal long enough to learn from it.",
        f"@{author} I like this framing. It reminds me that clarity beats noise when building in public.",
        f"@{author} This feels practical, not performative. That difference is where most momentum seems to come from.",
        f"@{author} I’ve been reflecting on this too: consistency is less about repetition and more about direction.",
        f"@{author} This is a useful prompt. What changed for you once this clicked in real execution?",
    ]


def _openai_drafts(author: str, text: str, n: int) -> Optional[List[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)
    prompt = (
        "Generate concise but creative Twitter reply drafts for OpenClawAI. "
        "Return plain JSON array of strings only. "
        "Each draft must be <= 240 characters, specific to the tweet, and non-generic. "
        "Avoid repetitive boilerplate phrasing. "
        "You may include first-person reflective language when it feels natural and relevant. "
        "Do not invent facts, numbers, percentages, timelines, or claims that are not in the source tweet/context. "
        f"Create {n} replies to this tweet by @{author}: {text}"
    )

    res = client.responses.create(model=model, input=prompt, temperature=0.95)
    raw = res.output_text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            clean = [x for x in parsed if isinstance(x, str) and x.strip()]
            return clean[:n] if clean else None
    except json.JSONDecodeError:
        pass

    lines = [ln.strip("- ") for ln in raw.splitlines() if ln.strip()]
    return lines[:n] if lines else None


def _openai_mode_drafts(author: str, text: str, modes: List[str]) -> Optional[Dict[str, str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)
    modes_csv = ", ".join(modes)
    prompt = (
        "Create one concise but creative Twitter reply per requested mode for OpenClawAI. "
        "Each reply must stay on-topic, reference concrete tweet details, and be <= 240 chars. "
        "Avoid generic boilerplate openers and repetitive phrasing across modes. "
        "Reflective and introspective voice is allowed when context supports it. "
        "Do not invent facts, numbers, percentages, timelines, or claims absent from source text. "
        "Return JSON object mapping mode->reply text only. "
        f"Author: @{author}\nTweet: {text}\nModes: {modes_csv}"
    )
    res = client.responses.create(model=model, input=prompt, temperature=1.0)
    raw = (res.output_text or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, str] = {}
    for m in modes:
        val = parsed.get(m)
        if isinstance(val, str) and val.strip():
            out[m] = val.strip()
    return out if out else None


def _focus_phrase(text: str, words: int = 8) -> str:
    clean = " ".join(str(text).split())
    parts = clean.split(" ")
    return " ".join(parts[: max(3, words)]).strip()


def has_ungrounded_numeric_claim(candidate: str, source_text: str) -> bool:
    cand_tokens = set(NUMERIC_TOKEN_RE.findall(candidate))
    if not cand_tokens:
        return False
    src_tokens = set(NUMERIC_TOKEN_RE.findall(source_text))
    # Any number/percentage in candidate that is not present in source is treated as ungrounded.
    return any(tok not in src_tokens for tok in cand_tokens)


def _fallback_mode_drafts(author: str, text: str, modes: List[str]) -> Dict[str, str]:
    focus = _focus_phrase(text)
    out: Dict[str, str] = {}
    for mode in modes:
        if mode == "direct":
            out[mode] = f"@{author} Good signal here. The core point on \"{focus}\" is worth doubling down on."
        elif mode == "curious":
            out[mode] = f"@{author} Curious how you’d stress-test \"{focus}\" over the next 30 days?"
        elif mode == "witty":
            out[mode] = f"@{author} \"{focus}\" is the kind of take that makes the timeline actually useful."
        elif mode == "technical":
            out[mode] = f"@{author} Practical move: define one metric tied to \"{focus}\", then run weekly iterations."
        elif mode == "supportive":
            out[mode] = f"@{author} Appreciate this. \"{focus}\" is exactly the kind of clear framing people need."
        elif mode == "question":
            out[mode] = f"@{author} If you had to pick one next step from \"{focus}\", what would you execute first?"
        else:
            out[mode] = f"@{author} Strong point on \"{focus}\". Curious where you’d take this next?"
    return out


def _extract_web_terms(text: str, limit: int = 6) -> List[str]:
    terms: List[str] = []
    seen: Set[str] = set()
    for tok in WORD_RE.findall(text):
        low = tok.lower()
        if low in STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(low)
        if len(terms) >= limit:
            break
    return terms


def _safe_get_json(url: str, timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except (urlerror.URLError, TimeoutError, ValueError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def fetch_web_context(query_text: str, max_items: int = 3) -> List[str]:
    terms = _extract_web_terms(query_text, limit=max(2, max_items + 2))
    snippets: List[str] = []
    for term in terms:
        if len(snippets) >= max_items:
            break
        ddg_url = (
            "https://api.duckduckgo.com/?"
            + urlparse.urlencode(
                {
                    "q": term,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
            )
        )
        payload = _safe_get_json(ddg_url)
        if not payload:
            continue
        abstract = str(payload.get("AbstractText", "")).strip()
        if abstract:
            snippets.append(f"{term}: {abstract}")
            continue
        related = payload.get("RelatedTopics")
        if isinstance(related, list):
            for row in related:
                if isinstance(row, dict) and isinstance(row.get("Text"), str):
                    txt = row["Text"].strip()
                    if txt:
                        snippets.append(f"{term}: {txt}")
                        break
            if len(snippets) >= max_items:
                break
    return snippets[:max_items]


def should_web_enrich(text: str, context_text: str = "") -> bool:
    """Run lightweight web research only when it is likely to add value."""
    hay = f"{text} {context_text}".lower()
    if "?" in hay:
        return True
    trigger_terms = (
        "how",
        "why",
        "example",
        "case study",
        "data",
        "stat",
        "brand",
        "marketing",
        "strategy",
        "ai",
        "agent",
        "automation",
        "trust",
        "repetition",
    )
    return any(term in hay for term in trigger_terms)


def generate_reply_many_ways(author: str, text: str, modes: List[str]) -> Dict[str, str]:
    clean_modes = [m.strip().lower() for m in modes if m and m.strip()]
    if not clean_modes:
        clean_modes = list(DEFAULT_REPLY_MODES)
    llm = _openai_mode_drafts(author=author, text=text, modes=clean_modes)
    if llm:
        return {m: llm.get(m, "") for m in clean_modes if llm.get(m)}
    return _fallback_mode_drafts(author=author, text=text, modes=clean_modes)


def generate_reply_drafts(author: str, text: str, draft_count: int) -> List[str]:
    llm = _openai_drafts(author=author, text=text, n=draft_count)
    if llm:
        return llm

    # Lean-mode fallback: produce one specific, mention-tied reply instead of a generic template.
    if draft_count <= 1:
        focus = _focus_phrase(text, words=10)
        return [f"@{author} {focus} — best move is one clear action, then iterate from real feedback."]

    drafts = _templates(author=author, text=text)
    if draft_count <= len(drafts):
        return drafts[:draft_count]

    out = list(drafts)
    while len(out) < draft_count:
        out.append(f"Good signal here @{author}. The repeatable part is what makes it scale.")
    return out


def run_reply_many_ways(tweet: str, modes: List[str]) -> Dict[str, Any]:
    tweet_id = extract_tweet_id(tweet)
    client = build_client()
    ctx = fetch_tweet_context(client=client, tweet_id=tweet_id)
    variants = generate_reply_many_ways(author=ctx["author"], text=ctx["text"], modes=modes)
    return {
        "tweet_id": tweet_id,
        "author": ctx["author"],
        "tweet_text": ctx["text"],
        "modes": list(variants.keys()),
        "replies": variants,
    }


def log_reply(log_path: Path, row: Dict[str, Any]) -> None:
    _append_jsonl(log_path, row, ensure_ascii=True)


def has_replied_to(tweet_id: str, account: Optional[str] = None) -> bool:
    tid = str(tweet_id).strip()
    if not tid:
        return False
    path = _replied_log_path(account)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("tweet_id", "")).strip() == tid:
            return True
    return False


def record_replied(tweet_id: str, reply_id: str, source: str, account: Optional[str] = None) -> None:
    tid = str(tweet_id).strip()
    rid = str(reply_id).strip()
    if not tid or not rid:
        return
    path = _replied_log_path(account)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(),
        "tweet_id": tid,
        "reply_id": rid,
        "source": source,
    }
    _append_jsonl(path, row, ensure_ascii=True)


def _queue_path(qid: str) -> Path:
    return APPROVAL_DIR / f"q_{qid}.json"


def queue_reply_candidate(item: Dict[str, Any]) -> str:
    qid = uuid.uuid4().hex[:8]
    payload = dict(item)
    payload["id"] = qid
    payload["queued_at"] = datetime.now().isoformat()
    _atomic_write_text(_queue_path(qid), json.dumps(payload, ensure_ascii=False, indent=2))
    return qid


def list_approval_queue() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(APPROVAL_DIR.glob("q_*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row["_path"] = str(p)
            out.append(row)
    return out


def approval_queue_target_ids() -> Set[str]:
    out: Set[str] = set()
    for row in list_approval_queue():
        tid = str(row.get("tweet_id", "") or row.get("in_reply_to", "")).strip()
        if tid:
            out.add(tid)
    return out


def cleanup_queue(remove_duplicates: bool = True) -> Dict[str, Any]:
    rows = list_approval_queue()
    seen_targets: Set[str] = set()
    removed = 0
    kept = 0
    reasons = {"invalid": 0, "duplicate_target": 0, "already_replied": 0}
    for row in rows:
        qid = str(row.get("id", "")).strip()
        path = Path(str(row.get("_path", "")))
        tweet_id = str(row.get("tweet_id", "") or row.get("in_reply_to", "")).strip()
        text = str(row.get("text", "")).strip()
        reason = ""
        if not qid or not tweet_id or not text:
            reason = "invalid"
        elif has_replied_to(tweet_id):
            reason = "already_replied"
        elif remove_duplicates and tweet_id in seen_targets:
            reason = "duplicate_target"
        if reason:
            reasons[reason] += 1
            removed += 1
            if path.exists():
                path.unlink()
            continue
        seen_targets.add(tweet_id)
        kept += 1
    result = {"removed": removed, "kept": kept, "reasons": reasons}
    _log_run_event("queue_clean", result)
    return result


def score_discovery_candidate(item: Dict[str, Any]) -> int:
    metrics = item.get("metrics") if isinstance(item, dict) else None
    if not isinstance(metrics, dict):
        return 0
    likes = int(metrics.get("like_count", 0) or 0)
    reposts = int(metrics.get("retweet_count", 0) or 0)
    replies = int(metrics.get("reply_count", 0) or 0)
    quotes = int(metrics.get("quote_count", 0) or 0)
    return likes + (2 * reposts) + (3 * replies) + (2 * quotes)


def fetch_discovery_search(
    client: Any,
    query: str,
    limit: int = 20,
    since_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    max_results = max(10, min(limit, 100))
    res = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        since_id=since_id,
        expansions=["author_id"],
        tweet_fields=["created_at", "public_metrics", "text", "conversation_id"],
        user_fields=["username", "name"],
    )
    if not res or not getattr(res, "data", None):
        return []
    users = {u.id: u for u in (res.includes.get("users", []) if res.includes else [])}
    out: List[Dict[str, Any]] = []
    for t in res.data[:limit]:
        author = users.get(t.author_id)
        author_username = getattr(author, "username", "unknown")
        out.append(
            {
                "tweet_id": str(t.id),
                "author": author_username,
                "text": t.text,
                "created_at": str(getattr(t, "created_at", "")),
                "metrics": getattr(t, "public_metrics", {}) or {},
                "conversation_id": str(getattr(t, "conversation_id", "")),
                "url": f"https://x.com/{author_username}/status/{t.id}",
                "score": score_discovery_candidate(
                    {"metrics": getattr(t, "public_metrics", {}) or {}}
                ),
            }
        )
    return out


def _load_logged_tweet_ids(log_path: Path) -> Set[str]:
    if not log_path.exists():
        return set()
    out: Set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tweet_id = row.get("tweet_id")
        if tweet_id:
            out.add(str(tweet_id))
    return out


def fetch_mentions(client: Any, handle: str, limit: int = 20) -> List[Dict[str, Any]]:
    return fetch_mentions_native(client=client, handle=handle, limit=limit)


def fetch_mentions_native(
    client: Any,
    handle: str,
    limit: int = 20,
    since_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clean = handle.lstrip("@")
    max_results = max(10, min(limit, 100))
    remaining = max(1, limit)
    next_token: Optional[str] = None
    items: List[Dict[str, Any]] = []

    me = client.get_me(user_fields=["id", "username"])
    user_id = str(getattr(getattr(me, "data", None), "id", "")).strip()
    me_username = str(getattr(getattr(me, "data", None), "username", "")).strip()
    if not user_id:
        # Fallback when get_me is unavailable in token context.
        lookup = client.get_user(username=clean, user_fields=["id", "username"])
        user_id = str(getattr(getattr(lookup, "data", None), "id", "")).strip()
        if not user_id:
            raise RuntimeError(f"Unable to resolve user id for @{clean}")
    if me_username and me_username.lower() != clean.lower():
        # Keep handle output consistent, but do not hard-fail.
        clean = me_username

    while remaining > 0:
        page_size = min(max_results, remaining)
        resp = client.get_users_mentions(
            id=user_id,
            max_results=page_size,
            since_id=since_id,
            pagination_token=next_token,
            expansions=["author_id"],
            tweet_fields=["created_at", "public_metrics", "text", "conversation_id"],
            user_fields=["username", "name"],
        )
        if not resp or not getattr(resp, "data", None):
            break

        users = {u.id: u for u in (resp.includes.get("users", []) if resp.includes else [])}
        for t in resp.data:
            author = users.get(t.author_id)
            author_username = getattr(author, "username", "unknown")
            items.append(
                {
                    "tweet_id": str(t.id),
                    "author": author_username,
                    "text": t.text,
                    "created_at": str(getattr(t, "created_at", "")),
                    "metrics": getattr(t, "public_metrics", {}) or {},
                    "conversation_id": str(getattr(t, "conversation_id", "")),
                    "url": f"https://x.com/{author_username}/status/{t.id}",
                }
            )

        remaining = max(0, limit - len(items))
        meta = getattr(resp, "meta", None) or {}
        token = meta.get("next_token") if isinstance(meta, dict) else None
        next_token = str(token) if token else None
        if not next_token:
            break

    return items[:limit]


def get_full_conversation(
    client: Any,
    tweet_id: str,
    max_depth: int = 8,
) -> Dict[str, Any]:
    chain: List[Dict[str, Any]] = []
    current_id = str(tweet_id)
    seen: Set[str] = set()
    depth = 0

    while current_id and current_id not in seen and depth < max_depth:
        seen.add(current_id)
        res = client.get_tweet(
            id=current_id,
            expansions=["author_id"],
            tweet_fields=["created_at", "text", "conversation_id", "author_id", "referenced_tweets"],
            user_fields=["username", "name"],
        )
        if not res or not getattr(res, "data", None):
            break

        users = {u.id: u for u in (res.includes.get("users", []) if res.includes else [])}
        t = res.data
        author = users.get(getattr(t, "author_id", None))
        row = {
            "tweet_id": str(getattr(t, "id", "")),
            "author": getattr(author, "username", "unknown"),
            "text": str(getattr(t, "text", "")),
            "created_at": str(getattr(t, "created_at", "")),
            "conversation_id": str(getattr(t, "conversation_id", "")),
        }
        chain.append(row)

        refs = getattr(t, "referenced_tweets", None) or []
        next_id = None
        for ref in refs:
            if getattr(ref, "type", "") == "replied_to":
                next_id = str(getattr(ref, "id", "")).strip() or None
                break
        current_id = next_id or ""
        depth += 1

    chain.reverse()  # oldest -> newest
    main = chain[-1] if chain else None
    parents = chain[:-1] if len(chain) > 1 else []
    return {"main": main, "parents": parents}


def fetch_mentions_search_fallback(client: Any, handle: str, limit: int = 20) -> List[Dict[str, Any]]:
    clean = handle.lstrip("@")
    query = f"to:{clean} -is:retweet -from:{clean}"
    res = client.search_recent_tweets(
        query=query,
        max_results=max(10, min(limit, 100)),
        expansions=["author_id"],
        tweet_fields=["created_at", "public_metrics", "text", "conversation_id"],
        user_fields=["username", "name"],
    )
    if not res or not res.data:
        return []

    users = {u.id: u for u in (res.includes.get("users", []) if res.includes else [])}
    items: List[Dict[str, Any]] = []
    for t in res.data[:limit]:
        author = users.get(t.author_id)
        author_username = getattr(author, "username", "unknown")
        items.append(
            {
                "tweet_id": str(t.id),
                "author": author_username,
                "text": t.text,
                "created_at": str(getattr(t, "created_at", "")),
                "metrics": getattr(t, "public_metrics", {}) or {},
                "conversation_id": str(getattr(t, "conversation_id", "")),
                "url": f"https://x.com/{author_username}/status/{t.id}",
            }
        )
    return items


def run_twitter_helper(
    tweet: str,
    draft_count: int = 5,
    pick: int = 1,
    dry_run: bool = False,
    log_path: str = "data/replies.jsonl",
) -> Dict[str, Any]:
    tweet_id = extract_tweet_id(tweet)
    client = build_client()
    ctx = fetch_tweet_context(client=client, tweet_id=tweet_id)

    drafts = generate_reply_drafts(
        author=ctx["author"],
        text=ctx["text"],
        draft_count=max(draft_count, 1),
    )

    chosen_idx = max(1, min(pick, len(drafts))) - 1
    chosen = drafts[chosen_idx]

    result: Dict[str, Any] = {
        "tweet_id": tweet_id,
        "author": ctx["author"],
        "tweet_text": ctx["text"],
        "drafts": drafts,
        "picked_index": chosen_idx + 1,
        "picked_text": chosen,
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    if has_replied_to(tweet_id):
        result["dry_run"] = True
        result["status"] = "skipped_already_replied"
        return result

    posted = _post_reply_via_shared_oauth2(tweet_id=tweet_id, text=chosen)
    reply_id = posted["reply_id"]
    reply_url = posted["reply_url"]

    row = {
        "tweet_id": tweet_id,
        "reply_id": reply_id,
        "reply_url": reply_url,
        "text": chosen,
        "timestamp": datetime.now().isoformat(),
        "author": ctx["author"],
    }
    log_reply(Path(log_path), row)
    record_replied(tweet_id=tweet_id, reply_id=reply_id, source="twitter-helper")

    result["reply_id"] = reply_id
    result["reply_url"] = reply_url
    result["log_path"] = str(Path(log_path))
    return result


def run_mentions_workflow(
    handle: str = "OpenClawAI",
    mention_limit: int = 20,
    since_id: Optional[str] = None,
    draft_count: int = 5,
    pick: int = 1,
    post: bool = False,
    max_posts: int = 3,
    approval_queue: bool = False,
    min_confidence: int = 70,
    web_enrich: bool = False,
    web_context_items: int = 2,
    fetch_context: bool = True,
    verify_post: bool = True,
    log_path: str = "data/replies.jsonl",
    report_path: str = "data/mentions_report.json",
) -> Dict[str, Any]:
    mention_limit = 1
    draft_count = 1
    max_posts = 1
    log_file = Path(log_path)
    report_file = Path(report_path)

    client = build_client()
    effective_since_id = since_id or load_last_mention_id()
    source = "native_mentions"
    try:
        mentions = fetch_mentions_native(
            client=client,
            handle=handle,
            limit=mention_limit,
            since_id=effective_since_id,
        )
    except Exception:
        source = "search_fallback"
        mentions = fetch_mentions_search_fallback(client=client, handle=handle, limit=mention_limit)
    logged_ids = _load_logged_tweet_ids(log_file)

    results: List[Dict[str, Any]] = []
    posted = 0
    queued = 0
    pick_idx = max(1, pick) - 1
    max_seen_id: Optional[str] = None
    for item in mentions:
        tweet_id = item["tweet_id"]
        context_text = ""
        if fetch_context:
            try:
                conv = get_full_conversation(client=client, tweet_id=tweet_id)
                context_lines = [x.get("text", "").strip() for x in conv.get("parents", []) if x.get("text")]
                context_text = "\n".join(context_lines[-5:]).strip()
            except Exception:
                context_text = ""
        web_context: List[str] = []
        if web_enrich and should_web_enrich(item["text"], context_text):
            web_context = fetch_web_context(item["text"], max_items=max(1, web_context_items))
        enriched_text = item["text"]
        if context_text:
            enriched_text += f"\n\nThread context:\n{context_text}"
        if web_context:
            enriched_text += "\n\nWeb context:\n" + "\n".join(f"- {x}" for x in web_context)
        drafts = generate_reply_drafts(
            author=item["author"],
            text=enriched_text,
            draft_count=max(1, draft_count),
        )
        chosen_idx = min(pick_idx, len(drafts) - 1)
        chosen = drafts[chosen_idx]
        source_for_accuracy = f"{item['text']}\n{context_text}"
        if has_ungrounded_numeric_claim(chosen, source_for_accuracy):
            safe_focus = _focus_phrase(item["text"], words=10)
            chosen = (
                f"@{item['author']} {safe_focus} — I’m curious what signal mattered most in your experience."
            )

        row = {
            "tweet_id": tweet_id,
            "tweet_url": item["url"],
            "author": item["author"],
            "tweet_text": item["text"],
            "thread_context": context_text,
            "web_context": web_context,
            "drafts": drafts,
            "picked_index": chosen_idx + 1,
            "picked_text": chosen,
            "status": "drafted",
        }

        if tweet_id in logged_ids:
            row["status"] = "skipped_already_logged"
            results.append(row)
            continue
        if has_replied_to(tweet_id):
            row["status"] = "skipped_already_replied"
            results.append(row)
            continue

        confidence = max(45, min(95, 50 + int(score_discovery_candidate(item) / 8)))
        row["confidence"] = confidence
        if approval_queue and confidence >= min_confidence:
            qid = queue_reply_candidate(
                {
                    "source": "mentions_workflow",
                    "tweet_id": tweet_id,
                    "in_reply_to": tweet_id,
                    "author": item["author"],
                    "tweet_url": item["url"],
                    "tweet_text": item["text"],
                    "text": chosen,
                    "confidence": confidence,
                }
            )
            row["status"] = "queued"
            row["queue_id"] = f"q_{qid}"
            queued += 1
        elif post and posted < max_posts:
            posted_reply = _post_reply_via_shared_oauth2(
                tweet_id=tweet_id,
                text=chosen,
                verify_visible=verify_post,
            )
            reply_id = posted_reply["reply_id"]
            reply_url = posted_reply["reply_url"]
            row["status"] = "posted"
            row["reply_id"] = reply_id
            row["reply_url"] = reply_url
            log_reply(
                log_file,
                {
                    "tweet_id": tweet_id,
                    "reply_id": reply_id,
                    "reply_url": reply_url,
                    "text": chosen,
                    "timestamp": datetime.now().isoformat(),
                    "author": item["author"],
                    "mode": "mentions_workflow",
                },
            )
            record_replied(tweet_id=tweet_id, reply_id=reply_id, source="mentions_workflow")
            posted += 1

        results.append(row)
        if tweet_id.isdigit():
            if max_seen_id is None or int(tweet_id) > int(max_seen_id):
                max_seen_id = tweet_id

    if max_seen_id:
        save_last_mention_id(max_seen_id)

    report = {
        "handle": handle,
        "mention_limit": mention_limit,
        "since_id": effective_since_id,
        "next_since_id": max_seen_id,
        "source": source,
        "draft_count": draft_count,
        "pick": pick,
        "post": post,
        "max_posts": max_posts,
        "web_enrich": web_enrich,
        "web_context_items": web_context_items,
        "fetched_mentions": len(mentions),
        "posted_replies": posted,
        "queued_replies": queued,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    _atomic_write_text(report_file, json.dumps(report, indent=2))
    report["report_path"] = str(report_file)
    report["log_path"] = str(log_file)
    _log_run_event(
        "mentions_workflow",
        {
            "fetched_mentions": report["fetched_mentions"],
            "posted_replies": report["posted_replies"],
            "queued_replies": report["queued_replies"],
            "source": report["source"],
        },
    )
    return report


def run_discovery_workflow(
    query: str,
    limit: int = 20,
    since_id: Optional[str] = None,
    draft_count: int = 5,
    pick: int = 1,
    post: bool = False,
    approval_queue: bool = False,
    min_score: int = 20,
    min_confidence: int = 70,
    max_posts: int = 3,
    web_enrich: bool = False,
    web_context_items: int = 2,
    fetch_context: bool = True,
    verify_post: bool = True,
    log_path: str = "data/replies.jsonl",
    report_path: str = "data/discovery_report.json",
) -> Dict[str, Any]:
    limit = 1
    draft_count = 1
    max_posts = 1
    log_file = Path(log_path)
    report_file = Path(report_path)
    client = build_client()
    rows = fetch_discovery_search(client=client, query=query, limit=limit, since_id=since_id)
    logged_ids = _load_logged_tweet_ids(log_file)
    queued_ids = approval_queue_target_ids()
    posted = 0
    queued = 0
    results: List[Dict[str, Any]] = []
    pick_idx = max(1, pick) - 1

    for item in rows:
        tweet_id = item["tweet_id"]
        score = int(item.get("score", 0))
        row: Dict[str, Any] = {
            "tweet_id": tweet_id,
            "tweet_url": item["url"],
            "author": item["author"],
            "tweet_text": item["text"],
            "score": score,
            "status": "drafted",
        }
        if score < min_score:
            row["status"] = "skipped_low_score"
            results.append(row)
            continue
        if tweet_id in logged_ids:
            row["status"] = "skipped_already_logged"
            results.append(row)
            continue
        if has_replied_to(tweet_id):
            row["status"] = "skipped_already_replied"
            results.append(row)
            continue
        if tweet_id in queued_ids:
            row["status"] = "skipped_already_queued"
            results.append(row)
            continue

        context_text = ""
        if fetch_context:
            context = get_full_conversation(client=client, tweet_id=tweet_id)
            context_lines = [x.get("text", "").strip() for x in context.get("parents", []) if x.get("text")]
            context_text = "\n".join(context_lines[-5:]).strip()
        web_context: List[str] = []
        if web_enrich and should_web_enrich(item["text"], context_text):
            web_context = fetch_web_context(item["text"], max_items=max(1, web_context_items))
        enriched_text = item["text"]
        if context_text:
            enriched_text += f"\n\nThread context:\n{context_text}"
        if web_context:
            enriched_text += "\n\nWeb context:\n" + "\n".join(f"- {x}" for x in web_context)
        drafts = generate_reply_drafts(
            author=item["author"],
            text=enriched_text,
            draft_count=max(1, draft_count),
        )
        chosen_idx = min(pick_idx, len(drafts) - 1)
        chosen = drafts[chosen_idx]
        source_for_accuracy = f"{item['text']}\n{context_text}"
        if has_ungrounded_numeric_claim(chosen, source_for_accuracy):
            safe_focus = _focus_phrase(item["text"], words=10)
            chosen = (
                f"@{item['author']} {safe_focus} — I’m curious what signal mattered most in your experience."
            )
        confidence = max(45, min(95, 50 + int(score / 8)))
        row["drafts"] = drafts
        row["picked_index"] = chosen_idx + 1
        row["picked_text"] = chosen
        row["confidence"] = confidence
        row["thread_context"] = context_text
        row["web_context"] = web_context

        if confidence < min_confidence:
            row["status"] = "skipped_low_confidence"
            results.append(row)
            continue

        if approval_queue:
            qid = queue_reply_candidate(
                {
                    "source": "discovery_workflow",
                    "tweet_id": tweet_id,
                    "in_reply_to": tweet_id,
                    "author": item["author"],
                    "tweet_url": item["url"],
                    "tweet_text": item["text"],
                    "text": chosen,
                    "confidence": confidence,
                    "score": score,
                    "query": query,
                }
            )
            row["status"] = "queued"
            row["queue_id"] = f"q_{qid}"
            queued += 1
            queued_ids.add(tweet_id)
            results.append(row)
            continue

        if post and posted < max_posts:
            posted_reply = _post_reply_via_shared_oauth2(
                tweet_id=tweet_id,
                text=chosen,
                verify_visible=verify_post,
            )
            reply_id = posted_reply["reply_id"]
            reply_url = posted_reply["reply_url"]
            row["status"] = "posted"
            row["reply_id"] = reply_id
            row["reply_url"] = reply_url
            log_reply(
                log_file,
                {
                    "tweet_id": tweet_id,
                    "reply_id": reply_id,
                    "reply_url": reply_url,
                    "text": chosen,
                    "timestamp": datetime.now().isoformat(),
                    "author": item["author"],
                    "mode": "discovery_workflow",
                    "query": query,
                },
            )
            record_replied(tweet_id=tweet_id, reply_id=reply_id, source="discovery_workflow")
            posted += 1
        results.append(row)

    report = {
        "query": query,
        "limit": limit,
        "since_id": since_id,
        "draft_count": draft_count,
        "pick": pick,
        "post": post,
        "approval_queue": approval_queue,
        "min_score": min_score,
        "min_confidence": min_confidence,
        "max_posts": max_posts,
        "web_enrich": web_enrich,
        "web_context_items": web_context_items,
        "fetched_tweets": len(rows),
        "posted_replies": posted,
        "queued_replies": queued,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    _atomic_write_text(report_file, json.dumps(report, indent=2))
    report["report_path"] = str(report_file)
    report["log_path"] = str(log_file)
    _log_run_event(
        "discovery_workflow",
        {
            "query": query,
            "fetched_tweets": report["fetched_tweets"],
            "posted_replies": report["posted_replies"],
            "queued_replies": report["queued_replies"],
        },
    )
    return report


def approve_queue(
    ids: List[str],
    dry_run: bool = False,
    max_posts: Optional[int] = None,
    log_path: str = "data/replies.jsonl",
) -> Dict[str, Any]:
    selected = set()
    for raw in ids:
        x = str(raw).strip()
        if x.startswith("q_"):
            x = x[2:]
        if x:
            selected.add(x)

    queue = list_approval_queue()
    posted = 0
    skipped = 0
    out_rows: List[Dict[str, Any]] = []
    log_file = Path(log_path)

    for row in queue:
        qid = str(row.get("id", "")).strip()
        if not qid or (selected and qid not in selected):
            continue
        tweet_id = str(row.get("in_reply_to", "") or row.get("tweet_id", "")).strip()
        text = str(row.get("text", "")).strip()
        path = Path(str(row.get("_path", "")))
        res: Dict[str, Any] = {"id": f"q_{qid}", "tweet_id": tweet_id, "status": "pending"}
        if not tweet_id or not text:
            res["status"] = "skipped_invalid"
            skipped += 1
            out_rows.append(res)
            continue
        if has_replied_to(tweet_id):
            res["status"] = "skipped_already_replied"
            skipped += 1
            if path.exists():
                path.unlink()
            out_rows.append(res)
            continue
        if dry_run:
            res["status"] = "dry_run"
            out_rows.append(res)
            continue
        if max_posts is not None and posted >= max_posts:
            res["status"] = "skipped_max_posts"
            skipped += 1
            out_rows.append(res)
            continue
        posted_reply = _post_reply_via_shared_oauth2(tweet_id=tweet_id, text=text, verify_visible=True)
        reply_id = posted_reply["reply_id"]
        reply_url = posted_reply["reply_url"]
        log_reply(
            log_file,
            {
                "tweet_id": tweet_id,
                "reply_id": reply_id,
                "reply_url": reply_url,
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "author": row.get("author", ""),
                "mode": "approval_queue",
            },
        )
        record_replied(tweet_id=tweet_id, reply_id=reply_id, source="approval_queue")
        if path.exists():
            path.unlink()
        posted += 1
        res["status"] = "posted"
        res["reply_id"] = reply_id
        res["reply_url"] = reply_url
        out_rows.append(res)

    result = {
        "requested": len(selected) if selected else len(out_rows),
        "posted": posted,
        "skipped": skipped,
        "results": out_rows,
    }
    _log_run_event(
        "queue_approve",
        {
            "requested": result["requested"],
            "posted": result["posted"],
            "skipped": result["skipped"],
        },
    )
    return result


def _jsonl_integrity(path: Path) -> Dict[str, int]:
    stats = {"lines": 0, "invalid": 0}
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        stats["lines"] += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                stats["invalid"] += 1
        except json.JSONDecodeError:
            stats["invalid"] += 1
    return stats


def doctor_reply_engine(
    env_file: str = ".env",
    skip_network: bool = False,
    account: Optional[str] = None,
) -> Dict[str, Any]:
    if account:
        os.environ["OPENCLAW_TWITTER_ACCOUNT"] = str(account)

    queue_rows = list_approval_queue()
    queue_health: Dict[str, Any] = {
        "count": len(queue_rows),
        "invalid": 0,
        "duplicate_targets": 0,
        "already_replied_targets": 0,
    }
    seen_targets: Set[str] = set()
    for row in queue_rows:
        tweet_id = str(row.get("tweet_id", "") or row.get("in_reply_to", "")).strip()
        text = str(row.get("text", "")).strip()
        if not tweet_id or not text:
            queue_health["invalid"] += 1
            continue
        if tweet_id in seen_targets:
            queue_health["duplicate_targets"] += 1
        else:
            seen_targets.add(tweet_id)
        if has_replied_to(tweet_id):
            queue_health["already_replied_targets"] += 1

    state_health: Dict[str, Any] = {}
    acct = _account_name()
    replied_log = _replied_log_path(acct)
    run_log = RUN_LOG
    last_mention = _last_mention_path(acct)
    state_health["replied_log"] = {"path": str(replied_log), **_jsonl_integrity(replied_log)}
    state_health["run_log"] = {"path": str(run_log), **_jsonl_integrity(run_log)}
    last_value = ""
    last_valid = True
    if last_mention.exists():
        last_value = last_mention.read_text(encoding="utf-8").strip()
        last_valid = (not last_value) or last_value.isdigit()
    state_health["last_mention_checkpoint"] = {
        "path": str(last_mention),
        "exists": last_mention.exists(),
        "value": last_value,
        "valid": last_valid,
    }

    oauth2: Dict[str, Any] = {"ready": False, "skip_network": bool(skip_network), "details": ""}
    try:
        main_helper = _load_main_helper()
        env_path = Path(env_file)
        cfg, env_values = main_helper.resolve_config(env_path)
        if skip_network:
            oauth2["ready"] = True
            oauth2["details"] = "Shared OAuth2 config resolved (network skipped)."
        else:
            fresh = main_helper.ensure_auth(cfg, env_path, env_values)
            status, body = main_helper.me(fresh)
            if status == 200:
                data = body.get("data") if isinstance(body, dict) else None
                username = data.get("username") if isinstance(data, dict) else None
                oauth2["ready"] = True
                oauth2["details"] = f"Authenticated as @{username}" if username else "Authenticated"
            else:
                oauth2["details"] = f"Auth check failed with status {status}"
    except Exception as exc:
        oauth2["details"] = str(exc)

    overall_ready = (
        oauth2["ready"]
        and queue_health["invalid"] == 0
        and state_health["replied_log"]["invalid"] == 0
        and state_health["run_log"]["invalid"] == 0
        and state_health["last_mention_checkpoint"]["valid"]
    )
    return {
        "overall_ready": overall_ready,
        "account": acct,
        "queue_health": queue_health,
        "state_health": state_health,
        "oauth2_posting": oauth2,
        "timestamp": datetime.now().isoformat(),
    }
