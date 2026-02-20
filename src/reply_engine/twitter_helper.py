from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def get_authenticated_username(client: Any) -> Optional[str]:
    try:
        me = client.get_me(user_fields=["username"])
    except Exception:
        return None
    if not me or not getattr(me, "data", None):
        return None
    return getattr(me.data, "username", None)


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
    return {
        "TWITTER_BEARER_TOKEN": bearer,
        "TWITTER_API_KEY": os.getenv("TWITTER_API_KEY"),
        "TWITTER_API_SECRET": os.getenv("TWITTER_API_SECRET"),
        "TWITTER_ACCESS_TOKEN": os.getenv("TWITTER_ACCESS_TOKEN"),
        "TWITTER_ACCESS_SECRET": os.getenv("TWITTER_ACCESS_SECRET"),
    }


def _account_name() -> str:
    return os.getenv("OPENCLAW_TWITTER_ACCOUNT", "default")


def _last_mention_path(account: Optional[str] = None) -> Path:
    acct = account or _account_name()
    return CONFIG_DIR / f"last_mention_id_{acct}.txt"


def load_last_mention_id(account: Optional[str] = None) -> Optional[str]:
    path = _last_mention_path(account)
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    return raw or None


def save_last_mention_id(tweet_id: str, account: Optional[str] = None) -> None:
    path = _last_mention_path(account)
    path.write_text(str(tweet_id).strip(), encoding="utf-8")


def build_client(require_write: bool = True) -> Any:
    try:
        import tweepy
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `tweepy`. Install reply-engine deps with: "
            "pip install -r requirements-reply-engine.txt"
        ) from exc

    env = _required_env()
    required = ["TWITTER_BEARER_TOKEN"]
    if require_write:
        required.extend([
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_SECRET",
        ])
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return tweepy.Client(
        bearer_token=env["TWITTER_BEARER_TOKEN"],
        consumer_key=env["TWITTER_API_KEY"],
        consumer_secret=env["TWITTER_API_SECRET"],
        access_token=env["TWITTER_ACCESS_TOKEN"],
        access_token_secret=env["TWITTER_ACCESS_SECRET"],
        wait_on_rate_limit=True,
    )


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
        f"Strong point @{author}. Biggest unlock is consistent repetition of one core promise, not chasing new angles every day.",
        f"Agree with @{author}. Clear positioning compounds faster than clever copy. One message, many formats, tracked weekly.",
        f"This stands out because it is operational, not just motivational. Teams that turn this into a repeatable system usually win.",
        f"Underrated growth lever: say the same core thing 100 ways instead of 100 different things once.",
        f"Great post @{author}. Curious which channel is converting best right now: organic, creators, or paid?",
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
        "Generate concise Twitter reply drafts for OpenClawAI. "
        "Return plain JSON array of strings only. "
        "Each draft must be <= 240 characters and specific. "
        f"Create {n} replies to this tweet by @{author}: {text}"
    )

    res = client.responses.create(model=model, input=prompt, temperature=0.7)
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


def generate_reply_drafts(author: str, text: str, draft_count: int) -> List[str]:
    llm = _openai_drafts(author=author, text=text, n=draft_count)
    if llm:
        return llm

    drafts = _templates(author=author, text=text)
    if draft_count <= len(drafts):
        return drafts[:draft_count]

    out = list(drafts)
    while len(out) < draft_count:
        out.append(f"Good signal here @{author}. The repeatable part is what makes it scale.")
    return out


def post_reply(client: Any, tweet_id: str, text: str) -> str:
    res = client.create_tweet(text=text, in_reply_to_tweet_id=tweet_id)
    if not res or not res.data or "id" not in res.data:
        raise RuntimeError("Twitter create_tweet returned no reply id")
    return str(res.data["id"])


def verify_reply_visible(
    client: Any,
    reply_id: str,
    expected_username: Optional[str] = None,
    attempts: int = 3,
    delay_seconds: float = 1.0,
) -> Dict[str, str]:
    last_error = "unknown"
    for attempt in range(1, max(1, attempts) + 1):
        try:
            res = client.get_tweet(
                id=reply_id,
                expansions=["author_id"],
                tweet_fields=["author_id", "created_at"],
                user_fields=["username"],
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(delay_seconds)
            continue

        if res and getattr(res, "data", None):
            author_id = getattr(res.data, "author_id", None)
            username = None
            includes = getattr(res, "includes", None) or {}
            users = includes.get("users", []) if isinstance(includes, dict) else []
            for user in users:
                if getattr(user, "id", None) == author_id:
                    username = getattr(user, "username", None)
                    break

            if expected_username and username and username != expected_username:
                raise RuntimeError(
                    f"Reply author mismatch after post: expected @{expected_username}, got @{username}."
                )

            return {
                "username": username or "",
                "url": build_tweet_url(reply_id, username=username),
            }

        if attempt < attempts:
            time.sleep(delay_seconds)

    raise RuntimeError(
        "Twitter returned a reply ID but visibility verification failed. "
        "Do not assume the reply is live. "
        f"Last error: {last_error}"
    )


def log_reply(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


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
    client = build_client(require_write=not dry_run)
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

    auth_username = get_authenticated_username(client)
    reply_id = post_reply(client=client, tweet_id=tweet_id, text=chosen)
    verify = verify_reply_visible(
        client=client,
        reply_id=reply_id,
        expected_username=auth_username,
    )
    reply_url = verify["url"]

    row = {
        "tweet_id": tweet_id,
        "reply_id": reply_id,
        "reply_url": reply_url,
        "text": chosen,
        "timestamp": datetime.now().isoformat(),
        "author": ctx["author"],
    }
    log_reply(Path(log_path), row)

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
    log_path: str = "data/replies.jsonl",
    report_path: str = "data/mentions_report.json",
) -> Dict[str, Any]:
    log_file = Path(log_path)
    report_file = Path(report_path)

    client = build_client(require_write=post)
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
    auth_username = get_authenticated_username(client) or handle.lstrip("@")
    logged_ids = _load_logged_tweet_ids(log_file)

    results: List[Dict[str, Any]] = []
    posted = 0
    pick_idx = max(1, pick) - 1
    max_seen_id: Optional[str] = None
    for item in mentions:
        tweet_id = item["tweet_id"]
        try:
            conv = get_full_conversation(client=client, tweet_id=tweet_id)
            context_lines = [x.get("text", "").strip() for x in conv.get("parents", []) if x.get("text")]
            context_text = "\n".join(context_lines[-5:]).strip()
        except Exception:
            context_text = ""
        drafts = generate_reply_drafts(
            author=item["author"],
            text=f"{item['text']}\n\nThread context:\n{context_text}" if context_text else item["text"],
            draft_count=max(1, draft_count),
        )
        chosen_idx = min(pick_idx, len(drafts) - 1)
        chosen = drafts[chosen_idx]

        row = {
            "tweet_id": tweet_id,
            "tweet_url": item["url"],
            "author": item["author"],
            "tweet_text": item["text"],
            "thread_context": context_text,
            "drafts": drafts,
            "picked_index": chosen_idx + 1,
            "picked_text": chosen,
            "status": "drafted",
        }

        if tweet_id in logged_ids:
            row["status"] = "skipped_already_logged"
            results.append(row)
            continue

        if post and posted < max_posts:
            reply_id = post_reply(client=client, tweet_id=tweet_id, text=chosen)
            verify = verify_reply_visible(
                client=client,
                reply_id=reply_id,
                expected_username=auth_username,
            )
            reply_url = verify["url"]
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
        "fetched_mentions": len(mentions),
        "posted_replies": posted,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_file)
    report["log_path"] = str(log_file)
    return report
