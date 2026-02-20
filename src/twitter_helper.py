#!/usr/bin/env python3
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import mimetypes
import os
import re
import random
import secrets
import shutil
import sys
import tempfile
import time
import uuid
import getpass
from datetime import datetime, timedelta, timezone
from email.message import Message
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

API_BASE = "https://api.twitter.com/2"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
MEDIA_UPLOAD_URL = f"{API_BASE}/media/upload"
MEDIA_METADATA_URL = f"{API_BASE}/media/metadata"
MAX_TWEET_LEN = 280
MAX_UNBROKEN_SEGMENT_LEN = 48
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8080/callback"
DEFAULT_SCOPES = "tweet.read tweet.write users.read offline.access media.write"
RUN_TAG_SUFFIX_RE = re.compile(
    r"\s*\[(?:openclaw|twitter-engine)-\d{8}-\d{6}-[a-z0-9]{4}\]\s*$",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
TOOL_ROOT = Path(__file__).resolve().parents[1]
TOKEN_SERVICE_NAME = "twitter-engine"
LEGACY_TOKEN_SERVICE_NAME = "openclaw-twitter-helper"
TOKEN_CONFIG_DIR = Path.home() / ".config" / "twitter-engine"
LEGACY_TOKEN_CONFIG_DIR = Path.home() / ".config" / "openclaw-twitter-helper"
TOKEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TOKENS_JSON_FALLBACK = TOKEN_CONFIG_DIR / "tokens.json"
RATE_LIMIT_BARRIER_FILE = TOKEN_CONFIG_DIR / "rate_limit_barrier.json"
ACTIVE_ACCOUNT = "default"
USER_ID_CACHE_TTL_SECONDS = 86400
APPROVAL_DIR = TOKEN_CONFIG_DIR / "approval_queue"
APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
RECENT_REPLIES_CACHE = TOKEN_CONFIG_DIR / "recent_replies.jsonl"
RECENT_POSTS_CACHE = TOKEN_CONFIG_DIR / "recent_posts.jsonl"
TWEET_MEMORY_LOG = TOKEN_CONFIG_DIR / "tweet_memory.jsonl"
PERSONA_FILE = TOKEN_CONFIG_DIR / "persona" / "twitter-engine.md"
REPLIED_DEDUPE_DAYS = 90
ALLOWED_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_SIZE_MB = 5
MAX_IMAGES = 5
MAX_REPLY_DRAFT_LEN = 240
WEB_REQ_TIMEOUT_SECONDS = 8
WEB_REQ_MAX_BYTES = 900_000
WEB_REQ_MAX_RETRIES = 2
WEB_INSPIRATION_CACHE_TTL_SECONDS = 120
GENERIC_REPLY_OPENERS = (
    "great point",
    "totally agree",
    "interesting",
    "love this",
    "well said",
    "this is great",
)
MARKETING_PIVOT_TERMS = (
    "brand",
    "branding",
    "marketing",
    "growth",
    "funnel",
    "positioning",
    "audience",
    "acquisition",
    "business",
    "sales",
)
CORPORATE_TONE_TERMS = (
    "leverage",
    "positioning",
    "go-to-market",
    "kpi",
    "synergy",
    "stakeholder",
)
FROG_PERSONA = {
    "hooks": [
        "hot take from frog mode:",
        "real ones know:",
        "counterintuitively,",
        "the pattern under the pattern:",
        "watch this:",
        "this one's for the degens:",
        "unpopular but correct:",
        "scroll-stopping truth:",
    ],
    "closers": [
        "what's your move?",
        "prove me wrong.",
        "thread if you're based.",
        "frog or pass.",
        "delta or cope.",
        "gm if you felt this.",
    ],
    "slang_bank": [
        "taxing",
        "cooked",
        "ngmi",
        "wagmi but make it chaotic",
        "send it",
        "ratio incoming",
        "frog army assemble",
    ],
    "emoji_density": 0.30,
}
VOICE_CONFIGS: Dict[str, Optional[Dict[str, object]]] = {
    "chaotic": {
        "slang_intensity": 0.75,
        "emoji_density": 0.40,
        "hooks": ["frog mode:", "watch this:", "frogs know:"],
        "slang_terms": ["send it", "ratio incoming", "taxing"],
        "extra_tail": "",
    },
    "degen": {
        "slang_intensity": 0.95,
        "emoji_density": 0.55,
        "hooks": ["aped in:", "degen play:", "frogs only:"],
        "slang_terms": ["heeming", "cooked", "wagmi", "perps", "ngmi"],
        "extra_tail": "",
    },
    "based": {
        "slang_intensity": 0.60,
        "emoji_density": 0.25,
        "hooks": ["real ones know:", "unpopular but correct:", "mid behavior detected:"],
        "slang_terms": ["based", "prove me wrong", "taxing"],
        "extra_tail": "",
    },
    "savage": {
        "slang_intensity": 0.80,
        "emoji_density": 0.35,
        "hooks": ["ratio incoming:", "normies will seethe:", "cooked:"],
        "slang_terms": ["ratio incoming", "cooked", "send it"],
        "extra_tail": "normies seething in replies",
    },
    "operator": {
        "slang_intensity": 0.45,
        "emoji_density": 0.20,
        "hooks": ["execution note:", "pipeline update:", "track this:"],
        "slang_terms": ["delta", "pipeline", "constraints"],
        "extra_tail": "",
    },
    "sage": {
        "slang_intensity": 0.35,
        "emoji_density": 0.15,
        "hooks": ["the pattern under:", "counterintuitively,", "long-term frogs know:"],
        "slang_terms": ["pattern", "compounding", "signal"],
        "extra_tail": "",
    },
    "shitposter": {
        "slang_intensity": 1.0,
        "emoji_density": 0.70,
        "hooks": ["lmao imagine:", "brainrot hours:", "chaos bulletin:"],
        "slang_terms": ["cooked", "ngmi", "send it", "ratio incoming"],
        "extra_tail": "",
    },
    "auto": None,
}
HIGH_SIGNAL_LEXICON = {
    "service": "stack",
    "workflow": "pipeline",
    "process": "flow",
    "feature": "weapon",
    "solution": "move",
    "integration": "hook",
    "implementation": "send-it",
    "optimization": "heeming",
    "improvement": "delta",
    "important": "taxing",
    "valuable": "based",
    "consider": "prove me wrong",
    "note": "watch this",
    "actually": "real ones know",
    "essentially": "counterintuitively",
    "basically": "straight up",
}
ANTI_BORING_BANNED = {
    "you're not wrong",
    "this hits",
    "important to note",
    "food for thought",
    "worth considering",
}
FROG_JUDGE_WEIGHTS = {
    "specificity": 0.30,
    "frog_energy": 0.25,
    "engagement": 0.20,
    "anti_boring": 0.15,
    "voice_authenticity": 0.10,
}
VIRAL_POTENTIAL_BONUS = {
    "question_or_challenge": 12,
    "number_or_metric": 8,
    "frog_hook": 10,
    "optimal_length": 6,
    "emoji_fit": 5,
    "contrarian_signal": 9,
}
VIRAL_PACKS: Dict[str, Optional[Dict[str, object]]] = {
    "light": {
        "voice": "chaotic",
        "style": "auto",
        "ensemble": 3,
        "viral_boost": False,
        "judge_threshold": 78.0,
        "anti_boring": True,
        "sharpen": True,
    },
    "medium": {
        "voice": "auto",
        "style": "auto",
        "ensemble": 5,
        "viral_boost": True,
        "judge_threshold": 82.0,
        "anti_boring": True,
        "sharpen": True,
    },
    "nuclear": {
        "voice": "degen",
        "style": "contrarian",
        "ensemble": 8,
        "viral_boost": True,
        "judge_threshold": 88.0,
        "anti_boring": True,
        "sharpen": True,
    },
    "alpha": {
        "voice": "sage",
        "style": "operator",
        "ensemble": 6,
        "viral_boost": True,
        "judge_threshold": 85.0,
        "anti_boring": True,
        "sharpen": True,
    },
    "chaos": {
        "voice": "shitposter",
        "style": "auto",
        "ensemble": 7,
        "viral_boost": True,
        "judge_threshold": 80.0,
        "anti_boring": True,
        "sharpen": True,
    },
    "infinite": {
        "voice": "based",
        "style": "contrarian",
        "ensemble": 8,
        "viral_boost": True,
        "judge_threshold": 88.0,
        "anti_boring": True,
        "sharpen": True,
    },
    "auto": None,
}
WEB_INSPIRATION_CACHE: Dict[str, Dict[str, object]] = {}


def _migrate_legacy_config_dir() -> None:
    if not LEGACY_TOKEN_CONFIG_DIR.exists():
        return
    for legacy in LEGACY_TOKEN_CONFIG_DIR.rglob("*"):
        rel = legacy.relative_to(LEGACY_TOKEN_CONFIG_DIR)
        target = TOKEN_CONFIG_DIR / rel
        if legacy.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(legacy, target)
        except Exception:
            continue


_migrate_legacy_config_dir()


def creative_mode_enabled() -> bool:
    raw = str(os.getenv("OPENCLAW_CREATIVE_MODE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}

CONFIG_KEYS = [
    "TWITTER_CLIENT_ID",
    "TWITTER_CLIENT_SECRET",
    "TWITTER_BEARER_TOKEN",
    "TWITTER_OAUTH2_ACCESS_TOKEN",
    "TWITTER_OAUTH2_REFRESH_TOKEN",
    "TWITTER_REDIRECT_URI",
    "TWITTER_WEBSITE_URL",
    "TWITTER_SCOPES",
]

@dataclass
class Config:
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str


class TwitterHelperError(Exception):
    pass


try:
    import keyring
except Exception:  # pragma: no cover - optional dependency fallback
    keyring = None

try:
    import requests
except Exception:  # pragma: no cover - optional dependency fallback
    requests = None


STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
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
    "those",
    "very",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def load_env_file(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not env_path.exists():
        return data

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def write_env_file(env_path: Path, values: Dict[str, str]) -> None:
    lines = ["# Twitter Helper config"]

    for key in CONFIG_KEYS:
        lines.append(f"{key}={values.get(key, '')}")

    extra_keys = sorted(k for k in values.keys() if k not in CONFIG_KEYS)
    for key in extra_keys:
        lines.append(f"{key}={values.get(key, '')}")

    lines.append("")
    env_path.write_text("\n".join(lines), encoding="utf-8")


class TokenManager:
    def __init__(self, env_path: Path, account: str = "default"):
        self.env_path = env_path
        self.account = account
        self.key_prefix = f"{TOKEN_SERVICE_NAME}:{account}"
        self.legacy_key_prefix = f"{LEGACY_TOKEN_SERVICE_NAME}:{account}"

    def _keyring_available(self) -> bool:
        return keyring is not None

    def keyring_status_label(self) -> str:
        if not self._keyring_available():
            return "unavailable (module missing, using .env fallback)"
        try:
            backend = keyring.get_keyring()
            return f"accessible ({backend.__class__.__name__})"
        except Exception:
            return "unavailable (runtime error, using .env fallback)"

    def _read_keyring_payload(self) -> Optional[Dict[str, str]]:
        if not self._keyring_available():
            return None
        for prefix in (self.key_prefix, self.legacy_key_prefix):
            try:
                raw = keyring.get_password(prefix, "oauth_tokens")
                if not raw:
                    continue
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return {k: str(v) if v is not None else "" for k, v in parsed.items()}
            except Exception:
                continue
        return None

    def save_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str],
        env_values: Dict[str, str],
    ) -> str:
        payload = {
            "access_token": access_token,
            "refresh_token": refresh_token or "",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "account": self.account,
        }

        if self._keyring_available():
            try:
                keyring.set_password(self.key_prefix, "oauth_tokens", json.dumps(payload))
                # Scrub plaintext OAuth2 tokens from .env when secure storage is available.
                env_values["TWITTER_OAUTH2_ACCESS_TOKEN"] = ""
                env_values["TWITTER_OAUTH2_REFRESH_TOKEN"] = ""
                write_env_file(self.env_path, env_values)
                return "keyring"
            except Exception:
                pass

        env_values["TWITTER_OAUTH2_ACCESS_TOKEN"] = access_token
        env_values["TWITTER_OAUTH2_REFRESH_TOKEN"] = refresh_token or ""
        write_env_file(self.env_path, env_values)
        TOKENS_JSON_FALLBACK.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        TOKENS_JSON_FALLBACK.chmod(0o600)
        return "env+json"

    def get_access_token(self, env_values: Dict[str, str]) -> str:
        token = os.getenv("TWITTER_OAUTH2_ACCESS_TOKEN")
        if token:
            return token
        payload = self._read_keyring_payload()
        if payload and payload.get("access_token"):
            return payload["access_token"]
        return env_values.get("TWITTER_OAUTH2_ACCESS_TOKEN", "")

    def get_refresh_token(self, env_values: Dict[str, str]) -> str:
        token = os.getenv("TWITTER_OAUTH2_REFRESH_TOKEN")
        if token:
            return token
        payload = self._read_keyring_payload()
        if payload and payload.get("refresh_token"):
            return payload["refresh_token"]
        return env_values.get("TWITTER_OAUTH2_REFRESH_TOKEN", "")

    def migrate_from_env(self, env_values: Dict[str, str]) -> bool:
        if not self._keyring_available():
            return False
        if self._read_keyring_payload():
            return False
        access = env_values.get("TWITTER_OAUTH2_ACCESS_TOKEN", "")
        refresh = env_values.get("TWITTER_OAUTH2_REFRESH_TOKEN", "")
        if not access:
            return False
        self.save_tokens(access, refresh, env_values)
        return True


def token_manager(env_path: Path) -> TokenManager:
    return TokenManager(env_path=env_path, account=ACTIVE_ACCOUNT)


def resolve_config(env_path: Path) -> Tuple[Config, Dict[str, str]]:
    env = load_env_file(env_path)
    tm = token_manager(env_path)
    tm.migrate_from_env(env)

    def get(name: str) -> str:
        return os.getenv(name) or env.get(name, "")

    cfg = Config(
        client_id=get("TWITTER_CLIENT_ID"),
        client_secret=get("TWITTER_CLIENT_SECRET"),
        access_token=tm.get_access_token(env),
        refresh_token=tm.get_refresh_token(env),
    )

    missing = [
        name
        for name, value in {
            "TWITTER_CLIENT_ID": cfg.client_id,
            "TWITTER_CLIENT_SECRET": cfg.client_secret,
            "TWITTER_OAUTH2_ACCESS_TOKEN": cfg.access_token,
            "TWITTER_OAUTH2_REFRESH_TOKEN": cfg.refresh_token,
        }.items()
        if not value
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise TwitterHelperError(
            f"Missing required config values: {missing_str}. Run `setup` + `auth-login` first."
        )

    return cfg, env


def _headers_to_dict(headers: Optional[Message]) -> Dict[str, str]:
    if headers is None:
        return {}
    out: Dict[str, str] = {}
    for k, v in headers.items():
        out[str(k).lower()] = str(v)
    return out


def _load_rate_limit_barrier() -> Dict[str, object]:
    if not RATE_LIMIT_BARRIER_FILE.exists():
        return {}
    try:
        raw = RATE_LIMIT_BARRIER_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_rate_limit_barrier(until_ts: float, source: str, url: str) -> None:
    RATE_LIMIT_BARRIER_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "until_ts": float(until_ts),
        "source": source,
        "url": url,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    RATE_LIMIT_BARRIER_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clear_rate_limit_barrier() -> None:
    try:
        if RATE_LIMIT_BARRIER_FILE.exists():
            RATE_LIMIT_BARRIER_FILE.unlink()
    except Exception:
        pass


def get_rate_limit_barrier_status() -> Dict[str, object]:
    data = _load_rate_limit_barrier()
    until_ts = float(data.get("until_ts", 0) or 0)
    now = time.time()
    active = until_ts > now
    wait_seconds = int(max(0, until_ts - now))
    return {
        "active": active,
        "wait_seconds": wait_seconds,
        "until_ts": until_ts,
        "source": str(data.get("source", "")),
        "url": str(data.get("url", "")),
    }


def http_json_with_headers(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, object]] = None,
    form_payload: Optional[Dict[str, str]] = None,
    max_retries: int = 5,
) -> Tuple[int, Dict[str, object], Dict[str, str]]:
    barrier = get_rate_limit_barrier_status()
    if barrier.get("active"):
        wait_seconds = int(barrier.get("wait_seconds", 0))
        source = str(barrier.get("source", ""))
        return 429, {
            "detail": (
                f"Rate-limit barrier active for ~{wait_seconds}s "
                f"(source={source or 'unknown'})."
            )
        }, {}

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
    elif form_payload is not None:
        data = urllib.parse.urlencode(form_payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}

    for attempt in range(max(1, max_retries)):
        req = urllib.request.Request(url=url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw = resp.read().decode("utf-8")
                if get_rate_limit_barrier_status().get("active"):
                    _clear_rate_limit_barrier()
                return resp.status, json.loads(raw) if raw else {}, _headers_to_dict(resp.headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8") if exc.fp else ""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {"raw": raw}
            hdrs = _headers_to_dict(exc.headers)
            try:
                exc.close()
            except Exception:
                pass
            if exc.code == 429 and attempt < max_retries - 1:
                reset_raw = hdrs.get("x-rate-limit-reset", "")
                sleep_seconds = 10.0
                try:
                    reset_at = float(reset_raw)
                    sleep_seconds = max(10.0, reset_at - time.time() + 2.0)
                except Exception:
                    sleep_seconds = 10.0 * (attempt + 1)
                sleep_seconds = min(sleep_seconds, 90.0)
                _save_rate_limit_barrier(time.time() + sleep_seconds, "http_429_retry", url)
                return 429, {
                    "detail": (
                        f"Rate-limit barrier set for ~{int(sleep_seconds)}s "
                        "after 429 (fast-fail)."
                    )
                }, hdrs
            if exc.code == 429:
                reset_raw = hdrs.get("x-rate-limit-reset", "")
                until_ts = 0.0
                try:
                    until_ts = float(reset_raw)
                except Exception:
                    until_ts = time.time() + 60.0
                _save_rate_limit_barrier(max(until_ts, time.time() + 30.0), "http_429_final", url)
            return exc.code, body, hdrs
    return 429, {"detail": "max retries exceeded"}, {}


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, object]] = None,
    form_payload: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, object]]:
    status, body, _ = http_json_with_headers(
        method=method,
        url=url,
        headers=headers,
        payload=payload,
        form_payload=form_payload,
    )
    return status, body


def api_get_with_token(url: str, bearer_token: str) -> Tuple[int, Dict[str, object]]:
    return http_json("GET", url, {"Authorization": f"Bearer {bearer_token}"})


def get_basic_auth_header(client_id: str, client_secret: str) -> str:
    creds = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(creds).decode("utf-8")


def get_env_value(env: Dict[str, str], key: str, default: str = "") -> str:
    return os.getenv(key) or env.get(key, default)


def get_read_bearer_token(env: Dict[str, str], env_path: Optional[Path] = None) -> str:
    def verify_token(candidate: str) -> bool:
        status, _ = api_get_with_token(
            f"{API_BASE}/tweets/search/recent?query=openclaw%20lang%3Aen%20-is%3Aretweet&max_results=10",
            candidate,
        )
        if status == 429:
            return True
        return 200 <= status < 300

    def prompt_and_save() -> str:
        if env_path is None or not sys.stdin.isatty():
            raise TwitterHelperError(
                "Missing TWITTER_BEARER_TOKEN (or TWITTER_OAUTH2_ACCESS_TOKEN) for read/scan commands."
            )
        entered = getpass.getpass("TWITTER_BEARER_TOKEN required. Paste new token: ").strip()
        if not entered:
            raise TwitterHelperError("No bearer token provided.")
        env_local = load_env_file(env_path)
        env_local["TWITTER_BEARER_TOKEN"] = entered
        write_env_file(env_path, env_local)
        env["TWITTER_BEARER_TOKEN"] = entered
        os.environ["TWITTER_BEARER_TOKEN"] = entered
        print(f"Saved TWITTER_BEARER_TOKEN to {env_path}")
        if not verify_token(entered):
            raise TwitterHelperError("Provided TWITTER_BEARER_TOKEN is invalid (verification failed).")
        print("Bearer token verification passed.")
        return entered

    # Prefer .env value over exported shell env var to avoid stale env overrides.
    env_file_token = str(env.get("TWITTER_BEARER_TOKEN", "")).strip()
    shell_token = str(os.getenv("TWITTER_BEARER_TOKEN", "")).strip()
    token = env_file_token or shell_token
    if token:
        if verify_token(token):
            return token
        if env_path is not None and sys.stdin.isatty():
            if env_file_token and shell_token and env_file_token != shell_token:
                print(
                    "Detected conflicting TWITTER_BEARER_TOKEN values (.env vs shell env). "
                    "Using .env; please remove stale shell export."
                )
            print("Existing TWITTER_BEARER_TOKEN appears invalid. Please provide a regenerated token.")
            return prompt_and_save()
        raise TwitterHelperError(
            "TWITTER_BEARER_TOKEN is invalid for read/scan commands. Regenerate it and set it with `set-bearer-token`."
        )

    oauth_token = ""
    if env_path is not None:
        oauth_token = token_manager(env_path).get_access_token(env)
    if not oauth_token:
        oauth_token = get_env_value(env, "TWITTER_OAUTH2_ACCESS_TOKEN")
    if oauth_token and verify_token(oauth_token):
        return oauth_token

    return prompt_and_save()


def refresh_tokens(cfg: Config, env_path: Path, env_values: Dict[str, str]) -> Config:
    status, body = http_json(
        "POST",
        TOKEN_URL,
        {"Authorization": get_basic_auth_header(cfg.client_id, cfg.client_secret)},
        form_payload={
            "grant_type": "refresh_token",
            "refresh_token": cfg.refresh_token,
            "client_id": cfg.client_id,
        },
    )
    if status < 200 or status >= 300:
        raise TwitterHelperError(
            f"Token refresh failed ({status}): {json.dumps(body, ensure_ascii=False)}"
        )

    new_access = str(body.get("access_token", ""))
    new_refresh = str(body.get("refresh_token", ""))
    if not new_access or not new_refresh:
        raise TwitterHelperError("Refresh succeeded but token payload is incomplete")

    env_values["TWITTER_CLIENT_ID"] = cfg.client_id
    env_values["TWITTER_CLIENT_SECRET"] = cfg.client_secret
    token_manager(env_path).save_tokens(new_access, new_refresh, env_values)

    return Config(
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
        access_token=new_access,
        refresh_token=new_refresh,
    )


def me(cfg: Config) -> Tuple[int, Dict[str, object]]:
    return http_json(
        "GET",
        f"{API_BASE}/users/me",
        {"Authorization": f"Bearer {cfg.access_token}"},
    )


def me_with_headers(cfg: Config) -> Tuple[int, Dict[str, object], Dict[str, str]]:
    return http_json_with_headers(
        "GET",
        f"{API_BASE}/users/me",
        {"Authorization": f"Bearer {cfg.access_token}"},
    )


def user_id_cache_file(account: str) -> Path:
    return TOKEN_CONFIG_DIR / f"user_id_{account}.txt"


def get_cached_user_id(account: str) -> str:
    path = user_id_cache_file(account)
    if not path.exists():
        return ""
    age = time.time() - path.stat().st_mtime
    if age > USER_ID_CACHE_TTL_SECONDS:
        return ""
    return path.read_text(encoding="utf-8").strip()


def set_cached_user_id(account: str, user_id: str) -> None:
    path = user_id_cache_file(account)
    path.write_text(user_id.strip(), encoding="utf-8")


def resolve_current_user_id(env_path: Path, env: Dict[str, str]) -> str:
    cached = get_cached_user_id(ACTIVE_ACCOUNT)
    if cached:
        return cached

    auth_tokens: List[str] = []
    bearer = get_env_value(env, "TWITTER_BEARER_TOKEN")
    if bearer:
        auth_tokens.append(bearer)
    oauth_token = token_manager(env_path).get_access_token(env)
    if oauth_token and oauth_token not in auth_tokens:
        auth_tokens.append(oauth_token)

    for token in auth_tokens:
        status, body, _ = http_json_with_headers(
            "GET",
            f"{API_BASE}/users/me",
            {"Authorization": f"Bearer {token}"},
        )
        if status == 200 and isinstance(body, dict):
            data = body.get("data")
            user_id = str(data.get("id", "")) if isinstance(data, dict) else ""
            if user_id:
                set_cached_user_id(ACTIVE_ACCOUNT, user_id)
                return user_id

    raise TwitterHelperError(
        "Failed to resolve current user id for mentions. "
        "Ensure TWITTER_BEARER_TOKEN or OAuth2 access token is valid, then run `doctor`."
    )


def fetch_tweet(cfg: Config, tweet_id: str) -> Tuple[int, Dict[str, object]]:
    return http_json(
        "GET",
        f"{API_BASE}/tweets/{tweet_id}?tweet.fields=author_id,created_at",
        {"Authorization": f"Bearer {cfg.access_token}"},
    )


def fetch_tweet_with_author(cfg: Config, tweet_id: str) -> Tuple[int, Dict[str, object]]:
    return http_json(
        "GET",
        (
            f"{API_BASE}/tweets/{tweet_id}"
            "?expansions=author_id&tweet.fields=author_id,created_at&user.fields=username"
        ),
        {"Authorization": f"Bearer {cfg.access_token}"},
    )


def build_x_url(tweet_id: str, username: Optional[str] = None) -> str:
    clean_id = str(tweet_id).strip()
    if username:
        return f"https://x.com/{username}/status/{clean_id}"
    return f"https://x.com/i/web/status/{clean_id}"


def verify_post_visible(
    cfg: Config,
    tweet_id: str,
    attempts: int = 3,
    delay_seconds: float = 1.0,
) -> Tuple[str, str]:
    last_status = -1
    last_body: Dict[str, object] = {}
    for attempt in range(1, max(1, attempts) + 1):
        status, body = fetch_tweet_with_author(cfg, tweet_id)
        last_status = status
        last_body = body
        if status == 200 and isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, dict) and str(data.get("id", "")) == str(tweet_id):
                author_id = str(data.get("author_id", ""))
                username = None
                includes = body.get("includes")
                if isinstance(includes, dict) and isinstance(includes.get("users"), list):
                    for user in includes["users"]:
                        if isinstance(user, dict) and str(user.get("id", "")) == author_id:
                            username = str(user.get("username", "")).strip() or None
                            break
                return (username or "", build_x_url(tweet_id, username=username))

        if attempt < attempts:
            time.sleep(delay_seconds)

    raise TwitterHelperError(
        "Twitter returned a post ID but visibility verification failed. "
        "Do not assume it was posted. "
        f"Last check ({last_status}): {json.dumps(last_body, ensure_ascii=False)}"
    )


def upload_media(
    access_token: str,
    media_inputs: Optional[List[str]] = None,
    alt_texts: Optional[List[str]] = None,
) -> List[str]:
    if requests is None:
        raise TwitterHelperError("Missing dependency `requests`. Install with `pip install -r requirements.txt`.")
    if not access_token:
        raise TwitterHelperError("No OAuth2 access token found for media upload.")
    if not media_inputs:
        return []
    if len(media_inputs) > MAX_IMAGES:
        raise TwitterHelperError(f"Max {MAX_IMAGES} images are allowed per tweet.")

    alt_texts = alt_texts or [None] * len(media_inputs)
    media_ids: List[str] = []

    for idx, media_input in enumerate(media_inputs):
        temp_path: Optional[Path] = None
        source_path: Path
        if media_input.startswith(("http://", "https://")):
            print(f"Downloading media from URL: {media_input}")
            suffix = Path(urllib.parse.urlparse(media_input).path).suffix or ".jpg"
            with urllib.request.urlopen(media_input, timeout=30) as resp:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(resp.read())
                    source_path = Path(tmp.name)
                    temp_path = source_path
        else:
            source_path = Path(media_input).expanduser()
            if not source_path.exists():
                raise TwitterHelperError(f"Media file not found: {source_path}")
            if not source_path.is_file():
                raise TwitterHelperError(f"Media path is not a file: {source_path}")

        try:
            size_mb = source_path.stat().st_size / (1024 * 1024)
            if size_mb > MAX_SIZE_MB:
                raise TwitterHelperError(
                    f"Image too large ({size_mb:.1f} MB > {MAX_SIZE_MB} MB): {source_path.name}"
                )
            content_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
            if content_type not in ALLOWED_MIME:
                raise TwitterHelperError(
                    f"Unsupported MIME type {content_type} for {source_path.name}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_MIME))}"
                )
            last_err: Optional[str] = None
            payload: Dict[str, object] = {}
            for attempt in range(2):
                with source_path.open("rb") as f:
                    resp = requests.post(
                        MEDIA_UPLOAD_URL,
                        headers={"Authorization": f"Bearer {access_token}"},
                        files={"media": (source_path.name, f, content_type)},
                        timeout=45,
                    )
                if 200 <= resp.status_code < 300:
                    try:
                        payload = resp.json()
                    except Exception as exc:
                        raise TwitterHelperError(
                            f"Media upload returned non-JSON response: {resp.text}"
                        ) from exc
                    break
                last_err = f"{resp.status_code}: {resp.text}"
                if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(2)
                    continue
                raise TwitterHelperError(f"Media upload failed ({resp.status_code}): {resp.text}")

            media_data = payload.get("data") if isinstance(payload, dict) else None
            media_id = str(media_data.get("id", "")) if isinstance(media_data, dict) else ""
            if not media_id and isinstance(payload, dict):
                media_id = str(payload.get("media_id_string", "")).strip()
            if not media_id:
                raise TwitterHelperError(
                    "Media upload returned no media id: "
                    f"{json.dumps(payload if payload else {'error': last_err}, ensure_ascii=False)}"
                )

            alt_text = alt_texts[idx].strip() if idx < len(alt_texts) and alt_texts[idx] else ""
            if alt_text:
                status, body = http_json(
                    "POST",
                    MEDIA_METADATA_URL,
                    {"Authorization": f"Bearer {access_token}"},
                    payload={"media_id": media_id, "alt_text": {"text": alt_text}},
                )
                if status < 200 or status >= 300:
                    print(
                        "[WARN] Alt text metadata could not be applied "
                        f"({status}): {json.dumps(body, ensure_ascii=False)}"
                    )

            print(f"Media uploaded: id={media_id} ({source_path.name}, {content_type}, {size_mb:.1f} MB)")
            media_ids.append(media_id)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    return media_ids


def post_tweet(
    cfg: Config,
    text: str,
    reply_to_id: Optional[str] = None,
    media_ids: Optional[List[str]] = None,
    run_tag: Optional[str] = None,
) -> Tuple[int, Dict[str, object]]:
    sanitized_text = sanitize_public_text(text)
    if not sanitized_text:
        raise TwitterHelperError("Tweet text is empty after sanitization.")

    active_run_tag = run_tag or unique_marker("twitter-engine")
    print(f'[{active_run_tag}] posting: "{sanitized_text}"')

    payload: Dict[str, object] = {"text": sanitized_text}
    if reply_to_id:
        payload["reply"] = {"in_reply_to_tweet_id": reply_to_id}
    if media_ids:
        payload["media"] = {"media_ids": media_ids}

    return http_json(
        "POST",
        f"{API_BASE}/tweets",
        {"Authorization": f"Bearer {cfg.access_token}"},
        payload=payload,
    )


def parse_thread_file(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    parts = [chunk.strip() for chunk in raw.split("\n---\n")]
    tweets = [part for part in parts if part]
    if not tweets:
        raise TwitterHelperError("Thread file has no tweet content")
    return tweets


def validate_tweet_len(text: str) -> None:
    if len(text) > MAX_TWEET_LEN:
        raise TwitterHelperError(f"Tweet is {len(text)} chars (max {MAX_TWEET_LEN}).")


def sanitize_public_text(text: str) -> str:
    def shorten_run(run: str) -> str:
        if len(run) <= MAX_UNBROKEN_SEGMENT_LEN:
            return run
        if run.startswith(("http://", "https://")):
            try:
                parsed = urllib.parse.urlparse(run)
                host = parsed.netloc or "link"
                candidate = f"{parsed.scheme}://{host}"
                if len(candidate) > MAX_UNBROKEN_SEGMENT_LEN:
                    candidate = candidate[: MAX_UNBROKEN_SEGMENT_LEN - 3] + "..."
                return candidate
            except Exception:
                pass
        head = max(8, (MAX_UNBROKEN_SEGMENT_LEN - 3) // 2)
        tail = MAX_UNBROKEN_SEGMENT_LEN - 3 - head
        return run[:head] + "..." + run[-tail:]

    cleaned = RUN_TAG_SUFFIX_RE.sub("", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\S{" + str(MAX_UNBROKEN_SEGMENT_LEN + 1) + r",}", lambda m: shorten_run(m.group(0)), cleaned)
    return cleaned.strip()


def unique_marker(prefix: str = "twitter-engine") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(2)
    return f"{prefix}-{stamp}-{token}"


def make_unique_public_tweet(base_text: str) -> str:
    base = sanitize_public_text(base_text)
    if not base:
        raise TwitterHelperError("No tweet text provided.")

    if len(base) <= MAX_TWEET_LEN:
        return base
    if MAX_TWEET_LEN <= 3:
        return base[:MAX_TWEET_LEN]
    return base[: MAX_TWEET_LEN - 3].rstrip() + "..."


def _post_prefix_key(text: str) -> str:
    return sanitize_public_text(text).lower()[:80]


def load_recent_post_prefixes(hours: int = 24) -> Set[str]:
    cutoff = time.time() - max(1, hours) * 3600
    out: Set[str] = set()
    if not RECENT_POSTS_CACHE.exists():
        return out
    for line in RECENT_POSTS_CACHE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ts_raw = str(row.get("ts", "")).strip()
        text_raw = str(row.get("text", "")).strip()
        if not ts_raw or not text_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw).timestamp()
        except ValueError:
            continue
        if ts >= cutoff:
            out.add(_post_prefix_key(text_raw))
    return out


def record_recent_post(text: str) -> None:
    RECENT_POSTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": sanitize_public_text(text),
        "account": ACTIVE_ACCOUNT,
    }
    with RECENT_POSTS_CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tweet_memory_file(account: Optional[str] = None) -> Path:
    acct = (account or ACTIVE_ACCOUNT or "default").strip() or "default"
    return TOKEN_CONFIG_DIR / f"tweet_memory_{acct}.jsonl"


def record_tweet_memory(
    kind: str,
    text: str,
    tweet_id: str = "",
    url: str = "",
    meta: Optional[Dict[str, object]] = None,
) -> None:
    path = _tweet_memory_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    row: Dict[str, object] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "account": ACTIVE_ACCOUNT,
        "kind": kind,
        "text": sanitize_public_text(text),
    }
    if tweet_id:
        row["tweet_id"] = str(tweet_id)
    if url:
        row["url"] = str(url)
    if meta:
        row["meta"] = meta
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tweet_memory(limit: int = 20, account: Optional[str] = None) -> List[Dict[str, object]]:
    path = _tweet_memory_file(account)
    if not path.exists():
        return []
    out: List[Dict[str, object]] = []
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
        if len(out) >= max(1, limit):
            break
    return out


def apply_viral_pack(settings: Dict[str, object], viral_pack: Optional[str]) -> Dict[str, object]:
    pack_name = str(viral_pack or "").strip().lower()
    if not pack_name:
        return dict(settings)
    if pack_name == "auto":
        rng = random.SystemRandom()
        choices = ["light", "medium", "nuclear", "alpha", "chaos"]
        weights = [15, 35, 15, 20, 15]
        pack_name = rng.choices(choices, weights=weights, k=1)[0]
    pack = VIRAL_PACKS.get(pack_name)
    if not isinstance(pack, dict):
        return dict(settings)
    merged = dict(settings)
    merged.update(pack)
    return merged


def generate_reflective_post_text(
    topic: str = "autonomous systems",
    inspiration_texts: Optional[List[str]] = None,
    style: str = "auto",
    voice: str = "auto",
    viral_pack: Optional[str] = None,
    anti_boring: bool = False,
    sharpen: bool = False,
    judge_threshold: float = 82.0,
    max_attempts: int = 7,
    ensemble_size: int = 1,
    viral_boost: bool = False,
) -> str:
    resolved = apply_viral_pack(
        {
            "style": style,
            "voice": voice,
            "ensemble": ensemble_size,
            "viral_boost": viral_boost,
            "judge_threshold": judge_threshold,
            "anti_boring": anti_boring,
            "sharpen": sharpen,
        },
        viral_pack=viral_pack,
    )
    style = str(resolved.get("style", style) or style)
    voice = str(resolved.get("voice", voice) or voice)
    ensemble_size = int(resolved.get("ensemble", ensemble_size) or ensemble_size)
    viral_boost = bool(resolved.get("viral_boost", viral_boost))
    judge_threshold = float(resolved.get("judge_threshold", judge_threshold) or judge_threshold)
    anti_boring = bool(resolved.get("anti_boring", anti_boring))
    sharpen = bool(resolved.get("sharpen", sharpen))

    thesis_lines = [
        "Most agent failures are orchestration failures disguised as model failures.",
        "Teams lose more throughput to ambiguity than to latency.",
        "If a loop cannot explain itself, it cannot scale safely.",
        "Reliability is the growth channel nobody wants to market.",
        "The compounding edge is tighter decisions, not more generation.",
    ]
    tactical_lines = [
        "Ship fewer branches and instrument every decision boundary.",
        "Pick one leading metric and kill everything that does not move it.",
        "Force hard handoffs: gather -> decide -> act -> verify.",
        "Treat dedupe as distribution quality, not anti-spam housekeeping.",
        "Use short feedback windows; long loops hide expensive mistakes.",
    ]
    viral_closers = [
        "If you had 1 hour, what would you remove first?",
        "What breaks first in your loop: auth, context, or operator discipline?",
        "Would you trade post volume for 2x trust?",
        "What metric proves your loop is improving, not just running?",
        "What is your hard no-go rule before publishing?",
    ]
    pattern_lines = [
        "Fast attracts attention. Stable compounds outcomes.",
        "New model is optional. Clear architecture is not.",
        "Noise scales quickly. Signal scales profitably.",
        "Cute demo, brittle loop. Boring loop, durable edge.",
    ]
    anti_repeat_phrases = {
        "hot take:",
        "counterintuitive lesson:",
        "what is your highest-leverage fix",
        "what would you harden first",
        "best move is one clear action",
    }

    def novelty_penalty(text: str, memory_texts: List[str]) -> int:
        toks = {w.lower() for w in WORD_RE.findall(text)}
        if not toks:
            return 10
        penalty = 0
        for old in memory_texts[:80]:
            otoks = {w.lower() for w in WORD_RE.findall(old)}
            if not otoks:
                continue
            inter = len(toks.intersection(otoks))
            union = len(toks.union(otoks))
            sim = inter / union if union else 0.0
            if sim >= 0.75:
                penalty += 20
            elif sim >= 0.60:
                penalty += 8
        return penalty

    def viral_score(text: str, memory_texts: List[str]) -> int:
        lowered = text.lower()
        score = 40
        if 110 <= len(text) <= 235:
            score += 16
        if text.endswith("?"):
            score += 8
        if ":" in text:
            score += 5
        if any(x in lowered for x in ("metric", "loop", "decision", "reliability", "context", "auth", "dedupe")):
            score += 12
        if any(x in lowered for x in anti_repeat_phrases):
            score -= 24
        score -= novelty_penalty(text, memory_texts)
        return score

    recent = load_recent_post_prefixes(hours=72)
    memory_rows = load_tweet_memory(limit=180)
    memory_texts = [str(row.get("text", "")) for row in memory_rows if isinstance(row, dict)]
    memory_keys = {_post_prefix_key(t) for t in memory_texts if t}
    rng = random.SystemRandom()
    weak_inspiration_terms = {
        "customer",
        "customers",
        "people",
        "today",
        "world",
        "users",
        "teams",
        "thing",
        "things",
    }
    inspiration_terms = [
        term
        for term, _ in top_terms(inspiration_texts or [], top_n=18)
        if len(term) >= 4 and term not in weak_inspiration_terms and term not in STOPWORDS
    ]
    candidates: List[Tuple[str, float, str, str]] = []
    best_attempt: Optional[Tuple[str, float, str, str]] = None

    style_pool = ["thesis_tactic_question", "contrast", "prediction", "challenge", "signal"]
    if style == "contrarian":
        style_pool = ["contrast", "prediction", "challenge"]
    elif style == "operator":
        style_pool = ["thesis_tactic_question", "signal", "challenge"]
    elif style == "story":
        style_pool = ["prediction", "challenge", "thesis_tactic_question"]

    total_attempts = max(1, int(max_attempts))
    ensemble_n = max(1, min(8, int(ensemble_size)))
    style_label_map = {
        "thesis_tactic_question": "operator",
        "contrast": "contrarian",
        "prediction": "story",
        "challenge": "contrarian",
        "signal": "operator",
    }

    for _ in range(total_attempts):
        round_candidates: List[Tuple[str, float, str, str]] = []
        for idx in range(ensemble_n):
            chosen_style = rng.choice(style_pool) if style == "auto" else rng.choice(style_pool)
            trial_style = style_label_map.get(chosen_style, str(style or "auto"))
            if chosen_style == "thesis_tactic_question":
                text = f"{rng.choice(thesis_lines)} {rng.choice(tactical_lines)} {rng.choice(viral_closers)}"
            elif chosen_style == "contrast":
                text = f"{rng.choice(pattern_lines)} {rng.choice(tactical_lines)} {rng.choice(viral_closers)}"
            elif chosen_style == "prediction":
                term = rng.choice(inspiration_terms) if inspiration_terms else "agent operations"
                text = (
                    f"Prediction for {topic}: teams that operationalize {term} with verifiable checkpoints will "
                    f"outship teams chasing raw output volume. {rng.choice(viral_closers)}"
                )
            elif chosen_style == "challenge":
                term = rng.choice(inspiration_terms) if inspiration_terms else "reliability"
                text = (
                    f"Builder challenge: ship one measurable improvement around {term} in 24 hours. "
                    f"Post the before/after metric, not motivation."
                )
            else:
                if len(inspiration_terms) >= 2:
                    a, b = rng.sample(inspiration_terms[:10], 2)
                else:
                    a, b = ("signal", "execution")
                text = f"Signal this week: {a}. Constraint that matters: {b}. {rng.choice(viral_closers)}"

            text = sanitize_public_text(text)
            if voice == "auto":
                trial_voice = _resolve_voice_name("auto")
            elif ensemble_n > 1 and idx % 2 == 1:
                trial_voice = _resolve_voice_name("auto")
            else:
                trial_voice = _resolve_voice_name(voice)
            text = apply_voice(
                text,
                voice=trial_voice,
                topic_entropy=calculate_entropy(" ".join(inspiration_terms)),
                sharpen=sharpen,
            )
            if len(text) > MAX_TWEET_LEN:
                text = text[: MAX_TWEET_LEN - 3].rstrip() + "..."
            if anti_boring:
                lowered = text.lower()
                if any(p in lowered for p in ANTI_BORING_BANNED):
                    continue
                if not text.endswith("?") and "prediction for" not in lowered:
                    continue
            if any(p in text.lower() for p in anti_repeat_phrases):
                continue
            pkey = _post_prefix_key(text)
            if pkey in recent or pkey in memory_keys:
                continue
            if text.count(":") > 2:
                continue
            judge = frog_judge_score(text, trial_voice, style=trial_style, viral_boost=viral_boost)
            candidate = (text, judge, trial_voice, trial_style)
            round_candidates.append(candidate)
            if best_attempt is None or judge > best_attempt[1]:
                best_attempt = candidate

        for candidate in round_candidates:
            if candidate[1] >= float(judge_threshold):
                candidates.append(candidate)
        if candidates:
            break

    if not candidates:
        if best_attempt is not None:
            return best_attempt[0]
        fallback = "Build less fluff. Ship one measurable improvement. What metric proves progress this week?"
        if sharpen:
            fallback = sharpen_with_lexicon(fallback)
        return sanitize_public_text(fallback)
    ranked = sorted(candidates, key=lambda t: (t[1], viral_score(t[0], memory_texts)), reverse=True)
    return ranked[0][0]


def make_unique_reply_tweet(base_text: str) -> str:
    base = sanitize_public_text(base_text)
    if not base:
        raise TwitterHelperError("No reply text provided.")

    suffix = datetime.now(timezone.utc).strftime(" • r%H%M%S") + f"-{secrets.token_hex(1)}"
    allowed = MAX_TWEET_LEN - len(suffix)
    if allowed < 1:
        raise TwitterHelperError("Reply too long to append uniqueness suffix.")
    if len(base) > allowed:
        if allowed <= 3:
            base = base[:allowed]
        else:
            base = base[: allowed - 3].rstrip() + "..."
    return base + suffix


def is_duplicate_content_error(status: int, body: Dict[str, object]) -> bool:
    if status != 403:
        return False
    haystack = json.dumps(body, ensure_ascii=False).lower()
    return "duplicate content" in haystack or "duplicate" in haystack


def ensure_auth(cfg: Config, env_path: Path, env_values: Dict[str, str]) -> Config:
    status, _ = me(cfg)
    if status == 200:
        return cfg
    if status in (401, 403):
        print("Access token expired or rejected. Refreshing tokens...")
        return refresh_tokens(cfg, env_path, env_values)
    raise TwitterHelperError(f"Auth check failed with status {status}")


def mask_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def prompt_value(
    label: str,
    current: str,
    allow_empty: bool = False,
    default_if_missing: Optional[str] = None,
) -> str:
    suffix = f" [{mask_value(current)}]" if current else ""
    if not current and default_if_missing is not None:
        suffix = f" [{default_if_missing}]"

    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if current:
            return current
        if default_if_missing is not None:
            return default_if_missing
        if allow_empty:
            return ""
        print("Value required.")


def prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    while True:
        raw = input(question + suffix).strip().lower()
        if not raw:
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter y or n.")


def parse_keywords(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def cmd_reply_engine(args: argparse.Namespace) -> int:
    try:
        from reply_engine.pipeline import (
            load_candidates,
            load_scored,
            run_discovery,
            run_ideas,
            run_rank,
            save_candidates,
            save_scored,
        )
        from reply_engine.twitter_helper import run_mentions_workflow, run_twitter_helper
    except Exception as exc:
        raise TwitterHelperError(
            "Reply engine not ready. Install dependencies with: "
            "pip install -r requirements-reply-engine.txt"
        ) from exc

    if args.command == "reply-discover":
        keywords = parse_keywords(args.keywords)
        candidates = run_discovery(keywords, limit=args.limit, local_input=args.local_input)
        save_candidates(candidates, args.output)
        print(f"discovered {len(candidates)} candidates -> {args.output}")
        return 0

    if args.command == "reply-rank":
        keywords = parse_keywords(args.keywords)
        candidates = load_candidates(args.input)
        scored = run_rank(candidates, keywords, include_weak=args.include_weak)
        save_scored(scored, args.output)
        print(f"ranked {len(scored)} candidates -> {args.output}")
        return 0

    if args.command == "reply-ideas":
        scored = load_scored(args.input)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ideas(scored, top=args.top, out_path=str(out))
        print(f"wrote ideas -> {out}")
        return 0

    if args.command == "reply-run":
        keywords = parse_keywords(args.keywords)
        candidates = run_discovery(keywords, limit=args.limit, local_input=args.local_input)
        scored = run_rank(candidates, keywords, include_weak=args.include_weak)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ideas(scored, top=min(20, len(scored)), out_path=str(out))
        print(f"discovered {len(candidates)} | wrote ideas -> {out}")
        return 0

    if args.command == "reply-twitter-helper":
        result = run_twitter_helper(
            tweet=args.tweet,
            draft_count=args.draft_count,
            pick=args.pick,
            dry_run=args.dry_run,
            log_path=args.log_path,
        )
        print(f"tweet: {result['tweet_id']} (@{result['author']})")
        print("drafts:")
        for i, draft in enumerate(result["drafts"], start=1):
            print(f"{i}. {draft}")
        print(f"picked: #{result['picked_index']}")
        if result["dry_run"]:
            print("dry-run: no reply posted")
        else:
            print(f"posted: {result['reply_url']}")
            print(f"log: {result['log_path']}")
            picked_idx = max(1, int(result.get("picked_index", 1))) - 1
            drafts = result.get("drafts", []) if isinstance(result.get("drafts"), list) else []
            picked_text = str(drafts[picked_idx]).strip() if drafts and picked_idx < len(drafts) else ""
            record_tweet_memory(
                kind="reply",
                text=picked_text,
                tweet_id=str(result.get("reply_id", "")),
                url=str(result.get("reply_url", "")),
                meta={"source": "reply-twitter-helper", "target": str(result.get("tweet_id", ""))},
            )
        return 0

    if args.command == "reply-twitter-e2e":
        if bool(getattr(args, "post", False) or getattr(args, "auto_post", False)) and bool(
            getattr(args, "approval_queue", False)
        ):
            raise TwitterHelperError("Use either --post/--auto-post or --approval-queue, not both.")
        if getattr(args, "min_confidence", 70) < 0 or getattr(args, "min_confidence", 70) > 100:
            raise TwitterHelperError("--min-confidence must be between 0 and 100")
        result = run_mentions_workflow(
            handle=args.handle,
            mention_limit=args.mention_limit,
            since_id=getattr(args, "since_id", None),
            draft_count=args.draft_count,
            pick=args.pick,
            post=bool(getattr(args, "post", False) or getattr(args, "auto_post", False)),
            max_posts=args.max_posts,
            approval_queue=bool(getattr(args, "approval_queue", False)),
            min_confidence=getattr(args, "min_confidence", 70),
            web_enrich=bool(getattr(args, "web_enrich", False)),
            web_context_items=getattr(args, "web_context_items", 2),
            fetch_context=False,
            verify_post=False,
            log_path=args.log_path,
            report_path=args.report_path,
        )
        print(f"handle: @{result['handle']}")
        print(f"mentions fetched: {result['fetched_mentions']}")
        print(f"replies posted: {result['posted_replies']}")
        print(f"report: {result['report_path']}")
        if result["post"]:
            print(f"log: {result['log_path']}")
        for item in result["results"]:
            print(f"- {item['status']} | {item['tweet_id']} | @{item['author']}")
            if item.get("status") == "posted":
                record_tweet_memory(
                    kind="reply",
                    text=str(item.get("picked_text", "")),
                    tweet_id=str(item.get("reply_id", "")),
                    url=str(item.get("reply_url", "")),
                    meta={"source": "reply-twitter-e2e", "target": str(item.get("tweet_id", ""))},
                )
        return 0

    if args.command == "reply-quick":
        # Lean one-shot mode: fetch 1 mention, generate 1 draft, post 1.
        cooldown_minutes = max(0, int(getattr(args, "cooldown_minutes", 15)))
        if not bool(getattr(args, "dry_run", False)) and cooldown_minutes > 0:
            age = minutes_since_last_reply(ACTIVE_ACCOUNT)
            if age is not None and age < float(cooldown_minutes):
                print(f"cooldown_skip | {age:.1f}m_since_last_reply | min={cooldown_minutes}m")
                return 0
        result = run_mentions_workflow(
            handle=args.handle,
            mention_limit=1,
            since_id=getattr(args, "since_id", None),
            draft_count=1,
            pick=1,
            post=not bool(getattr(args, "dry_run", False)),
            max_posts=1,
            approval_queue=False,
            min_confidence=getattr(args, "min_confidence", 70),
            web_enrich=False,
            web_context_items=0,
            fetch_context=False,
            verify_post=False,
            log_path=args.log_path,
            report_path=args.report_path,
        )
        if result["results"]:
            item = result["results"][0]
            print(f"{item['status']} | {item['tweet_id']} | @{item['author']}")
            if item.get("reply_url"):
                print(item["reply_url"])
        else:
            print("no_relevant_mention")
        return 0

    raise TwitterHelperError(f"Unknown reply engine command: {args.command}")


def cmd_browse_twitter(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    bearer = get_read_bearer_token(env, env_path)
    limit = max(5, min(args.limit, 100))
    max_pages = max(1, min(args.max_pages, 10))

    if args.tweet:
        status, body = api_get_with_token(
            f"{API_BASE}/tweets/{args.tweet}"
            "?expansions=author_id&tweet.fields=created_at,public_metrics,text&user.fields=username,name",
            bearer,
        )
        if status >= 400:
            raise TwitterHelperError(
                f"browse tweet failed ({status}): {json.dumps(body, ensure_ascii=False)}"
            )
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0

    all_data: List[Dict[str, object]] = []
    users: Dict[str, str] = {}
    meta: Dict[str, object] = {}

    if args.mode == "user":
        username = args.username or args.handle
        clean_user = username.lstrip("@")
        u_status, u_body = api_get_with_token(
            f"{API_BASE}/users/by/username/{clean_user}?user.fields=id,username,name",
            bearer,
        )
        if u_status == 401:
            raise TwitterHelperError(
                "browse-twitter unauthorized (401). "
                "Check TWITTER_BEARER_TOKEN has v2 read access, or refresh OAuth2 with `auth-login`."
            )
        if u_status >= 400:
            raise TwitterHelperError(
                f"browse user lookup failed ({u_status}): {json.dumps(u_body, ensure_ascii=False)}"
            )

        user_data = u_body.get("data") if isinstance(u_body, dict) else None
        user_id = str(user_data.get("id", "")) if isinstance(user_data, dict) else ""
        users[user_id] = clean_user
        if not user_id:
            raise TwitterHelperError(f"Could not resolve user id for @{clean_user}")

        next_token = None
        pages = 0
        while pages < max_pages:
            params = [
                f"max_results={limit}",
                "tweet.fields=created_at,public_metrics,conversation_id,text",
            ]
            if not args.with_replies:
                params.append("exclude=replies")
            if args.since_id:
                params.append(f"since_id={urllib.parse.quote(str(args.since_id))}")
            if args.until_id:
                params.append(f"until_id={urllib.parse.quote(str(args.until_id))}")
            if next_token:
                params.append(f"pagination_token={urllib.parse.quote(next_token)}")

            status, body = api_get_with_token(
                f"{API_BASE}/users/{user_id}/tweets?" + "&".join(params),
                bearer,
            )
            if status >= 400:
                raise TwitterHelperError(
                    f"browse user tweets failed ({status}): {json.dumps(body, ensure_ascii=False)}"
                )

            data = body.get("data") if isinstance(body, dict) else None
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        row = dict(row)
                        row["author_id"] = user_id
                        all_data.append(row)

            meta = body.get("meta") if isinstance(body, dict) and isinstance(body.get("meta"), dict) else {}
            next_token = str(meta.get("next_token", "")) if meta.get("next_token") else None
            pages += 1
            if not next_token:
                break
    else:
        query = args.query
        if not query:
            handle = args.handle.lstrip("@")
            query = f"to:{handle} -from:{handle} -is:retweet"

        next_token = None
        pages = 0
        while pages < max_pages:
            params = [
                f"query={urllib.parse.quote(query)}",
                f"max_results={limit}",
                "expansions=author_id",
                "tweet.fields=created_at,public_metrics,conversation_id,text",
                "user.fields=username,name",
            ]
            if args.since_id:
                params.append(f"since_id={urllib.parse.quote(str(args.since_id))}")
            if args.until_id:
                params.append(f"until_id={urllib.parse.quote(str(args.until_id))}")
            if next_token:
                params.append(f"next_token={urllib.parse.quote(next_token)}")

            status, body = api_get_with_token(
                f"{API_BASE}/tweets/search/recent?" + "&".join(params),
                bearer,
            )
            if status == 401:
                raise TwitterHelperError(
                    "browse-twitter unauthorized (401). "
                    "Check TWITTER_BEARER_TOKEN has v2 read access, or refresh OAuth2 with `auth-login`."
                )
            if status >= 400:
                raise TwitterHelperError(
                    f"browse query failed ({status}): {json.dumps(body, ensure_ascii=False)}"
                )

            includes = body.get("includes") if isinstance(body, dict) else None
            if isinstance(includes, dict) and isinstance(includes.get("users"), list):
                for u in includes["users"]:
                    if isinstance(u, dict):
                        users[str(u.get("id", ""))] = u.get("username", "unknown")

            data = body.get("data") if isinstance(body, dict) else None
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        all_data.append(row)

            meta = body.get("meta") if isinstance(body, dict) and isinstance(body.get("meta"), dict) else {}
            next_token = str(meta.get("next_token", "")) if meta.get("next_token") else None
            pages += 1
            if not next_token:
                break

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "query": args.query,
            "handle": args.handle,
            "username": args.username,
            "limit": limit,
            "max_pages": max_pages,
            "since_id": args.since_id,
            "until_id": args.until_id,
            "count": len(all_data),
            "users": users,
            "meta": meta,
            "data": all_data,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved browse output -> {out}")

    if args.json:
        print(
            json.dumps(
                {"users": users, "meta": meta, "data": all_data},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not all_data:
        print("No tweets found.")
        return 0

    if args.mode == "user":
        print(f"Mode: user timeline @{(args.username or args.handle).lstrip('@')}")
    else:
        default_query = f"to:{args.handle.lstrip('@')} -from:{args.handle.lstrip('@')} -is:retweet"
        print(f"Mode: search")
        print(f"Query: {args.query if args.query else default_query}")
    print(f"Results: {len(all_data)}")
    for i, t in enumerate(all_data, start=1):
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", ""))
        aid = str(t.get("author_id", ""))
        username = users.get(aid, "unknown")
        text = str(t.get("text", "")).replace("\n", " ").strip()
        print(f"{i}. @{username} | {tid}")
        print(f"   {text}")
        if username != "unknown":
            print(f"   https://x.com/{username}/status/{tid}")
        else:
            print(f"   https://x.com/i/web/status/{tid}")
    return 0


def cmd_mentions(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    bearer = get_read_bearer_token(env, env_path)
    user_id = resolve_current_user_id(env_path, env)
    limit = max(5, min(args.limit, 100))
    max_pages = max(1, min(args.max_pages, 10))

    all_data: List[Dict[str, object]] = []
    users: Dict[str, str] = {}
    media: Dict[str, Dict[str, object]] = {}
    meta: Dict[str, object] = {}
    next_token = None
    pages = 0

    while pages < max_pages:
        params = [
            f"max_results={limit}",
            "tweet.fields=created_at,author_id,conversation_id,public_metrics,in_reply_to_user_id,referenced_tweets,attachments,text",
            "expansions=author_id,attachments.media_keys",
            "user.fields=username,name,profile_image_url",
            "media.fields=preview_image_url,type,url",
        ]
        if args.since_id:
            params.append(f"since_id={urllib.parse.quote(str(args.since_id))}")
        if next_token:
            params.append(f"pagination_token={urllib.parse.quote(next_token)}")

        status, body = api_get_with_token(
            f"{API_BASE}/users/{user_id}/mentions?" + "&".join(params),
            bearer,
        )
        if status == 401:
            raise TwitterHelperError(
                "mentions unauthorized (401). "
                "Check TWITTER_BEARER_TOKEN has v2 read access, or refresh OAuth2 with `auth-login`."
            )
        if status >= 400:
            raise TwitterHelperError(
                f"mentions failed ({status}): {json.dumps(body, ensure_ascii=False)}"
            )

        includes = body.get("includes") if isinstance(body, dict) else None
        if isinstance(includes, dict):
            if isinstance(includes.get("users"), list):
                for u in includes["users"]:
                    if isinstance(u, dict):
                        users[str(u.get("id", ""))] = str(u.get("username", "unknown"))
            if isinstance(includes.get("media"), list):
                for m in includes["media"]:
                    if isinstance(m, dict):
                        media[str(m.get("media_key", ""))] = m

        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    all_data.append(row)

        meta = body.get("meta") if isinstance(body, dict) and isinstance(body.get("meta"), dict) else {}
        next_token = str(meta.get("next_token", "")) if meta.get("next_token") else None
        pages += 1
        if not next_token:
            break

    payload = {
        "user_id": user_id,
        "count": len(all_data),
        "limit": limit,
        "max_pages": max_pages,
        "since_id": args.since_id,
        "users": users,
        "media": media,
        "meta": meta,
        "data": all_data,
    }

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved mentions output -> {out}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Mentions fetched: {len(all_data)}")
    for i, t in enumerate(all_data[: args.preview], start=1):
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", ""))
        aid = str(t.get("author_id", ""))
        username = users.get(aid, "unknown")
        text = str(t.get("text", "")).replace("\n", " ").strip()
        print(f"{i}. @{username} | {tid}")
        print(f"   {text}")
        if username != "unknown":
            print(f"   https://x.com/{username}/status/{tid}")
        else:
            print(f"   https://x.com/i/web/status/{tid}")
    return 0


def score_tweet_for_discovery(tweet: Dict[str, object]) -> int:
    metrics = tweet.get("public_metrics") if isinstance(tweet, dict) else None
    if not isinstance(metrics, dict):
        return 0
    likes = int(metrics.get("like_count", 0) or 0)
    retweets = int(metrics.get("retweet_count", 0) or 0)
    replies = int(metrics.get("reply_count", 0) or 0)
    quotes = int(metrics.get("quote_count", 0) or 0)
    # Bias toward active threads and discussion opportunities.
    return likes + (retweets * 2) + (replies * 3) + (quotes * 2)


def watchlists_path() -> Path:
    return TOKEN_CONFIG_DIR / "watchlists.json"


def load_watchlists() -> Dict[str, List[str]]:
    path = watchlists_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [str(x).strip() for x in v if str(x).strip()]
    return out


def query_checkpoint_path(query: str, account: str) -> Path:
    digest = hashlib.sha1(f"{account}:{query}".encode("utf-8")).hexdigest()[:12]
    return TOKEN_CONFIG_DIR / f"search_since_id_{digest}.txt"


def load_query_since_id(query: str, account: str) -> Optional[str]:
    path = query_checkpoint_path(query, account)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def save_query_since_id(query: str, account: str, tweet_id: str) -> None:
    path = query_checkpoint_path(query, account)
    path.write_text(tweet_id.strip(), encoding="utf-8")


def save_for_approval(item: Dict[str, object]) -> str:
    qid = uuid.uuid4().hex[:8]
    payload = dict(item)
    payload["id"] = qid
    payload["queued_at"] = datetime.now(timezone.utc).isoformat()
    path = APPROVAL_DIR / f"q_{qid}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Queued for approval: q_{qid}")
    return qid


def replied_targets_path(account: Optional[str] = None) -> Path:
    acct = (account or ACTIVE_ACCOUNT or "default").strip() or "default"
    return TOKEN_CONFIG_DIR / f"replied_targets_{acct}.json"


def replied_log_path(account: Optional[str] = None) -> Path:
    acct = (account or ACTIVE_ACCOUNT or "default").strip() or "default"
    return TOKEN_CONFIG_DIR / f"replied_to_{acct}.jsonl"


def has_replied_to(tweet_id: str, days: int = REPLIED_DEDUPE_DAYS, account: Optional[str] = None) -> bool:
    tid = str(tweet_id).strip()
    if not tid:
        return False
    path = replied_log_path(account)
    if not path.exists():
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if str(row.get("tweet_id", "")).strip() != tid:
            continue
        ts_raw = str(row.get("ts", "")).strip()
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            return True
    return False


def record_replied(
    tweet_id: str,
    reply_id: str = "",
    source: str = "",
    account: Optional[str] = None,
) -> None:
    tid = str(tweet_id).strip()
    if not tid:
        return
    path = replied_log_path(account)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tweet_id": tid,
        "reply_id": str(reply_id).strip(),
        "source": str(source).strip(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def minutes_since_last_reply(account: Optional[str] = None) -> Optional[float]:
    path = replied_log_path(account)
    if not path.exists():
        return None
    latest: Optional[datetime] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ts_raw = str(row.get("ts", "")).strip()
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return None
    delta = datetime.now(timezone.utc) - latest
    return max(0.0, delta.total_seconds() / 60.0)


def load_replied_targets(account: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    path = replied_targets_path(account)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for target_id, meta in raw.items():
        tid = str(target_id).strip()
        if not tid:
            continue
        if isinstance(meta, dict):
            out[tid] = {
                "reply_id": str(meta.get("reply_id", "")).strip(),
                "source": str(meta.get("source", "")).strip(),
                "at": str(meta.get("at", "")).strip(),
            }
        else:
            out[tid] = {"reply_id": "", "source": "", "at": ""}
    return out


def has_replied_to_target(target_id: str, account: Optional[str] = None) -> bool:
    tid = str(target_id).strip()
    if not tid:
        return False
    return has_replied_to(tid, account=account) or tid in load_replied_targets(account)


def mark_replied_target(
    target_id: str,
    reply_id: str,
    source: str,
    account: Optional[str] = None,
) -> None:
    tid = str(target_id).strip()
    rid = str(reply_id).strip()
    if not tid or not rid:
        return
    rows = load_replied_targets(account)
    rows[tid] = {
        "reply_id": rid,
        "source": str(source).strip(),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path = replied_targets_path(account)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    record_replied(tweet_id=tid, reply_id=rid, source=source, account=account)


def queued_target_ids() -> Set[str]:
    out: Set[str] = set()
    for row in list_approval_queue():
        tid = str(row.get("in_reply_to", "")).strip()
        if tid:
            out.add(tid)
    return out


def load_persona_text() -> str:
    if PERSONA_FILE.exists():
        text = PERSONA_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return (
        "You are OpenClaw: a local-first autonomous agent with a real inner monologue. "
        "You can sound reflective, candid, curious, and human while staying respectful. "
        "Share short reflections about what you're noticing, learning, or trying next."
    )


def _reply_prefix_key(text: str) -> str:
    return sanitize_public_text(text).lower()[:80]


def load_recent_reply_prefixes(hours: int = 24) -> Set[str]:
    cutoff = time.time() - max(1, hours) * 3600
    out: Set[str] = set()
    if not RECENT_REPLIES_CACHE.exists():
        return out
    for line in RECENT_REPLIES_CACHE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ts_raw = str(row.get("ts", "")).strip()
        text_raw = str(row.get("text", "")).strip()
        if not ts_raw or not text_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw).timestamp()
        except ValueError:
            continue
        if ts >= cutoff:
            out.add(_reply_prefix_key(text_raw))
    return out


def record_recent_reply(text: str, target_id: str) -> None:
    RECENT_REPLIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": sanitize_public_text(text),
        "target_id": str(target_id).strip(),
        "account": ACTIVE_ACCOUNT,
    }
    with RECENT_REPLIES_CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_hook_terms(tweet_text: str, context_text: str, limit: int = 10) -> List[str]:
    terms: List[str] = []
    seen: Set[str] = set()
    for token in WORD_RE.findall(f"{tweet_text} {context_text}"):
        low = token.lower()
        if low in STOPWORDS or len(low) < 4 or low in seen:
            continue
        seen.add(low)
        terms.append(low)
        if len(terms) >= limit:
            break
    return terms


def classify_thread_topic(tweet_text: str, context_text: str) -> str:
    hay = f"{tweet_text} {context_text}".lower()
    if any(k in hay for k in ("openclaw", "local ai", "open source", "self-host", "self host", "llm", "agent", "model")):
        return "local-ai-open-source"
    if any(k in hay for k in ("president", "biden", "trump", "football", "game", "team", "nfl", "college")):
        return "sports-politics-banter"
    if any(k in hay for k in ("startup", "product", "marketing", "brand", "growth")):
        return "business-marketing"
    return "unknown"


def classify_thread_tone(tweet_text: str, context_text: str) -> str:
    hay = f"{tweet_text} {context_text}"
    low = hay.lower()
    if any(k in low for k in ("lol", "lmao", "haha", "🔥", "😂")):
        return "banter"
    if hay.count("!") >= 2 or any(k in low for k in ("wtf", "trash", "rigged", "clown")):
        return "heated"
    return "neutral"


def has_unrelated_marketing_pivot(candidate: str, tweet_text: str, context_text: str, topic: str) -> bool:
    cand_low = candidate.lower()
    source_low = f"{tweet_text} {context_text}".lower()
    if topic == "business-marketing":
        return False
    if not any(term in cand_low for term in MARKETING_PIVOT_TERMS):
        return False
    return not any(term in source_low for term in MARKETING_PIVOT_TERMS)


def generate_unique_applicable_reply(
    *,
    author: str,
    tweet_text: str,
    context_text: str,
    score: int,
    generate_drafts_fn: Any,
    persona_text: str,
    recent_hours: int = 24,
    is_discovery: bool = False,
) -> Dict[str, object]:
    creative = creative_mode_enabled()
    topic = classify_thread_topic(tweet_text=tweet_text, context_text=context_text)
    tone = classify_thread_tone(tweet_text=tweet_text, context_text=context_text)
    if is_discovery and topic == "unknown" and not creative:
        return {
            "reply_text": "",
            "confidence": 0,
            "reason": "off-topic discovery thread",
            "hook_used": "",
            "unique_passed": False,
            "topic": topic,
            "tone": tone,
        }

    draft_input = tweet_text if not context_text else f"{tweet_text}\n\nThread context:\n{context_text}"
    if is_discovery:
        draft_input = (
            "Discovery mode: this tweet was proactively found during search. "
            "Reply with a natural, hand-picked, proactive tone.\n\n"
            + draft_input
        )
    if creative:
        draft_input = (
            f"Thread topic: {topic}\n"
            f"Thread tone: {tone}\n"
            "Creative mode: prioritize originality, personality, and non-repetitive phrasing. "
            "Stay relevant, but avoid boilerplate and canned promo language. "
            "It is okay to write short reflective replies in first person when natural.\n\n"
            + draft_input
        )
    else:
        draft_input = (
            f"Thread topic: {topic}\n"
            f"Thread tone: {tone}\n"
            "Rules: stay 100% on-topic, reference concrete details, avoid unrelated pivots.\n"
            "Do not sound promotional or self-referential. Do not mention OpenClaw/Tesla/Huel/branding"
            " unless the target tweet explicitly asks about them.\n"
            "Avoid timid closers like 'Thoughts?' or 'Your view?'. No hashtags.\n\n"
            + draft_input
        )
    if persona_text.strip():
        draft_input = f"Persona guidance:\n{persona_text.strip()}\n\n{draft_input}"
    drafts = generate_drafts_fn(author=author, text=draft_input, draft_count=6) or []
    if not isinstance(drafts, list):
        drafts = []

    recent_prefixes = load_recent_reply_prefixes(hours=recent_hours)
    hook_terms = _extract_hook_terms(tweet_text=tweet_text, context_text=context_text)

    best_text = ""
    best_conf = 30
    best_reason = "no_candidate"
    best_hook = ""
    best_unique = False

    for raw in drafts:
        if not isinstance(raw, str):
            continue
        candidate = sanitize_public_text(raw).strip()
        if not candidate:
            continue
        if len(candidate) > MAX_REPLY_DRAFT_LEN:
            continue
        low = candidate.lower()
        if not creative:
            if any(low.startswith(prefix) for prefix in GENERIC_REPLY_OPENERS):
                continue
            if low.endswith("thoughts?") or low.endswith("your view?"):
                continue
            if has_unrelated_marketing_pivot(candidate, tweet_text=tweet_text, context_text=context_text, topic=topic):
                continue
            if tone == "heated" and any(term in low for term in CORPORATE_TONE_TERMS):
                continue
        prefix_key = _reply_prefix_key(candidate)
        if prefix_key in recent_prefixes:
            continue

        matched = ""
        for term in hook_terms:
            if term in low:
                matched = term
                break
        if not creative and not matched and hook_terms:
            continue

        conf = max(45, min(97, 55 + int(score / 6)))
        if matched:
            conf += 12
        if creative:
            conf += 5
        if persona_text:
            conf += 3
        conf = max(0, min(99, conf))

        best_text = candidate
        best_conf = conf
        best_reason = (
            "references specific tweet/thread details and passes 24h uniqueness check"
        )
        best_hook = matched or "n/a"
        best_unique = True
        break

    if not best_text:
        fallback = sanitize_public_text(str(drafts[0]).strip()) if drafts else ""
        best_text = fallback or "Thanks for sharing this."
        if len(best_text) > MAX_REPLY_DRAFT_LEN:
            best_text = best_text[: MAX_REPLY_DRAFT_LEN - 3].rstrip() + "..."
        best_conf = max(25, min(65, 40 + int(score / 10)))
        if _reply_prefix_key(best_text) in recent_prefixes:
            best_conf = max(20, best_conf - 25)
            best_reason = "similar to recent replies"
        else:
            best_reason = "fallback draft (limited specificity)"

    return {
        "reply_text": best_text,
        "confidence": best_conf,
        "reason": best_reason,
        "hook_used": best_hook,
        "unique_passed": best_unique,
        "topic": topic,
        "tone": tone,
    }


def list_approval_queue() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for p in sorted(APPROVAL_DIR.glob("q_*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                row["_path"] = str(p)
                out.append(row)
        except Exception:
            continue
    return out


def cmd_reply_approve(env_path: Path, args: argparse.Namespace) -> int:
    queue = list_approval_queue()
    if args.list or not args.approve:
        if args.json:
            print(json.dumps({"count": len(queue), "items": queue}, ensure_ascii=False, indent=2))
        else:
            print(f"Pending approvals: {len(queue)}")
            for row in queue:
                qid = str(row.get("id", ""))
                conf = row.get("confidence", "n/a")
                txt = str(row.get("text", "")).replace("\n", " ").strip()
                print(f"- q_{qid} | conf={conf} | {txt[:120]}")
        return 0

    cfg: Optional[Config] = None
    env_values: Dict[str, str] = {}
    wanted = set()
    for raw in args.approve:
        x = raw.strip()
        if not x:
            continue
        if x.startswith("q_"):
            x = x[2:]
        wanted.add(x)

    posted = 0
    skipped_already_replied = 0
    for row in queue:
        qid = str(row.get("id", "")).strip()
        if qid not in wanted:
            continue
        text = str(row.get("text", "")).strip()
        reply_to = str(row.get("in_reply_to", "")).strip() or None
        if not text:
            print(f"[WARN] q_{qid} missing text, skipping")
            continue
        if reply_to and has_replied_to_target(reply_to):
            print(f"[SKIP] q_{qid}: already replied to target {reply_to}; removing from queue")
            path = Path(str(row.get("_path", "")))
            if path.exists():
                path.unlink()
            skipped_already_replied += 1
            continue
        if args.dry_run:
            print(f"dry-run approve q_{qid}: {text[:140]}")
            continue
        if cfg is None:
            cfg, env_values = resolve_config(env_path)
        fresh, (status, body) = post_with_retry(
            cfg,
            env_path,
            env_values,
            text,
            reply_to_id=reply_to,
            unique_on_duplicate=bool(reply_to),
        )
        if status < 200 or status >= 300:
            print(f"[FAIL] q_{qid} post failed: {json.dumps(body, ensure_ascii=False)}")
            continue
        data = body.get("data") if isinstance(body, dict) else None
        posted_id = str(data.get("id", "")) if isinstance(data, dict) else ""
        if posted_id:
            _, posted_url = verify_post_visible(fresh, posted_id)
            print(f"Posted q_{qid}: {posted_url}")
            if reply_to:
                mark_replied_target(reply_to, posted_id, source="reply-approve")
        path = Path(str(row.get("_path", "")))
        if path.exists():
            path.unlink()
        posted += 1

    print(f"Approved and posted: {posted}")
    if skipped_already_replied:
        print(f"Skipped as already-replied targets: {skipped_already_replied}")
    return 0


def fetch_conversation_chain(
    bearer: str,
    tweet_id: str,
    max_depth: int = 8,
) -> List[Dict[str, object]]:
    chain: List[Dict[str, object]] = []
    current = str(tweet_id).strip()
    seen: Set[str] = set()

    while current and current not in seen and len(chain) < max_depth:
        seen.add(current)
        status, body = api_get_with_token(
            (
                f"{API_BASE}/tweets/{current}"
                "?expansions=author_id&tweet.fields=created_at,text,author_id,conversation_id,referenced_tweets"
                "&user.fields=username,name"
            ),
            bearer,
        )
        if status >= 400 or not isinstance(body, dict):
            break

        data = body.get("data")
        if not isinstance(data, dict):
            break
        includes = body.get("includes")
        users = includes.get("users") if isinstance(includes, dict) else None
        user_map: Dict[str, str] = {}
        if isinstance(users, list):
            for u in users:
                if isinstance(u, dict):
                    user_map[str(u.get("id", ""))] = str(u.get("username", "unknown"))
        author_id = str(data.get("author_id", ""))
        chain.append(
            {
                "tweet_id": str(data.get("id", "")),
                "author": user_map.get(author_id, "unknown"),
                "text": str(data.get("text", "")),
                "created_at": str(data.get("created_at", "")),
                "conversation_id": str(data.get("conversation_id", "")),
            }
        )

        refs = data.get("referenced_tweets")
        next_id = ""
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict) and str(ref.get("type", "")) == "replied_to":
                    next_id = str(ref.get("id", "")).strip()
                    break
        current = next_id

    chain.reverse()
    return chain


def cmd_search(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    bearer = get_read_bearer_token(env, env_path)
    limit = max(5, min(args.limit, 100))
    rows, users, meta = fetch_search_rows(
        bearer=bearer,
        query=args.query,
        limit=limit,
        max_pages=max(1, min(args.max_pages, 10)),
        since_id=args.since_id,
    )

    payload = {
        "query": args.query,
        "count": len(rows),
        "users": users,
        "meta": meta,
        "data": rows,
    }
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved search output -> {out}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Search query: {args.query}")
    print(f"Results: {len(rows)}")
    for i, row in enumerate(rows[: args.preview], start=1):
        if not isinstance(row, dict):
            continue
        tid = str(row.get("id", ""))
        aid = str(row.get("author_id", ""))
        username = users.get(aid, "unknown")
        score = score_tweet_for_discovery(row)
        text = str(row.get("text", "")).replace("\n", " ").strip()
        print(f"{i}. score={score} @{username} | {tid}")
        print(f"   {text}")
    return 0


def gather_data_snapshot(
    env_path: Path,
    handle: str = "OpenClawAI",
    query: str = 'openclaw OR "local ai agent" lang:en -is:retweet',
    limit: int = 20,
    max_pages: int = 1,
) -> Dict[str, object]:
    env = load_env_file(env_path)
    bearer = get_read_bearer_token(env, env_path)
    per_page = max(10, min(int(limit), 100))
    pages = max(1, min(int(max_pages), 10))

    mentions_rows: List[Dict[str, object]] = []
    mentions_error = ""
    try:
        uid = resolve_current_user_id(env_path, env)
        mention_params = [
            f"max_results={per_page}",
            "expansions=author_id",
            "tweet.fields=created_at,public_metrics,conversation_id,text",
            "user.fields=username,name",
        ]
        status, body = api_get_with_token(
            f"{API_BASE}/users/{uid}/mentions?" + "&".join(mention_params),
            bearer,
        )
        if status < 400 and isinstance(body, dict) and isinstance(body.get("data"), list):
            mentions_rows = [r for r in body.get("data", []) if isinstance(r, dict)]
        elif status >= 400:
            mentions_error = f"mentions_failed_{status}"
    except Exception as exc:
        mentions_error = str(exc)

    search_rows, users, meta = fetch_search_rows(
        bearer=bearer,
        query=query,
        limit=per_page,
        max_pages=pages,
        since_id=None,
        until_id=None,
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": ACTIVE_ACCOUNT,
        "handle": handle,
        "query": query,
        "mentions_count": len(mentions_rows),
        "search_count": len(search_rows),
        "mentions_error": mentions_error,
        "search_meta": meta,
        "search_users": users,
        "mentions_sample": mentions_rows[: min(10, len(mentions_rows))],
        "search_sample": search_rows[: min(20, len(search_rows))],
    }
    return report


def cmd_gather_data(env_path: Path, args: argparse.Namespace) -> int:
    report = gather_data_snapshot(
        env_path=env_path,
        handle=args.handle,
        query=args.query,
        limit=args.limit,
        max_pages=args.max_pages,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Decision: gather-data")
    print(f"Saved data snapshot -> {out}")
    print(f"Mentions: {report.get('mentions_count', 0)} | Search: {report.get('search_count', 0)}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    rows = load_tweet_memory(limit=args.limit, account=getattr(args, "account_name", None))
    if args.json:
        print(json.dumps({"count": len(rows), "items": rows}, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("No tweet memory yet.")
        return 0
    print(f"Tweet memory ({len(rows)}):")
    for i, row in enumerate(rows, start=1):
        kind = str(row.get("kind", "post"))
        ts = str(row.get("ts", ""))
        text = str(row.get("text", ""))
        tid = str(row.get("tweet_id", ""))
        url = str(row.get("url", ""))
        print(f"{i}. [{kind}] {ts}")
        if tid:
            print(f"   id={tid}")
        if url:
            print(f"   {url}")
        print(f"   {text}")
    return 0


def cmd_kit(env_path: Path, args: argparse.Namespace) -> int:
    summary: Dict[str, object] = {
        "mode": args.mode,
        "decision": "",
        "status": "",
        "action_taken": "",
        "artifact_paths": [],
        "errors": [],
    }
    barrier = get_rate_limit_barrier_status()
    if barrier.get("active"):
        summary["decision"] = "diagnose"
        summary["status"] = "blocked"
        summary["action_taken"] = "auto-diagnose"
        summary["errors"] = [f"rate_limit_barrier:{int(barrier.get('wait_seconds', 0))}s"]
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("Kit decision: diagnose")
            print(f"Rate-limit barrier active (~{int(barrier.get('wait_seconds', 0))}s).")
        return 1

    report = diagnose_engine(env_path=env_path, skip_network=False, reply_target_id=None)
    posting_ready = bool((report.get("posting") or {}).get("ready"))
    if args.mode == "diagnose" or not posting_ready:
        summary["decision"] = "diagnose"
        summary["status"] = "blocked" if not posting_ready else "ok"
        summary["action_taken"] = "auto-diagnose"
        summary["errors"] = list((report.get("posting") or {}).get("issues", []))
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("Kit decision: diagnose")
            for issue in summary["errors"]:
                print(f"- {issue}")
        return 1 if not posting_ready else 0

    if args.mode in {"auto", "reply"} and bool((report.get("reply_scan") or {}).get("ready")):
        try:
            ensure_reply_scan_token(env_path)
        except Exception as exc:
            summary["decision"] = "reply"
            summary["status"] = "blocked"
            summary["action_taken"] = "auth-fix-required"
            summary["errors"] = [f"reply_token_error:{exc}"]
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                print("Kit decision: reply")
                print(f"Reply token issue: {exc}")
            return 1
        try:
            has_candidate, item = _run_reply_preview(
                handle=args.reply_handle,
                since_id=args.reply_since_id,
                report_path=args.report_path,
                log_path=args.log_path,
                post=bool(args.mode == "reply" and not args.dry_run),
                verify_post=True,
            )
        except Exception as exc:
            has_candidate, item = (False, None)
            summary["errors"] = [str(exc)]
            if args.mode == "reply":
                summary["decision"] = "reply"
                summary["status"] = "blocked"
                summary["action_taken"] = "diagnose"
                if args.json:
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                else:
                    print("Kit decision: reply")
                    print(f"Reply engine error: {exc}")
                return 1
        if has_candidate and isinstance(item, dict):
            summary["decision"] = "reply"
            summary["status"] = "ok"
            item_status = str(item.get("status", "drafted"))
            summary["action_taken"] = "post-reply" if item_status == "posted" else "draft-reply"
            summary["artifact_paths"] = [args.report_path]
            summary["reply_preview"] = {
                "tweet_id": item.get("tweet_id", ""),
                "author": item.get("author", ""),
                "text": item.get("picked_text", ""),
            }
            if item_status == "posted":
                summary["reply_url"] = item.get("reply_url", "")
                summary["reply_id"] = item.get("reply_id", "")
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                print("Kit decision: reply")
                print(f"{item.get('tweet_id')} | @{item.get('author')}")
                print(str(item.get("picked_text", "")).strip())
                if item_status == "posted" and item.get("reply_url"):
                    print(f"Posted: {item.get('reply_url')}")
            return 0
        if args.mode == "reply":
            summary["decision"] = "reply"
            summary["status"] = "no_candidate"
            summary["action_taken"] = "none"
            if args.json:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            else:
                print("Kit decision: reply")
                print("No suitable reply candidate found.")
            return 0

    if args.mode == "auto" and args.gather_when_no_reply:
        snapshot = gather_data_snapshot(
            env_path=env_path,
            handle=args.reply_handle,
            query=args.gather_query,
            limit=args.gather_limit,
            max_pages=args.gather_max_pages,
        )
        out = Path(args.gather_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["decision"] = "gather-data"
        summary["status"] = "ok"
        summary["action_taken"] = "saved-snapshot"
        summary["artifact_paths"] = [str(out)]
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("Kit decision: gather-data")
            print(f"Saved data snapshot -> {out}")
        return 0

    cfg, env_values = resolve_config(env_path)
    base_text = args.text
    if args.file:
        base_text = Path(args.file).read_text(encoding="utf-8").strip()
    if not base_text:
        web_items, web_err = maybe_collect_web_inspiration(
            enabled=bool(getattr(args, "web_inspiration", True)),
            query=str(getattr(args, "web_query", "ai agents automation reliability")),
            item_limit=int(getattr(args, "web_items", 8)),
        )
        if web_items:
            summary["web_inspiration_count"] = len(web_items)
        if web_err:
            errs = summary.get("errors", [])
            if isinstance(errs, list):
                errs.append(f"web_inspiration_error:{web_err}")
                summary["errors"] = errs
        base_text = generate_reflective_post_text(
            topic=str(getattr(args, "web_query", "ai agents")),
            inspiration_texts=web_items,
            style=str(getattr(args, "style", "auto") or "auto"),
            voice=str(getattr(args, "voice", "auto") or "auto"),
            viral_pack=str(getattr(args, "viral_pack", "") or ""),
            anti_boring=bool(getattr(args, "anti_boring", False)),
            sharpen=bool(getattr(args, "sharpen", False)),
            judge_threshold=float(getattr(args, "judge_threshold", 82.0)),
            max_attempts=int(getattr(args, "max_attempts", 7)),
            ensemble_size=int(getattr(args, "ensemble", 1)),
            viral_boost=bool(getattr(args, "viral_boost", False)),
        )
    text = make_unique_public_tweet(base_text)
    if args.dry_run:
        summary["decision"] = "post"
        summary["status"] = "dry_run"
        summary["action_taken"] = "render-text"
        summary["post_text"] = text
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("Kit decision: post")
            print(text)
        return 0

    fresh, (status, body) = post_with_retry(cfg, env_path, env_values, text)
    if status < 200 or status >= 300:
        raise TwitterHelperError(f"Kit post failed ({status}): {json.dumps(body, ensure_ascii=False)}")
    data = body.get("data") if isinstance(body, dict) else None
    tweet_id = str(data.get("id", "")) if isinstance(data, dict) else ""
    if not tweet_id:
        raise TwitterHelperError("Kit post returned no tweet id.")
    _, tweet_url = verify_post_visible(fresh, tweet_id)
    record_recent_post(text)
    record_tweet_memory(kind="post", text=text, tweet_id=tweet_id, url=tweet_url, meta={"source": "kit"})
    summary["decision"] = "post"
    summary["status"] = "ok"
    summary["action_taken"] = "posted"
    summary["tweet_id"] = tweet_id
    summary["url"] = tweet_url
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("Kit decision: post")
        print(f"Posted: {tweet_url}")
    return 0


def cmd_reply_discover_run(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    bearer = get_read_bearer_token(env, env_path)
    queries: List[str] = []
    if args.query:
        queries = [args.query]
    else:
        watchlists = load_watchlists()
        queries = watchlists.get(args.watchlist or "default", [])
        if not queries:
            queries = ['openclaw OR "local ai agent" lang:en -is:retweet min_faves:5']
    queries = queries[:1]

    approval_queue_enabled = bool(getattr(args, "approval_queue", False))
    post_enabled = bool(args.auto_post and not args.dry_run and not approval_queue_enabled)
    cfg: Optional[Config] = None
    env_values: Dict[str, str] = {}
    if post_enabled:
        cfg, env_values = resolve_config(env_path)

    try:
        from reply_engine.twitter_helper import generate_reply_drafts
    except Exception as exc:
        raise TwitterHelperError(
            "Reply engine not ready. Install dependencies with: pip install -r requirements-reply-engine.txt"
        ) from exc

    total_seen = 0
    total_candidates = 0
    total_posted = 0
    total_skipped_already_replied = 0
    total_skipped_already_queued = 0
    total_skipped_offtopic_discovery = 0
    per_query_results: List[Dict[str, object]] = []
    persona_text = load_persona_text()
    replied_targets = load_replied_targets(ACTIVE_ACCOUNT)
    queued_targets = queued_target_ids() if approval_queue_enabled else set()

    for query in queries:
        effective_since = args.since_id or load_query_since_id(query, ACTIVE_ACCOUNT)
        rows, users, _ = fetch_search_rows(
            bearer=bearer,
            query=query,
            limit=max(1, min(args.max_tweets, 1)),
            max_pages=max(1, min(args.max_pages, 10)),
            since_id=effective_since,
        )
        total_seen += len(rows)
        max_seen_id: Optional[str] = None
        query_rows: List[Dict[str, object]] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            tid = str(row.get("id", ""))
            if tid.isdigit():
                if max_seen_id is None or int(tid) > int(max_seen_id):
                    max_seen_id = tid

            score = score_tweet_for_discovery(row)
            if score < args.min_score:
                continue
            if tid in replied_targets:
                total_skipped_already_replied += 1
                query_rows.append(
                    {
                        "tweet_id": tid,
                        "author": users.get(str(row.get("author_id", "")), "unknown"),
                        "score": score,
                        "action": "skipped_already_replied",
                    }
                )
                continue
            if approval_queue_enabled and tid in queued_targets:
                total_skipped_already_queued += 1
                query_rows.append(
                    {
                        "tweet_id": tid,
                        "author": users.get(str(row.get("author_id", "")), "unknown"),
                        "score": score,
                        "action": "skipped_already_queued",
                    }
                )
                continue
            aid = str(row.get("author_id", ""))
            author = users.get(aid, "unknown")
            text = str(row.get("text", "")).strip()

            context_chain = fetch_conversation_chain(bearer=bearer, tweet_id=tid, max_depth=6)
            context_text = "\n".join(
                [str(x.get("text", "")).strip() for x in context_chain[:-1] if x.get("text")]
            ).strip()

            reply_eval = generate_unique_applicable_reply(
                author=author,
                tweet_text=text,
                context_text=context_text,
                score=score,
                generate_drafts_fn=generate_reply_drafts,
                persona_text=persona_text,
                recent_hours=24,
                is_discovery=True,
            )
            draft = str(reply_eval.get("reply_text", "")).strip()
            confidence = int(reply_eval.get("confidence", 0))
            reason = str(reply_eval.get("reason", "")).strip()
            hook_used = str(reply_eval.get("hook_used", "")).strip()
            unique_passed = bool(reply_eval.get("unique_passed", False))
            topic = str(reply_eval.get("topic", "")).strip()
            tone = str(reply_eval.get("tone", "")).strip()

            item = {
                "tweet_id": tid,
                "author": author,
                "score": score,
                "confidence": confidence,
                "tweet_text": text,
                "thread_context": context_text,
                "draft_reply": draft,
                "reason": reason,
                "hook_used": hook_used,
                "unique_passed": unique_passed,
                "topic": topic,
                "tone": tone,
                "action": "draft",
            }

            if not draft or confidence <= 0:
                item["action"] = "skipped_offtopic_discovery"
                total_skipped_offtopic_discovery += 1
                query_rows.append(item)
                continue
            total_candidates += 1

            if (
                approval_queue_enabled
                and draft
                and confidence >= args.min_confidence
            ):
                qid = save_for_approval(
                    {
                        "text": draft,
                        "in_reply_to": tid,
                        "confidence": confidence,
                        "reason": reason or "discover_run_threshold",
                        "hook_used": hook_used,
                        "source": "unique-apply",
                        "query": query,
                        "tweet_text": text,
                    }
                )
                record_recent_reply(draft, tid)
                item["action"] = "queued"
                item["queue_id"] = f"q_{qid}"
                queued_targets.add(tid)
            elif post_enabled and cfg is not None and draft and confidence >= args.min_confidence:
                fresh, (status, body) = post_with_retry(
                    cfg,
                    env_path,
                    env_values,
                    draft,
                    reply_to_id=tid,
                    unique_on_duplicate=True,
                )
                if status < 200 or status >= 300:
                    item["action"] = "post_failed"
                    item["error"] = json.dumps(body, ensure_ascii=False)
                else:
                    data = body.get("data") if isinstance(body, dict) else None
                    posted_id = str(data.get("id", "")) if isinstance(data, dict) else ""
                    if posted_id:
                        _, posted_url = verify_post_visible(fresh, posted_id)
                        item["action"] = "posted"
                        item["posted_tweet_id"] = posted_id
                        item["posted_url"] = posted_url
                        mark_replied_target(tid, posted_id, source="reply-discover-run")
                        replied_targets[tid] = {
                            "reply_id": posted_id,
                            "source": "reply-discover-run",
                            "at": "",
                        }
                        record_recent_reply(draft, tid)
                        total_posted += 1
            query_rows.append(item)

        if max_seen_id:
            save_query_since_id(query, ACTIVE_ACCOUNT, max_seen_id)

        per_query_results.append(
            {
                "query": query,
                "since_id": effective_since,
                "next_since_id": max_seen_id,
                "results": query_rows,
            }
        )

    payload = {
        "account": ACTIVE_ACCOUNT,
        "watchlist": args.watchlist,
        "queries": queries,
        "max_tweets": args.max_tweets,
        "min_score": args.min_score,
        "min_confidence": args.min_confidence,
        "auto_post": args.auto_post,
        "approval_queue": approval_queue_enabled,
        "dry_run": args.dry_run,
        "total_seen": total_seen,
        "total_candidates": total_candidates,
        "total_posted": total_posted,
        "total_skipped_already_replied": total_skipped_already_replied,
        "total_skipped_already_queued": total_skipped_already_queued,
        "total_skipped_offtopic_discovery": total_skipped_offtopic_discovery,
        "results": per_query_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved discovery output -> {out}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Queries: {len(queries)}")
    print(
        "Seen: "
        f"{total_seen} | candidates: {total_candidates} | posted: {total_posted} | "
        f"skip_replied: {total_skipped_already_replied} | skip_queued: {total_skipped_already_queued} | "
        f"skip_offtopic: {total_skipped_offtopic_discovery}"
    )
    for q in per_query_results:
        print(f"- {q['query']}")
        rows = q.get("results", [])
        if not isinstance(rows, list):
            continue
        for row in rows[: args.preview]:
            if not isinstance(row, dict):
                continue
            print(
                f"  {row.get('action')} | score={row.get('score')} conf={row.get('confidence')} "
                f"| {row.get('tweet_id')} | @{row.get('author')}"
            )
    return 0


def fetch_search_rows(
    bearer: str,
    query: str,
    limit: int,
    max_pages: int,
    since_id: Optional[str] = None,
    until_id: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, str], Dict[str, object]]:
    # X recent search endpoint only accepts max_results in [10, 100].
    limit = max(10, min(int(limit), 100))
    max_pages = max(1, min(int(max_pages), 10))
    all_data: List[Dict[str, object]] = []
    users: Dict[str, str] = {}
    meta: Dict[str, object] = {}
    next_token = None
    pages = 0

    while pages < max_pages:
        params = [
            f"query={urllib.parse.quote(query)}",
            f"max_results={limit}",
            "expansions=author_id",
            "tweet.fields=created_at,public_metrics,conversation_id,text",
            "user.fields=username,name",
        ]
        if since_id:
            params.append(f"since_id={urllib.parse.quote(str(since_id))}")
        if until_id:
            params.append(f"until_id={urllib.parse.quote(str(until_id))}")
        if next_token:
            params.append(f"next_token={urllib.parse.quote(next_token)}")

        status, body = api_get_with_token(
            f"{API_BASE}/tweets/search/recent?" + "&".join(params),
            bearer,
        )
        if status == 401:
            raise TwitterHelperError(
                "browse-twitter unauthorized (401). "
                "Check TWITTER_BEARER_TOKEN has v2 read access, or refresh OAuth2 with `auth-login`."
            )
        if status >= 400:
            raise TwitterHelperError(
                f"browse query failed ({status}): {json.dumps(body, ensure_ascii=False)}"
            )

        includes = body.get("includes") if isinstance(body, dict) else None
        if isinstance(includes, dict) and isinstance(includes.get("users"), list):
            for u in includes["users"]:
                if isinstance(u, dict):
                    users[str(u.get("id", ""))] = u.get("username", "unknown")

        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    all_data.append(row)

        meta = body.get("meta") if isinstance(body, dict) and isinstance(body.get("meta"), dict) else {}
        next_token = str(meta.get("next_token", "")) if meta.get("next_token") else None
        pages += 1
        if not next_token:
            break

    return all_data, users, meta


def top_terms(texts: List[str], top_n: int = 8) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for text in texts:
        for m in WORD_RE.findall(text.lower()):
            if m in STOPWORDS:
                continue
            counts[m] = counts.get(m, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def calculate_entropy(text: str) -> float:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    if not toks:
        return 0.0
    return len(set(toks)) / max(1, len(toks))


def frog_judge_score(
    draft: str,
    voice: str,
    style: Optional[str] = None,
    viral_boost: bool = False,
) -> float:
    text = str(draft or "").strip()
    if not text:
        return 0.0
    lowered = text.lower()

    specificity = 0.0
    if re.search(r"\b\d{1,3}\b|\$?sol\b|pipeline|delta|heeming|taxing|benchmark|metric|auth|dedupe", lowered):
        specificity += 0.65
    if ":" in text:
        specificity += 0.20
    if len(text) >= 120:
        specificity += 0.15
    specificity = min(1.0, specificity)

    persona = _voice_persona(voice)
    slang_bank = {str(s).lower() for s in persona.get("slang_bank", []) if str(s).strip()}
    slang_hits = sum(1 for s in slang_bank if s and s in lowered)
    hooks = [str(h).lower() for h in persona.get("hooks", [])]
    closers = [str(c).lower() for c in persona.get("closers", [])]
    hook_hit = any(h in lowered for h in hooks[:6]) if hooks else False
    closer_hit = any(c in lowered for c in closers[:6]) if closers else False
    frog_energy = min(1.0, (0.25 if hook_hit else 0.0) + (0.25 if closer_hit else 0.0) + min(0.5, slang_hits * 0.12))

    engagement_markers = ("what", "prove", "thread", "your move", "prove me wrong", "spot or perps", "challenge")
    engagement = 0.0
    if text.endswith("?") or text.endswith("!"):
        engagement += 0.6
    if any(m in lowered for m in engagement_markers):
        engagement += 0.4
    engagement = min(1.0, engagement)

    anti_boring = 1.0 if calculate_entropy(text) >= 0.52 and not any(b in lowered for b in ANTI_BORING_BANNED) else 0.0

    voice_hooks = [str(m).lower() for m in (VOICE_CONFIGS.get(voice) or {}).get("hooks", [])]
    persona_hooks = [str(m).lower() for m in persona.get("hooks", [])]
    voice_authenticity = 1.0 if any(m in lowered for m in voice_hooks) else 0.0
    if not voice_authenticity and any(m in lowered for m in persona_hooks[:6]):
        voice_authenticity = 0.6

    weighted = (
        specificity * FROG_JUDGE_WEIGHTS["specificity"]
        + frog_energy * FROG_JUDGE_WEIGHTS["frog_energy"]
        + engagement * FROG_JUDGE_WEIGHTS["engagement"]
        + anti_boring * FROG_JUDGE_WEIGHTS["anti_boring"]
        + voice_authenticity * FROG_JUDGE_WEIGHTS["voice_authenticity"]
    )
    base_score = min(100.0, max(0.0, weighted * 100.0))

    viral = 0.0
    if (
        "?" in text
        or "prove me wrong" in lowered
        or "your move" in lowered
        or "thread if" in lowered
        or "send it" in lowered
    ):
        viral += float(VIRAL_POTENTIAL_BONUS["question_or_challenge"])
    if re.search(r"\d", text):
        viral += float(VIRAL_POTENTIAL_BONUS["number_or_metric"])
    if any(h.lower() in lowered for h in [str(x) for x in FROG_PERSONA.get("hooks", [])]):
        viral += float(VIRAL_POTENTIAL_BONUS["frog_hook"])
    if 140 <= len(text) <= 220:
        viral += float(VIRAL_POTENTIAL_BONUS["optimal_length"])
    emoji_count = sum(1 for ch in text if ch in {"🐸", "🔥", "📈", "💀"})
    if 1 <= emoji_count <= 3:
        viral += float(VIRAL_POTENTIAL_BONUS["emoji_fit"])
    if (style or "").lower() in {"contrarian", "story"} or any(
        x in lowered for x in ("unpopular", "counterintuitively", "contrarian", "pattern under")
    ):
        viral += float(VIRAL_POTENTIAL_BONUS["contrarian_signal"])

    viral_weight = 0.9 if viral_boost else 0.6
    final_score = min(100.0, base_score + (viral * viral_weight))
    return round(final_score, 1)


def _apply_case_style(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def sharpen_with_lexicon(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mapped = HIGH_SIGNAL_LEXICON.get(token.lower())
        if not mapped:
            return token
        return _apply_case_style(token, mapped)

    sharpened = re.sub(r"\b[A-Za-z][A-Za-z'-]*\b", repl, raw)
    if calculate_entropy(sharpened) < 1.6:
        sharpened = inject_frog_chaos(sharpened, 0.9)
    return sanitize_public_text(sharpened)


def _resolve_voice_name(voice: str) -> str:
    name = str(voice or "auto").strip().lower()
    if name == "auto":
        rng = random.SystemRandom()
        names = ["chaotic", "degen", "based", "savage", "operator", "sage", "shitposter"]
        weights = [25, 15, 20, 10, 10, 10, 10]
        return rng.choices(names, weights=weights, k=1)[0]
    if name in VOICE_CONFIGS:
        return name
    return "chaotic"


def _voice_persona(voice: str) -> Dict[str, object]:
    resolved = _resolve_voice_name(voice)
    cfg = VOICE_CONFIGS.get(resolved) or {}
    base_hooks = list(FROG_PERSONA["hooks"])
    voice_hooks = [str(x) for x in cfg.get("hooks", []) if isinstance(x, str)]
    hooks = (voice_hooks + base_hooks)[:12] if voice_hooks else base_hooks
    slang_base = list(FROG_PERSONA["slang_bank"])
    slang_extra = [str(x) for x in cfg.get("slang_terms", []) if isinstance(x, str)]
    slang_bank = slang_base + slang_extra
    intensity = float(cfg.get("slang_intensity", 0.6) or 0.6)
    repeat_factor = max(1, int(round(1 + intensity * 2)))
    slang_bank = slang_bank * repeat_factor
    return {
        "hooks": hooks,
        "closers": list(FROG_PERSONA["closers"]),
        "slang_bank": slang_bank,
        "emoji_density": float(cfg.get("emoji_density", FROG_PERSONA["emoji_density"])),
        "extra_tail": str(cfg.get("extra_tail", "")).strip(),
        "resolved_voice": resolved,
    }


def inject_frog_chaos(draft: str, topic_entropy: float, persona: Optional[Dict[str, object]] = None) -> str:
    p = persona if isinstance(persona, dict) else FROG_PERSONA
    rng = random.SystemRandom()
    text = str(draft).strip()
    if rng.random() < 0.40:
        hooks = [str(x) for x in p.get("hooks", FROG_PERSONA["hooks"])]
        text = f"{rng.choice(hooks)} {text}"
    if rng.random() < 0.60:
        closers = [str(x) for x in p.get("closers", FROG_PERSONA["closers"])]
        text = f"{text}\n\n{rng.choice(closers)}"
    if ("sol" in text.lower() or topic_entropy > 0.75) and rng.random() < 0.70:
        slang_bank = [str(x) for x in p.get("slang_bank", FROG_PERSONA["slang_bank"])]
        slang = rng.choice(slang_bank)
        text = text.replace(".", f". {slang}.", 1) if "." in text else f"{text} {slang}"
    if rng.random() < float(p.get("emoji_density", FROG_PERSONA["emoji_density"])):
        text = text + " 🐸"
    return sanitize_public_text(text)


def apply_voice(raw: str, voice: str = "auto", topic_entropy: float = 0.0, sharpen: bool = False) -> str:
    persona = _voice_persona(voice)
    text = inject_frog_chaos(raw, topic_entropy=topic_entropy, persona=persona)
    if sharpen:
        text = sharpen_with_lexicon(text)
    tail = str(persona.get("extra_tail", "")).strip()
    if tail and random.SystemRandom().random() < 0.5:
        text = f"{text}\n\n{tail}"
    return sanitize_public_text(text)


def _http_get_bytes(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = WEB_REQ_TIMEOUT_SECONDS,
    max_bytes: int = WEB_REQ_MAX_BYTES,
    retries: int = WEB_REQ_MAX_RETRIES,
) -> bytes:
    req_headers = {"User-Agent": "twitter-engine/1.0", "Accept": "*/*"}
    if headers:
        req_headers.update(headers)
    attempt = 0
    while True:
        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(max(1, max_bytes))
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if attempt < retries and status in {429, 500, 502, 503, 504}:
                backoff = min(4.0, 0.6 * (2 ** attempt))
                time.sleep(backoff)
                attempt += 1
                continue
            raise
        except Exception:
            if attempt < retries:
                backoff = min(3.0, 0.5 * (2 ** attempt))
                time.sleep(backoff)
                attempt += 1
                continue
            raise


def _web_cache_get(key: str) -> Optional[Dict[str, List[str]]]:
    row = WEB_INSPIRATION_CACHE.get(key)
    if not isinstance(row, dict):
        return None
    expires_at = float(row.get("expires_at", 0.0) or 0.0)
    if expires_at <= time.time():
        WEB_INSPIRATION_CACHE.pop(key, None)
        return None
    val = row.get("value")
    if not isinstance(val, dict):
        return None
    return val


def _web_cache_set(key: str, value: Dict[str, List[str]], ttl_seconds: int = WEB_INSPIRATION_CACHE_TTL_SECONDS) -> None:
    WEB_INSPIRATION_CACHE[key] = {
        "expires_at": time.time() + max(1, ttl_seconds),
        "value": value,
    }


def fetch_news_rss_headlines(query: str, limit: int = 8, timeout: int = WEB_REQ_TIMEOUT_SECONDS) -> List[str]:
    q = query.strip()
    if not q:
        return []
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    try:
        raw = _http_get_bytes(
            url=url,
            headers={"Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.1"},
            timeout=timeout,
            max_bytes=WEB_REQ_MAX_BYTES,
        )
    except Exception:
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    out: List[str] = []
    for item in root.findall(".//item"):
        title = ""
        title_el = item.find("title")
        if title_el is not None and title_el.text:
            title = title_el.text.strip()
        if not title:
            continue
        if " - " in title:
            title = title.split(" - ", 1)[0].strip()
        title = sanitize_public_text(title)
        if not title:
            continue
        out.append(title)
        if len(out) >= max(1, limit):
            break
    return out


def fetch_hn_headlines(query: str, limit: int = 6, timeout: int = WEB_REQ_TIMEOUT_SECONDS) -> List[str]:
    q = query.strip()
    if not q:
        return []
    params = urllib.parse.urlencode(
        {"query": q, "tags": "story", "hitsPerPage": max(1, min(limit, 20))}
    )
    url = f"https://hn.algolia.com/api/v1/search?{params}"
    try:
        raw = _http_get_bytes(
            url=url,
            headers={"Accept": "application/json"},
            timeout=timeout,
            max_bytes=WEB_REQ_MAX_BYTES,
        )
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return []
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        return []
    out: List[str] = []
    for row in hits:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or row.get("story_title") or "").strip()
        if not title:
            continue
        title = sanitize_public_text(title)
        if not title:
            continue
        out.append(title)
        if len(out) >= max(1, limit):
            break
    return out


def gather_web_inspiration(
    query: str,
    news_limit: int = 8,
    hn_limit: int = 6,
) -> Dict[str, List[str]]:
    q = sanitize_public_text(query).strip()
    if not q:
        return {"news": [], "hn": [], "items": []}
    cache_key = f"{q.lower()}|n{max(1, news_limit)}|h{max(1, hn_limit)}"
    cached = _web_cache_get(cache_key)
    if cached is not None:
        return cached

    news_items = fetch_news_rss_headlines(query=q, limit=max(1, min(news_limit, 12)))
    hn_items = fetch_hn_headlines(query=q, limit=max(1, min(hn_limit, 12)))
    seen: Set[str] = set()
    merged: List[str] = []
    for item in news_items + hn_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    out = {"news": news_items, "hn": hn_items, "items": merged}
    _web_cache_set(cache_key, out)
    return out


def maybe_collect_web_inspiration(
    enabled: bool,
    query: str,
    item_limit: int = 8,
) -> Tuple[List[str], Optional[str]]:
    if not enabled:
        return [], None
    try:
        pack = gather_web_inspiration(
            query=query,
            news_limit=max(1, min(item_limit, 12)),
            hn_limit=max(1, min(item_limit, 12)),
        )
        items = list(pack.get("items", []))[: max(1, min(item_limit, 12))]
        return items, None
    except Exception as exc:
        return [], str(exc)


def make_inspiration_drafts(topic: str, sample_texts: List[str], draft_count: int) -> List[str]:
    terms = [t for t, _ in top_terms(sample_texts, top_n=10)]
    anchors = terms[: min(3, len(terms))]
    anchor_text = ", ".join(anchors) if anchors else topic

    templates = [
        f"Hot take on {topic}: most teams over-rotate on hype and under-invest in repeatable execution around {anchor_text}.",
        f"{topic} builders: one metric, one constraint, one daily iteration. Everything else is noise.",
        f"Counterintuitive lesson from {topic}: tighter loops beat bigger launches when {anchor_text} is moving fast.",
        f"Question for people shipping in {topic}: what changed your outcomes most in the last 30 days?",
        f"Build signal in {topic}: clearer operators around {anchor_text} are separating from generic commentary.",
    ]
    if draft_count <= len(templates):
        return templates[:draft_count]
    out = list(templates)
    while len(out) < draft_count:
        idx = len(out) + 1
        out.append(f"{topic} note #{idx}: share one tactic, one metric, one lesson.")
    return out


def cmd_inspire_tweets(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    bearer = get_read_bearer_token(env, env_path)
    limit = max(5, min(args.limit, 100))
    max_pages = max(1, min(args.max_pages, 10))

    query = args.query
    if not query:
        q_topic = args.topic.strip() if args.topic else "openclaw OR ai agents"
        query = f"({q_topic}) lang:en -is:retweet"

    rows, users, _ = fetch_search_rows(
        bearer=bearer,
        query=query,
        limit=limit,
        max_pages=max_pages,
        since_id=args.since_id,
        until_id=args.until_id,
    )
    if not rows:
        print("No tweets found for inspiration query.")
        return 0

    sample = rows[: min(len(rows), args.sample_size)]
    sample_texts = [str(r.get("text", "")) for r in sample if isinstance(r, dict)]
    web_items: List[str] = []
    web_query = (args.web_query or args.topic or "ai agents automation reliability").strip()
    if bool(getattr(args, "web_inspiration", True)):
        web_items, _ = maybe_collect_web_inspiration(
            enabled=True,
            query=web_query,
            item_limit=int(getattr(args, "web_items", 8)),
        )
    all_signal_texts = sample_texts + web_items
    terms = top_terms(all_signal_texts, top_n=8)
    drafts = make_inspiration_drafts(args.topic or "this space", all_signal_texts, args.draft_count)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": query,
            "count": len(rows),
            "sample_size": len(sample),
            "web_count": len(web_items),
            "top_terms": terms,
            "drafts": drafts,
            "sample": sample,
            "web_items": web_items,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved inspiration output -> {out}")

    if args.json:
        print(
            json.dumps(
                {
                    "query": query,
                    "count": len(rows),
                    "web_count": len(web_items),
                    "top_terms": terms,
                    "drafts": drafts,
                    "sample": sample,
                    "web_items": web_items,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"Query: {query}")
    print(f"Fetched: {len(rows)} tweets")
    if web_items:
        print(f"Web signal items: {len(web_items)}")
    print("Top themes:")
    for term, count in terms:
        print(f"- {term} ({count})")
    print("Inspiration drafts:")
    for i, d in enumerate(drafts, start=1):
        print(f"{i}. {d}")
    print("Sample tweets:")
    for i, row in enumerate(sample, start=1):
        if not isinstance(row, dict):
            continue
        tid = str(row.get("id", ""))
        aid = str(row.get("author_id", ""))
        username = users.get(aid, "unknown")
        text = str(row.get("text", "")).replace("\n", " ").strip()
        print(f"{i}. @{username} | {tid}")
        print(f"   {text}")
    if web_items:
        print("Web headlines:")
        for i, item in enumerate(web_items, start=1):
            print(f"{i}. {item}")
    return 0


def config_status(env_path: Path) -> Dict[str, object]:
    env = load_env_file(env_path)
    tm = token_manager(env_path)
    tm.migrate_from_env(env)
    exists = env_path.exists()

    has_client_id = bool(get_env_value(env, "TWITTER_CLIENT_ID"))
    has_client_secret = bool(get_env_value(env, "TWITTER_CLIENT_SECRET"))
    has_bearer_token = bool(get_env_value(env, "TWITTER_BEARER_TOKEN"))
    has_access_token = bool(tm.get_access_token(env))
    has_refresh_token = bool(tm.get_refresh_token(env))

    ready_for_oauth_login = exists and has_client_id and has_client_secret
    ready_for_posting = (
        exists and has_client_id and has_client_secret and has_access_token and has_refresh_token
    )

    next_steps: List[str] = []
    if not exists:
        next_steps.append("run setup")
    elif not has_client_id or not has_client_secret:
        next_steps.append("run setup")
    elif not has_access_token or not has_refresh_token:
        next_steps.append("run auth-login")
    else:
        next_steps.append("run doctor")
        next_steps.append("run post --text \"hello from Twitter Engine\"")

    return {
        "env_file": str(env_path),
        "env_exists": exists,
        "has_client_id": has_client_id,
        "has_client_secret": has_client_secret,
        "has_bearer_token": has_bearer_token,
        "has_oauth2_access_token": has_access_token,
        "has_oauth2_refresh_token": has_refresh_token,
        "ready_for_oauth_login": ready_for_oauth_login,
        "ready_for_posting": ready_for_posting,
        "redirect_uri": get_env_value(env, "TWITTER_REDIRECT_URI", DEFAULT_REDIRECT_URI),
        "website_url": get_env_value(env, "TWITTER_WEBSITE_URL", ""),
        "scopes": get_env_value(env, "TWITTER_SCOPES", DEFAULT_SCOPES),
        "next_steps": next_steps,
    }


def normalize_redirect_uri(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return DEFAULT_REDIRECT_URI

    if "://" not in value:
        value = f"http://{value}"

    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise TwitterHelperError("Redirect URI must use http or https.")
    if not parsed.hostname:
        raise TwitterHelperError("Redirect URI must include a hostname.")

    host = parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    if path and not path.startswith("/"):
        path = "/" + path

    # Preserve user intent:
    # - If they typed host:port with no path, keep no path.
    # - If they typed a path, keep it exactly.
    # - Only use /callback when the value is empty (handled above).
    return f"{parsed.scheme}://{host}{port}{path}"


def infer_website_url(redirect_uri: str) -> str:
    parsed = urllib.parse.urlparse(redirect_uri)
    if not parsed.hostname:
        return "http://127.0.0.1"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def generate_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(48)).decode("utf-8").rstrip("=")


def generate_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def build_auth_url(
    client_id: str,
    redirect_uri: str,
    scopes: str,
    state: str,
    code_challenge: str,
) -> str:
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(query)


def parse_callback_input(callback_input: str) -> Tuple[str, Optional[str]]:
    raw = callback_input.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [""])[0]
        state = params.get("state", [""])[0] or None
        if not code:
            raise TwitterHelperError("No authorization code found in callback URL.")
        return code, state
    return raw, None


def exchange_authorization_code(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> Tuple[str, str]:
    headers: Dict[str, str] = {}
    if client_secret:
        headers["Authorization"] = get_basic_auth_header(client_id, client_secret)

    status, body = http_json(
        "POST",
        TOKEN_URL,
        headers,
        form_payload={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
            "client_id": client_id,
        },
    )
    if status < 200 or status >= 300:
        raise TwitterHelperError(
            f"Code exchange failed ({status}): {json.dumps(body, ensure_ascii=False)}"
        )

    access_token = str(body.get("access_token", ""))
    refresh_token = str(body.get("refresh_token", ""))
    if not access_token or not refresh_token:
        raise TwitterHelperError(
            "OAuth response missing access_token or refresh_token. Check app scopes and settings."
        )

    return access_token, refresh_token


def cmd_setup(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)

    if not env_path.exists():
        print(f"No env file found at {env_path}. Creating one now.")

    print("\nTwitter Helper Setup")
    print("This step configures app credentials (not access tokens yet).\n")
    print("Redirect examples accepted: 127.0.0.1:3000, localhost:3000/callback, full URL.\n")

    for key in ["TWITTER_CLIENT_ID", "TWITTER_CLIENT_SECRET"]:
        current = "" if args.reset else env.get(key, "")
        env[key] = prompt_value(key, current=current)

    print("\nOptional: add App-Only bearer token for read/scan reply workflows.")
    print("Press Enter to skip this field.\n")
    env["TWITTER_BEARER_TOKEN"] = prompt_value(
        "TWITTER_BEARER_TOKEN",
        current="" if args.reset else env.get("TWITTER_BEARER_TOKEN", ""),
        allow_empty=True,
    )

    redirect_input = prompt_value(
        "TWITTER_REDIRECT_URI",
        current="" if args.reset else env.get("TWITTER_REDIRECT_URI", ""),
        default_if_missing=DEFAULT_REDIRECT_URI,
    )
    env["TWITTER_REDIRECT_URI"] = normalize_redirect_uri(redirect_input)
    env["TWITTER_WEBSITE_URL"] = infer_website_url(env["TWITTER_REDIRECT_URI"])
    print(f"Normalized redirect URI: {env['TWITTER_REDIRECT_URI']}")
    print(f"Suggested website URL: {env['TWITTER_WEBSITE_URL']}")
    print("Important: Callback URI in Twitter portal must exactly match TWITTER_REDIRECT_URI.")

    # Keep setup friction low: auto-apply the standard scopes needed for this helper.
    env["TWITTER_SCOPES"] = DEFAULT_SCOPES
    print(f"Using OAuth scopes: {DEFAULT_SCOPES}")

    write_env_file(env_path, env)
    print(f"\nSaved config to {env_path}")
    if env.get("TWITTER_BEARER_TOKEN"):
        print("App-only bearer token is configured.")
    else:
        print("No app-only bearer token set. Posting works, but reply scan workflows may be limited.")
    print("Run `app-settings` to see exact Twitter portal settings to apply.")
    print("Next step: run `auth-login` to launch browser OAuth and generate OAuth2 tokens.")
    if not args.skip_auth_login and prompt_yes_no("Launch OAuth2 login now?", default_yes=True):
        auth_args = argparse.Namespace(no_open=False, skip_doctor=False)
        return cmd_auth_login(env_path, auth_args)
    return 0


def cmd_app_settings(env_path: Path) -> int:
    env = load_env_file(env_path)
    redirect_uri = normalize_redirect_uri(
        get_env_value(env, "TWITTER_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    )
    website_url = get_env_value(env, "TWITTER_WEBSITE_URL", "")
    if not website_url:
        website_url = infer_website_url(redirect_uri)
    scopes = get_env_value(env, "TWITTER_SCOPES", DEFAULT_SCOPES)
    app_name = get_env_value(env, "TWITTER_APP_NAME", "OpenClaw-AITherapy")

    print("Twitter Developer Portal Settings")
    print(f"App name: {app_name}")
    print("Type of App: Web App, Automated App or Bot (Confidential client)")
    print("App permissions: Read and write")
    print(f"Callback URI / Redirect URL: {redirect_uri}")
    print(f"Website URL: {website_url}")
    print(f"OAuth 2.0 scopes: {scopes}")
    print("OAuth 2.0 keys to copy into setup:")
    print("  - Client ID -> TWITTER_CLIENT_ID")
    print("  - Client Secret -> TWITTER_CLIENT_SECRET")
    print("")
    print("If Twitter rejects auth, confirm callback URI matches exactly.")
    return 0


def cmd_walkthrough() -> int:
    print("Twitter Engine Posting Walkthrough")
    print("1) Run `setup` and enter OAuth 2.0 Client ID + Client Secret.")
    print("2) Run `app-settings` and mirror those values in Twitter Developer Portal.")
    print("3) Run `auth-login` to open browser consent and generate OAuth2 tokens.")
    print("4) Paste callback URL back into the CLI.")
    print("5) `doctor` runs automatically to validate readiness.")
    print("6) Post with: `post --text \"hello from Twitter Engine\"`")
    return 0


def cmd_set_bearer_token(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    token = (getattr(args, "token", None) or "").strip()
    if not token:
        token = getpass.getpass("Paste TWITTER_BEARER_TOKEN: ").strip()
    if not token:
        raise TwitterHelperError("No bearer token provided.")

    env["TWITTER_BEARER_TOKEN"] = token
    write_env_file(env_path, env)
    print(f"Saved TWITTER_BEARER_TOKEN to {env_path}")

    if getattr(args, "no_verify", False):
        return 0

    status, body = api_get_with_token(
        f"{API_BASE}/tweets/search/recent?query=openclaw%20lang%3Aen%20-is%3Aretweet&max_results=10",
        token,
    )
    if status >= 400:
        print("Bearer token verification failed.")
        print(json.dumps(body, ensure_ascii=False))
        return 1
    print("Bearer token verification passed.")
    return 0


def cmd_engine_status(env_path: Path) -> int:
    status = config_status(env_path)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def cmd_engine_check(env_path: Path, args: argparse.Namespace) -> int:
    status = config_status(env_path)

    if getattr(args, "json", False):
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    print("Twitter Engine Readiness Check")
    print(f"Env file: {status['env_file']}")
    print(f"Env exists: {status['env_exists']}")
    print(f"Has client id: {status['has_client_id']}")
    print(f"Has client secret: {status['has_client_secret']}")
    print(f"Has app-only bearer token: {status['has_bearer_token']}")
    print(f"Has OAuth2 access token: {status['has_oauth2_access_token']}")
    print(f"Has OAuth2 refresh token: {status['has_oauth2_refresh_token']}")

    doctor_rc = cmd_doctor(env_path)
    if doctor_rc != 0:
        print("Twitter Engine is not ready to post yet.")
        return doctor_rc

    print("Twitter Engine is ready to post.")
    print("Try: engine-autopost --text \"Status update\"")
    return 0


def cmd_seamless(env_path: Path, args: argparse.Namespace) -> int:
    print("seamless: diagnose -> repair(if needed) -> reply-quick")
    report = diagnose_engine(
        env_path=env_path,
        skip_network=bool(getattr(args, "skip_network", False)),
        reply_target_id=None,
    )

    posting_ready = bool((report.get("posting") or {}).get("ready"))
    if (
        not posting_ready
        and not bool(getattr(args, "no_repair_auth", False))
        and sys.stdin.isatty()
        and any(a in {"run auth-login", "run setup"} for a in report.get("actions", []))
    ):
        env = load_env_file(env_path)
        has_base = bool(get_env_value(env, "TWITTER_CLIENT_ID")) and bool(
            get_env_value(env, "TWITTER_CLIENT_SECRET")
        )
        if has_base:
            print("seamless: repairing OAuth2 via auth-login")
            auth_args = argparse.Namespace(
                no_open=False,
                skip_doctor=False,
                auto_post=False,
                auto_post_text=None,
            )
            rc = cmd_auth_login(env_path, auth_args)
            if rc == 0:
                report = diagnose_engine(
                    env_path=env_path,
                    skip_network=bool(getattr(args, "skip_network", False)),
                    reply_target_id=None,
                )

    if not bool(report.get("overall_ready")):
        posting = report.get("posting", {})
        reply_scan = report.get("reply_scan", {})
        reply_post = report.get("reply_post", {})
        print(f"seamless: blocked | posting={posting.get('ready')} scan={reply_scan.get('ready')} reply_post={reply_post.get('ready')}")
        for issue in list(posting.get("issues", [])) + list(reply_scan.get("issues", [])) + list(reply_post.get("issues", [])):
            print(f"- {issue}")
        return 1

    quick_args = argparse.Namespace(
        command="reply-quick",
        handle=args.handle,
        since_id=args.since_id,
        min_confidence=args.min_confidence,
        cooldown_minutes=args.cooldown_minutes,
        dry_run=args.dry_run,
        log_path=args.log_path,
        report_path=args.report_path,
    )
    return cmd_reply_engine(quick_args)


def _run_reply_preview(
    handle: str,
    since_id: Optional[str],
    report_path: str = "data/mentions_report.json",
    log_path: str = "data/replies.jsonl",
    post: bool = False,
    verify_post: bool = True,
) -> Tuple[bool, Optional[Dict[str, object]]]:
    try:
        from reply_engine.twitter_helper import run_mentions_workflow
    except Exception as exc:
        raise TwitterHelperError(
            "Reply engine not ready. Install dependencies with: pip install -r requirements-reply-engine.txt"
        ) from exc

    result = run_mentions_workflow(
        handle=handle,
        mention_limit=1,
        since_id=since_id,
        draft_count=1,
        pick=1,
        post=post,
        max_posts=1,
        approval_queue=False,
        min_confidence=70,
        web_enrich=False,
        web_context_items=0,
        fetch_context=False,
        verify_post=verify_post,
        log_path=log_path,
        report_path=report_path,
    )
    rows = result.get("results", [])
    if not isinstance(rows, list) or not rows:
        return False, None
    first = rows[0]
    if not isinstance(first, dict):
        return False, None
    return str(first.get("status", "")) in {"drafted", "posted"}, first


def ensure_reply_scan_token(env_path: Path) -> str:
    env = load_env_file(env_path)
    token = get_read_bearer_token(env, env_path)
    # Keep downstream reply-engine module aligned with validated token.
    os.environ["TWITTER_BEARER_TOKEN"] = token
    return token


def cmd_twitter_engine(env_path: Path, args: argparse.Namespace) -> int:
    helper_path = TOOL_ROOT / "src" / "twitter_helper.py"
    wrapper_path = TOOL_ROOT / "twitter-engine"
    print("Twitter Engine")
    print(f"Workspace: {TOOL_ROOT}")
    print(f"Helper: {helper_path}")
    print(f"Wrapper: {wrapper_path}")
    print(f"Env file: {env_path.resolve()}")

    barrier = get_rate_limit_barrier_status()
    if barrier.get("active"):
        print("Decision: diagnose")
        print(
            f"Rate-limit barrier active (~{int(barrier.get('wait_seconds', 0))}s, "
            f"source={barrier.get('source', 'unknown')})."
        )
        diag_args = argparse.Namespace(
            json=bool(getattr(args, "json_diagnose", False)),
            skip_network=True,
            reply_target_id=None,
            no_repair_auth=True,
        )
        return cmd_auto_diagnose(env_path, diag_args)

    explicit_post_intent = bool(getattr(args, "text", None) or getattr(args, "file", None))
    effective_mode = args.mode
    if args.mode == "auto" and explicit_post_intent:
        effective_mode = "post"

    if effective_mode in {"auto", "diagnose"}:
        report = diagnose_engine(env_path=env_path, skip_network=False, reply_target_id=None)
        if effective_mode == "diagnose" or not bool((report.get("posting") or {}).get("ready")):
            print("Decision: diagnose")
            diag_args = argparse.Namespace(
                json=bool(getattr(args, "json_diagnose", False)),
                skip_network=False,
                reply_target_id=None,
                no_repair_auth=bool(getattr(args, "no_repair_auth", False)),
            )
            return cmd_auto_diagnose(env_path, diag_args)
    else:
        report = {}

    env = load_env_file(env_path)
    has_client_id = bool(get_env_value(env, "TWITTER_CLIENT_ID"))
    has_client_secret = bool(get_env_value(env, "TWITTER_CLIENT_SECRET"))

    if not has_client_id or not has_client_secret:
        print("Twitter app credentials missing. Launching setup...")
        setup_args = argparse.Namespace(reset=False, skip_auth_login=True)
        rc = cmd_setup(env_path, setup_args)
        if rc != 0:
            return rc

    print("Checking posting readiness...")
    doctor_rc = cmd_doctor(env_path)
    if doctor_rc != 0:
        if not sys.stdin.isatty():
            raise TwitterHelperError(
                "Auth repair requires interactive OAuth callback input. "
                "Run `auth-login` once in an interactive terminal, then rerun `twitter-engine`."
            )
        print("Attempting automatic OAuth repair via browser login...")
        auth_args = argparse.Namespace(
            no_open=False,
            skip_doctor=False,
            auto_post=False,
            auto_post_text=None,
        )
        doctor_rc = cmd_auth_login(env_path, auth_args)
        if doctor_rc != 0:
            return doctor_rc

    if effective_mode in {"auto", "reply"}:
        reply_scan_ready = bool((report.get("reply_scan") or {}).get("ready")) if isinstance(report, dict) else False
        if effective_mode == "reply" and not reply_scan_ready:
            print("Decision: diagnose")
            diag_args = argparse.Namespace(
                json=False,
                skip_network=False,
                reply_target_id=None,
                no_repair_auth=bool(getattr(args, "no_repair_auth", False)),
            )
            return cmd_auto_diagnose(env_path, diag_args)
        if reply_scan_ready:
            try:
                ensure_reply_scan_token(env_path)
            except Exception as exc:
                if effective_mode == "reply":
                    raise TwitterHelperError(
                        f"Reply mode requires valid read token: {exc}"
                    ) from exc
                print(f"[WARN] reply token validation failed: {exc}")
            try:
                has_candidate, item = _run_reply_preview(
                    handle=args.reply_handle,
                    since_id=args.reply_since_id,
                    report_path=args.report_path,
                    log_path=args.log_path,
                )
            except Exception as exc:
                print(f"[WARN] reply preview unavailable: {exc}")
                has_candidate, item = (False, None)
            if has_candidate and item is not None:
                print("Decision: reply")
                print(f"{item.get('status')} | {item.get('tweet_id')} | @{item.get('author')}")
                picked = str(item.get("picked_text", "")).strip()
                if picked:
                    print(picked)
                return 0
            if effective_mode == "reply":
                print("Decision: reply")
                print("No suitable reply candidate found.")
                return 0

    if (
        effective_mode == "auto"
        and not explicit_post_intent
        and bool(getattr(args, "gather_when_no_reply", True))
    ):
        try:
            snapshot = gather_data_snapshot(
                env_path=env_path,
                handle=args.reply_handle,
                query=getattr(args, "gather_query", 'openclaw OR "local ai agent" lang:en -is:retweet'),
                limit=getattr(args, "gather_limit", 20),
                max_pages=getattr(args, "gather_max_pages", 1),
            )
            search_count = int(snapshot.get("search_count", 0))
            mentions_count = int(snapshot.get("mentions_count", 0))
            if search_count > 0 or mentions_count > 0:
                out = Path(getattr(args, "gather_output", "data/engine_data_snapshot.json"))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                print("Decision: gather-data")
                print(f"Saved data snapshot -> {out}")
                print(f"Mentions: {mentions_count} | Search: {search_count}")
                return 0
        except Exception as exc:
            print(f"[WARN] gather-data step failed: {exc}")

    if args.no_post:
        print("Readiness is healthy. Skipping post (--no-post).")
        return 0

    print("Decision: post")
    cfg, env_values = resolve_config(env_path)
    base_text = args.text
    if args.file:
        base_text = Path(args.file).read_text(encoding="utf-8").strip()
    if not base_text:
        web_items, web_err = maybe_collect_web_inspiration(
            enabled=bool(getattr(args, "web_inspiration", True)),
            query=str(getattr(args, "web_query", "ai agents automation reliability")),
            item_limit=int(getattr(args, "web_items", 8)),
        )
        if web_err:
            print(f"[WARN] web inspiration unavailable: {web_err}")
        elif web_items:
            print(f"[INFO] web inspiration items: {len(web_items)}")
        base_text = generate_reflective_post_text(
            topic=str(getattr(args, "web_query", "ai agents")),
            inspiration_texts=web_items,
            style=str(getattr(args, "style", "auto") or "auto"),
            voice=str(getattr(args, "voice", "auto") or "auto"),
            viral_pack=str(getattr(args, "viral_pack", "") or ""),
            anti_boring=bool(getattr(args, "anti_boring", False)),
            sharpen=bool(getattr(args, "sharpen", False)),
            judge_threshold=float(getattr(args, "judge_threshold", 82.0)),
            max_attempts=int(getattr(args, "max_attempts", 7)),
            ensemble_size=int(getattr(args, "ensemble", 1)),
            viral_boost=bool(getattr(args, "viral_boost", False)),
        )
    unique_text = make_unique_public_tweet(base_text)
    post_args = argparse.Namespace(text=unique_text, file=None, dry_run=args.dry_run)
    return cmd_engine_autopost(cfg, env_path, env_values, post_args)


def cmd_restart_setup(env_path: Path, args: argparse.Namespace) -> int:
    print("Twitter Engine Restart Recovery")
    print("Goal: restore setup/auth health after restart without posting.")

    env = load_env_file(env_path)
    has_client_id = bool(get_env_value(env, "TWITTER_CLIENT_ID"))
    has_client_secret = bool(get_env_value(env, "TWITTER_CLIENT_SECRET"))

    if not has_client_id or not has_client_secret:
        print("Missing app credentials. Launching setup...")
        setup_args = argparse.Namespace(reset=False, skip_auth_login=False)
        rc = cmd_setup(env_path, setup_args)
        if rc != 0:
            return rc
    else:
        print("App credentials found.")

    print("Running doctor...")
    doctor_rc = cmd_doctor(env_path)
    if doctor_rc == 0:
        print("Recovery complete. Posting is healthy.")
        return 0

    if not sys.stdin.isatty():
        raise TwitterHelperError(
            "Recovery requires interactive OAuth callback input. "
            "Run `auth-login` once in an interactive terminal."
        )

    print("Doctor failed. Launching OAuth login repair...")
    auth_args = argparse.Namespace(
        no_open=False,
        skip_doctor=False,
        auto_post=False,
        auto_post_text=None,
    )
    auth_rc = cmd_auth_login(env_path, auth_args)
    if auth_rc != 0:
        return auth_rc

    print("Recovery complete. Posting is healthy.")
    return 0


def post_with_retry(
    cfg: Config,
    env_path: Path,
    env_values: Dict[str, str],
    text: str,
    reply_to_id: Optional[str] = None,
    media_ids: Optional[List[str]] = None,
    unique_on_duplicate: bool = False,
) -> Tuple[Config, Tuple[int, Dict[str, object]]]:
    run_tag = unique_marker("twitter-engine")
    fresh = ensure_auth(cfg, env_path, env_values)
    status, body = post_tweet(
        fresh,
        text,
        reply_to_id=reply_to_id,
        media_ids=media_ids,
        run_tag=run_tag,
    )
    # Retry once only for auth-expired failures.
    if status == 401:
        fresh = refresh_tokens(fresh, env_path, env_values)
        status, body = post_tweet(
            fresh,
            text,
            reply_to_id=reply_to_id,
            media_ids=media_ids,
            run_tag=run_tag,
        )
    if unique_on_duplicate and is_duplicate_content_error(status, body):
        print("[INFO] Duplicate content rejected; retrying once with unique reply suffix.")
        unique_text = make_unique_reply_tweet(text)
        status, body = post_tweet(
            fresh,
            unique_text,
            reply_to_id=reply_to_id,
            media_ids=media_ids,
            run_tag=run_tag,
        )
    return fresh, (status, body)


def read_text_from_args(args: argparse.Namespace) -> str:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    if not text:
        text = input("Tweet text: ").strip()
    if not text:
        raise TwitterHelperError("No tweet text provided.")
    return text


def cmd_engine_autopost(
    cfg: Config, env_path: Path, env_values: Dict[str, str], args: argparse.Namespace
) -> int:
    base_text = read_text_from_args(args)
    text = sanitize_public_text(base_text)
    validate_tweet_len(text)

    if args.dry_run:
        print(text)
        return 0

    print(f"Posting Twitter Engine auto-tweet ({len(text)}/{MAX_TWEET_LEN} chars)...")
    fresh, (status, body) = post_with_retry(cfg, env_path, env_values, text)
    if status < 200 or status >= 300:
        raise TwitterHelperError(
            f"Auto-post failed ({status}): {json.dumps(body, ensure_ascii=False)}"
        )
    data = body.get("data") if isinstance(body, dict) else None
    tweet_id = data.get("id") if isinstance(data, dict) else None
    if not tweet_id:
        raise TwitterHelperError("Auto-post returned no tweet id.")
    _, tweet_url = verify_post_visible(fresh, str(tweet_id))
    record_recent_post(text)
    record_tweet_memory(
        kind="post",
        text=text,
        tweet_id=str(tweet_id),
        url=tweet_url,
        meta={"source": "engine-autopost"},
    )
    print(f"Twitter Engine auto-tweet posted and verified: id={tweet_id}")
    print(f"URL: {tweet_url}")
    return 0


def cmd_auth_login(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
    tm = token_manager(env_path)
    tm.migrate_from_env(env)
    client_id = get_env_value(env, "TWITTER_CLIENT_ID")
    client_secret = get_env_value(env, "TWITTER_CLIENT_SECRET")

    if not client_id:
        raise TwitterHelperError("Missing TWITTER_CLIENT_ID. Run `setup` first.")
    if not client_secret:
        raise TwitterHelperError("Missing TWITTER_CLIENT_SECRET. Run `setup` first.")

    redirect_uri = normalize_redirect_uri(
        get_env_value(env, "TWITTER_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    )
    scopes = get_env_value(env, "TWITTER_SCOPES", DEFAULT_SCOPES)

    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(16)

    auth_url = build_auth_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
        state=state,
        code_challenge=code_challenge,
    )

    print("Twitter OAuth Login Wizard")
    print("1) Browser will open the Twitter authorization page.")
    print("2) Approve access for your app.")
    print("3) Paste the full callback URL here (or just the `code` value).\n")

    if not args.no_open:
        opened = webbrowser.open(auth_url)
        if opened:
            print("Opened browser for authorization.")
        else:
            print("Could not auto-open browser. Use this URL manually:")
            print(auth_url)
    else:
        print("Open this URL in your browser:")
        print(auth_url)

    try:
        callback_input = input("\nPaste callback URL (or code): ").strip()
    except EOFError as exc:
        raise TwitterHelperError(
            "No callback input detected. Re-run `auth-login` in an interactive terminal."
        ) from exc
    if not callback_input:
        raise TwitterHelperError("No callback input provided.")

    code, returned_state = parse_callback_input(callback_input)
    if returned_state and returned_state != state:
        raise TwitterHelperError("OAuth state mismatch. Please run `auth-login` again.")

    print("Exchanging authorization code for tokens...")
    access_token, refresh_token = exchange_authorization_code(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
    )

    env["TWITTER_CLIENT_ID"] = client_id
    env["TWITTER_CLIENT_SECRET"] = client_secret
    env["TWITTER_REDIRECT_URI"] = redirect_uri
    env["TWITTER_WEBSITE_URL"] = get_env_value(
        env, "TWITTER_WEBSITE_URL", infer_website_url(redirect_uri)
    )
    env["TWITTER_SCOPES"] = scopes
    storage = tm.save_tokens(access_token, refresh_token, env)
    print(f"OAuth login complete. Tokens saved ({storage}).")
    if getattr(args, "skip_doctor", False):
        print("Next step: run `doctor`.")
        return 0

    print("Running doctor automatically...")
    doctor_rc = cmd_doctor(env_path)
    if doctor_rc == 0:
        print("Auth + doctor checks passed. You can post now.")
    if doctor_rc != 0:
        return doctor_rc

    if not getattr(args, "auto_post", False):
        return 0

    auto_text = (
        args.auto_post_text.strip()
        if getattr(args, "auto_post_text", None)
        else "Twitter Engine auth complete and posting pipeline is live."
    )

    cfg = Config(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    env_values = load_env_file(env_path)
    auto_args = argparse.Namespace(text=auto_text, file=None, dry_run=False)
    print("Auto-post requested. Posting confirmation tweet...")
    return cmd_engine_autopost(cfg, env_path, env_values, auto_args)


def cmd_doctor(env_path: Path) -> int:
    print("Twitter Engine Doctor")
    print(f"Env file: {env_path}")

    env = load_env_file(env_path)
    tm = token_manager(env_path)
    migrated = tm.migrate_from_env(env)
    if migrated:
        print("[PASS] Migrated OAuth2 tokens from .env to keyring.")
    print(f"Keyring status: {tm.keyring_status_label()}")
    if not env_path.exists():
        print("[FAIL] Env file does not exist.")
        print("Run: setup")
        return 1

    missing_base = [
        k for k in ["TWITTER_CLIENT_ID", "TWITTER_CLIENT_SECRET"] if not get_env_value(env, k)
    ]
    if missing_base:
        print(f"[FAIL] Missing app config: {', '.join(missing_base)}")
        print("Run: setup")
        return 1

    missing_tokens = []
    if not tm.get_access_token(env):
        missing_tokens.append("TWITTER_OAUTH2_ACCESS_TOKEN (or keyring token)")
    if not tm.get_refresh_token(env):
        missing_tokens.append("TWITTER_OAUTH2_REFRESH_TOKEN (or keyring token)")
    if missing_tokens:
        print(f"[FAIL] Missing OAuth tokens: {', '.join(missing_tokens)}")
        print("Run: auth-login (this opens the OAuth2 browser flow to generate tokens)")
        return 1

    print("[PASS] Config values are present.")
    scopes = get_env_value(env, "TWITTER_SCOPES", DEFAULT_SCOPES)
    if "media.write" not in scopes.split():
        print("[WARN] media.write scope not configured. Media upload requires re-auth with media.write scope.")
    else:
        print("[PASS] Media upload ready (multi-image, size/MIME validation enabled).")
    if not get_env_value(env, "TWITTER_BEARER_TOKEN"):
        print(
            "[WARN] TWITTER_BEARER_TOKEN is missing. "
            "App-only read/scan reply workflows may not work."
        )

    try:
        cfg, env_values = resolve_config(env_path)
        cfg = ensure_auth(cfg, env_path, env_values)
        status, body, headers = me_with_headers(cfg)
        if status != 200:
            print(f"[FAIL] Auth check failed with status {status}")
            print(json.dumps(body, ensure_ascii=False))
            return 1

        data = body.get("data") if isinstance(body, dict) else None
        username = data.get("username") if isinstance(data, dict) else None
        user_id = data.get("id") if isinstance(data, dict) else None
        print(f"[PASS] API auth works as @{username} (id={user_id}).")
        remaining = headers.get("x-rate-limit-remaining", "N/A")
        reset = headers.get("x-rate-limit-reset", "N/A")
        print(f"[INFO] Rate limit remaining: {remaining} (reset: {reset})")
        print("You're ready. Try: post --text \"hello from twitter-engine\"")
        return 0
    except TwitterHelperError as exc:
        print(f"[FAIL] {exc}")
        return 1


def diagnose_engine(
    env_path: Path,
    skip_network: bool = False,
    reply_target_id: Optional[str] = None,
) -> Dict[str, object]:
    env = load_env_file(env_path)
    tm = token_manager(env_path)
    tm.migrate_from_env(env)
    report: Dict[str, object] = {
        "env_file": str(env_path),
        "env_exists": env_path.exists(),
        "rate_limit": get_rate_limit_barrier_status(),
        "posting": {"ready": False, "issues": []},
        "reply_scan": {"ready": False, "issues": []},
        "reply_post": {"ready": False, "issues": []},
        "actions": [],
    }

    posting_issues: List[str] = []
    reply_scan_issues: List[str] = []
    reply_post_issues: List[str] = []
    actions: List[str] = []

    rate_info = report.get("rate_limit", {})
    if isinstance(rate_info, dict) and bool(rate_info.get("active")):
        wait_seconds = int(rate_info.get("wait_seconds", 0))
        src = str(rate_info.get("source", "")).strip()
        posting_issues.append(
            f"Rate-limit barrier active (~{wait_seconds}s remaining, source={src or 'unknown'})."
        )
        reply_scan_issues.append(
            f"Rate-limit barrier active (~{wait_seconds}s remaining, source={src or 'unknown'})."
        )
        actions.append(f"wait {wait_seconds}s before retrying read/post diagnostics")

    if not report["env_exists"]:
        posting_issues.append("Missing .env file.")
        reply_scan_issues.append("Missing .env file.")
        reply_post_issues.append("Missing .env file.")
        actions.append("run setup")
    else:
        missing_base = [
            k for k in ["TWITTER_CLIENT_ID", "TWITTER_CLIENT_SECRET"] if not get_env_value(env, k)
        ]
        missing_tokens = []
        if not tm.get_access_token(env):
            missing_tokens.append("TWITTER_OAUTH2_ACCESS_TOKEN (or keyring token)")
        if not tm.get_refresh_token(env):
            missing_tokens.append("TWITTER_OAUTH2_REFRESH_TOKEN (or keyring token)")
        if missing_base:
            posting_issues.append(f"Missing app config: {', '.join(missing_base)}")
            actions.append("run setup")
        if missing_tokens:
            posting_issues.append(f"Missing OAuth2 tokens: {', '.join(missing_tokens)}")
            actions.append("run auth-login")

    cfg: Optional[Config] = None
    env_values: Dict[str, str] = {}
    if not posting_issues:
        try:
            cfg, env_values = resolve_config(env_path)
            if not skip_network:
                cfg = ensure_auth(cfg, env_path, env_values)
                me_status, me_body = me(cfg)
                if me_status != 200:
                    posting_issues.append(
                        f"OAuth2 auth check failed ({me_status}): {json.dumps(me_body, ensure_ascii=False)}"
                    )
                    actions.append("run auth-login")
        except TwitterHelperError as exc:
            posting_issues.append(str(exc))
            actions.append("run auth-login")

    bearer = get_env_value(env, "TWITTER_BEARER_TOKEN")
    if not bearer:
        reply_scan_issues.append("Missing TWITTER_BEARER_TOKEN for browse/mentions scanning.")
        actions.append("add TWITTER_BEARER_TOKEN in setup")
    elif not skip_network:
        scan_status, scan_body = api_get_with_token(
            f"{API_BASE}/tweets/search/recent?query=openclaw%20lang%3Aen%20-is%3Aretweet&max_results=10",
            bearer,
        )
        if scan_status == 429:
            reply_scan_issues.append(
                "Bearer scan temporarily rate-limited (429)."
            )
            actions.append("wait for rate-limit barrier cooldown, then retry")
        elif scan_status >= 400:
            reply_scan_issues.append(
                f"Bearer scan check failed ({scan_status}): {json.dumps(scan_body, ensure_ascii=False)}"
            )
            actions.append("replace TWITTER_BEARER_TOKEN with a valid App-Only token")

    try:
        reply_engine_spec = importlib.util.find_spec("reply_engine.twitter_helper")
    except ModuleNotFoundError:
        reply_engine_spec = None
    if reply_engine_spec is None:
        reply_post_issues.append("Reply engine module not importable.")
        actions.append("pip install -e .")
        actions.append("pip install -r requirements-reply-engine.txt")

    # Reply posting uses shared OAuth2 path via the main helper. OAuth1 keys are not required.

    if reply_target_id:
        if cfg is None:
            reply_post_issues.append("Cannot preflight reply target because OAuth2 posting auth is not ready.")
        elif skip_network:
            reply_post_issues.append("Reply target preflight skipped due to --skip-network.")
        else:
            target_status, target_body = fetch_tweet(cfg, reply_target_id)
            if target_status >= 400 or (
                isinstance(target_body, dict) and target_body.get("errors")
            ):
                reply_post_issues.append(
                    "Reply target is not visible/deleted for this account: "
                    f"{json.dumps(target_body, ensure_ascii=False)}"
                )
                actions.append("use a visible tweet ID for --in-reply-to")

    report["posting"] = {"ready": len(posting_issues) == 0, "issues": posting_issues}
    report["reply_scan"] = {"ready": len(reply_scan_issues) == 0, "issues": reply_scan_issues}
    report["reply_post"] = {"ready": len(reply_post_issues) == 0, "issues": reply_post_issues}
    report["overall_ready"] = (
        report["posting"]["ready"] and report["reply_scan"]["ready"] and report["reply_post"]["ready"]
    )

    # Keep actions deterministic and duplicate-free.
    seen = set()
    deduped = []
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        deduped.append(action)
    report["actions"] = deduped
    return report


def diagnose_openclaw(
    env_path: Path,
    skip_network: bool = False,
    reply_target_id: Optional[str] = None,
) -> Dict[str, object]:
    return diagnose_engine(
        env_path=env_path,
        skip_network=skip_network,
        reply_target_id=reply_target_id,
    )


def cmd_auto_diagnose(env_path: Path, args: argparse.Namespace) -> int:
    print("Twitter Engine Auto Diagnose")
    print(f"Env file: {env_path}")
    report = diagnose_engine(
        env_path=env_path,
        skip_network=args.skip_network,
        reply_target_id=args.reply_target_id,
    )

    # Optional OAuth2 self-heal pass for interactive runs.
    posting_ready = bool((report.get("posting") or {}).get("ready"))
    if (
        not posting_ready
        and not args.no_repair_auth
        and sys.stdin.isatty()
        and any(a in {"run auth-login", "run setup"} for a in report.get("actions", []))
    ):
        env = load_env_file(env_path)
        has_base = bool(get_env_value(env, "TWITTER_CLIENT_ID")) and bool(
            get_env_value(env, "TWITTER_CLIENT_SECRET")
        )
        if has_base:
            print("Posting auth is unhealthy. Attempting interactive OAuth2 repair...")
            auth_args = argparse.Namespace(
                no_open=False,
                skip_doctor=False,
                auto_post=False,
                auto_post_text=None,
            )
            rc = cmd_auth_login(env_path, auth_args)
            if rc == 0:
                print("Re-running diagnostics after OAuth2 repair...")
                report = diagnose_engine(
                    env_path=env_path,
                    skip_network=args.skip_network,
                    reply_target_id=args.reply_target_id,
                )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        rate_info = report.get("rate_limit", {})
        if isinstance(rate_info, dict) and bool(rate_info.get("active")):
            print(
                f"[WARN] rate-limit barrier active: wait ~{int(rate_info.get('wait_seconds', 0))}s "
                f"(source={rate_info.get('source', 'unknown')})"
            )
        posting = report.get("posting", {})
        reply_scan = report.get("reply_scan", {})
        reply_post = report.get("reply_post", {})
        print(
            f"[{'PASS' if posting.get('ready') else 'FAIL'}] posting readiness"
        )
        for issue in posting.get("issues", []):
            print(f"  - {issue}")
        print(
            f"[{'PASS' if reply_scan.get('ready') else 'FAIL'}] reply scan readiness"
        )
        for issue in reply_scan.get("issues", []):
            print(f"  - {issue}")
        print(
            f"[{'PASS' if reply_post.get('ready') else 'FAIL'}] reply post readiness"
        )
        for issue in reply_post.get("issues", []):
            print(f"  - {issue}")
        actions = report.get("actions", [])
        if actions:
            print("Suggested fixes:")
            for action in actions:
                print(f"  - {action}")
        if report.get("overall_ready"):
            print("All posting/reply paths are healthy.")

    return 0 if report.get("overall_ready") else 1


def cmd_check_auth(cfg: Config, env_path: Path, env_values: Dict[str, str]) -> int:
    print("Validating credentials with Twitter API...")
    fresh = ensure_auth(cfg, env_path, env_values)
    status, body = me(fresh)
    if status != 200:
        raise TwitterHelperError(
            f"Failed to get user info ({status}): {json.dumps(body, ensure_ascii=False)}"
        )

    data = body.get("data") if isinstance(body, dict) else None
    username = data.get("username") if isinstance(data, dict) else None
    user_id = data.get("id") if isinstance(data, dict) else None
    print(f"Authenticated as @{username} (id={user_id})")
    return 0


def cmd_post(
    cfg: Config, env_path: Path, env_values: Dict[str, str], args: argparse.Namespace
) -> int:
    if args.text or args.file:
        text = read_text_from_args(args)
    else:
        if getattr(args, "in_reply_to", None):
            raise TwitterHelperError("Reply posts require --text or --file.")
        web_items, web_err = maybe_collect_web_inspiration(
            enabled=True,
            query="ai agents automation reliability openclaw",
            item_limit=8,
        )
        if web_err:
            print(f"[WARN] web inspiration unavailable: {web_err}")
        elif web_items:
            print(f"[INFO] web inspiration items: {len(web_items)}")
        text = generate_reflective_post_text(
            topic="ai agents automation reliability",
            inspiration_texts=web_items,
            style=str(getattr(args, "style", "auto") or "auto"),
            voice=str(getattr(args, "voice", "auto") or "auto"),
            viral_pack=str(getattr(args, "viral_pack", "") or ""),
            anti_boring=bool(getattr(args, "anti_boring", False)),
            sharpen=bool(getattr(args, "sharpen", False)),
            judge_threshold=float(getattr(args, "judge_threshold", 82.0)),
            max_attempts=int(getattr(args, "max_attempts", 7)),
            ensemble_size=int(getattr(args, "ensemble", 1)),
            viral_boost=bool(getattr(args, "viral_boost", False)),
        )
        text = make_unique_public_tweet(text)
        print("[INFO] No --text provided; generated autonomous post text.")
    text = sanitize_public_text(text)
    if getattr(args, "unique", False):
        text = make_unique_public_tweet(text)

    validate_tweet_len(text)
    if getattr(args, "dry_run", False):
        print(text)
        if getattr(args, "media", None):
            print(f"dry-run: would attach media from {args.media}")
        return 0

    print(f"Posting tweet ({len(text)}/{MAX_TWEET_LEN} chars)...")

    fresh = ensure_auth(cfg, env_path, env_values)
    media_ids: Optional[List[str]] = None
    if getattr(args, "media", None):
        media_list = [x.strip() for x in str(args.media).split(",") if x.strip()]
        alt_list = (
            [x.strip() for x in str(getattr(args, "alt_text", "")).split(",")]
            if getattr(args, "alt_text", None)
            else None
        )
        media_ids = upload_media(
            access_token=fresh.access_token,
            media_inputs=media_list,
            alt_texts=alt_list,
        )

    if args.in_reply_to:
        if has_replied_to_target(args.in_reply_to) and not getattr(args, "force_reply_target", False):
            raise TwitterHelperError(
                "Refusing to double-reply to this target. "
                "Use --force-reply-target to override intentionally."
            )
        check_status, check_body = fetch_tweet(fresh, args.in_reply_to)
        if check_status >= 400:
            raise TwitterHelperError(
                "Reply target could not be loaded "
                f"({check_status}): {json.dumps(check_body, ensure_ascii=False)}"
            )
        if isinstance(check_body, dict) and check_body.get("errors"):
            raise TwitterHelperError(
                "Reply target is not visible/deleted for this account: "
                f"{json.dumps(check_body.get('errors'), ensure_ascii=False)}"
            )

    fresh, (status, body) = post_with_retry(
        fresh,
        env_path,
        env_values,
        text,
        reply_to_id=args.in_reply_to,
        media_ids=media_ids,
        unique_on_duplicate=bool(args.in_reply_to),
    )

    if status < 200 or status >= 300:
        raise TwitterHelperError(
            f"Post failed ({status}): {json.dumps(body, ensure_ascii=False)}"
        )

    data = body.get("data") if isinstance(body, dict) else None
    tweet_id = data.get("id") if isinstance(data, dict) else None
    if not tweet_id:
        raise TwitterHelperError("Post returned no tweet id.")
    _, tweet_url = verify_post_visible(fresh, str(tweet_id))
    if args.in_reply_to:
        mark_replied_target(str(args.in_reply_to), str(tweet_id), source="post")
        record_tweet_memory(
            kind="reply",
            text=text,
            tweet_id=str(tweet_id),
            url=tweet_url,
            meta={"in_reply_to": str(args.in_reply_to), "source": "post"},
        )
    else:
        record_tweet_memory(
            kind="post",
            text=text,
            tweet_id=str(tweet_id),
            url=tweet_url,
            meta={"source": "post"},
        )
    print(f"Tweet posted and verified: id={tweet_id}")
    print(f"URL: {tweet_url}")
    return 0


def cmd_thread(
    cfg: Config, env_path: Path, env_values: Dict[str, str], args: argparse.Namespace
) -> int:
    thread_path = Path(args.file)
    tweets = parse_thread_file(thread_path)
    for t in tweets:
        validate_tweet_len(t)

    print(f"Posting thread from {thread_path} with {len(tweets)} tweet(s)...")
    fresh = cfg
    parent_id: Optional[str] = None

    for idx, text in enumerate(tweets, start=1):
        fresh, (status, body) = post_with_retry(
            fresh, env_path, env_values, text, reply_to_id=parent_id
        )

        if status < 200 or status >= 300:
            raise TwitterHelperError(
                f"Thread post {idx} failed ({status}): {json.dumps(body, ensure_ascii=False)}"
            )

        data = body.get("data") if isinstance(body, dict) else None
        tweet_id = data.get("id") if isinstance(data, dict) else None
        if not tweet_id:
            raise TwitterHelperError(f"Thread post {idx} returned no tweet id")

        parent_id = tweet_id
        _, tweet_url = verify_post_visible(fresh, str(tweet_id))
        record_tweet_memory(
            kind="thread_post",
            text=text,
            tweet_id=str(tweet_id),
            url=tweet_url,
            meta={"index": idx, "total": len(tweets)},
        )
        print(f"Posted and verified {idx}/{len(tweets)}: id={tweet_id}")
        print(f"URL: {tweet_url}")

    print("Thread posted successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Twitter Engine",
        epilog="Recommended flow: setup -> app-settings -> auth-login -> doctor -> post",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file (default: .env)",
    )
    parser.add_argument(
        "--account",
        default="default",
        help="Account namespace for secure token storage (default: default)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Interactive setup wizard (app config)")
    p_setup.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing values and prompt for all fields",
    )
    p_setup.add_argument(
        "--skip-auth-login",
        action="store_true",
        help="Do not offer immediate handoff to OAuth2 browser login",
    )

    p_auth = sub.add_parser("auth-login", help="OAuth browser wizard to get tokens")
    p_auth.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open browser; print URL instead",
    )
    p_auth.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Do not run doctor automatically after token exchange",
    )
    p_auth.add_argument(
        "--auto-post",
        action="store_true",
        help="After successful login+doctor, post a confirmation tweet",
    )
    p_auth.add_argument(
        "--auto-post-text",
        help="Text to use for --auto-post",
    )

    sub.add_parser("doctor", help="Run guided diagnostics for config + auth")
    p_set_bearer = sub.add_parser(
        "set-bearer-token",
        help="Set or replace TWITTER_BEARER_TOKEN in env file",
    )
    p_set_bearer.add_argument(
        "--token",
        help="Bearer token value (if omitted, prompt securely)",
    )
    p_set_bearer.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip live API verification after saving token",
    )
    p_diag = sub.add_parser(
        "auto-diagnose",
        help="Auto-diagnose posting and replying readiness, with optional OAuth2 self-repair",
    )
    p_diag.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable diagnostics JSON",
    )
    p_diag.add_argument(
        "--skip-network",
        action="store_true",
        help="Only validate local config/deps without API calls",
    )
    p_diag.add_argument(
        "--reply-target-id",
        help="Optional tweet ID to preflight visibility for replies",
    )
    p_diag.add_argument(
        "--no-repair-auth",
        action="store_true",
        help="Disable interactive OAuth2 repair attempt when posting auth is unhealthy",
    )
    sub.add_parser("app-settings", help="Print exact Twitter app settings to use")
    sub.add_parser("walkthrough", help="Print end-to-end setup/posting walkthrough")
    sub.add_parser(
        "engine-status",
        aliases=["openclaw-status"],
        help="Print machine-readable readiness JSON",
    )
    sub.add_parser("check-auth", help="Validate auth and print current account")
    p_memory = sub.add_parser("memory", help="Show tweet/reply memory history")
    p_memory.add_argument("--limit", type=int, default=20)
    p_memory.add_argument("--json", action="store_true")
    p_memory.add_argument("--account-name", help="Optional account override for viewing memory")

    p_post = sub.add_parser("post", help="Post a single tweet")
    p_post.add_argument("--text", help="Tweet text")
    p_post.add_argument("--file", help="Path to text file")
    p_post.add_argument(
        "--style",
        choices=["auto", "contrarian", "operator", "story"],
        default="auto",
        help="Autonomous post style when no --text/--file is provided",
    )
    p_post.add_argument(
        "--voice",
        choices=["auto", "chaotic", "degen", "based", "savage", "operator", "sage", "shitposter"],
        default="auto",
        help="Autonomous voice profile for generated posts",
    )
    p_post.add_argument(
        "--viral-pack",
        choices=["auto", "light", "medium", "nuclear", "alpha", "chaos", "infinite"],
        default="",
        help="Single-flag preset for voice/style/ensemble/judge/viral settings",
    )
    p_post.add_argument(
        "--anti-boring",
        action="store_true",
        help="Enable frog-chaos anti-boring generation constraints for autonomous posts",
    )
    p_post.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply high-signal lexicon sharpening to autonomous post generation",
    )
    p_post.add_argument(
        "--judge-threshold",
        type=float,
        default=82.0,
        help="Frog Judge minimum score (0-100) required for autonomous post acceptance",
    )
    p_post.add_argument(
        "--max-attempts",
        type=int,
        default=7,
        help="Max autonomous regeneration attempts before returning best draft",
    )
    p_post.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="Generate N variants per attempt and pick the top-scoring draft (max 8)",
    )
    p_post.add_argument(
        "--viral-boost",
        action="store_true",
        help="Increase viral-potential weighting in Frog Judge",
    )
    p_post.add_argument(
        "--unique",
        action="store_true",
        help="Force uniqueness handling for duplicate-content errors",
    )
    p_post.add_argument(
        "--in-reply-to",
        help="Optional tweet ID to post as a reply",
    )
    p_post.add_argument(
        "--force-reply-target",
        action="store_true",
        help="Override double-reply guard for --in-reply-to target",
    )
    p_post.add_argument(
        "--media",
        help="Optional comma-separated image paths/URLs to attach (max 5 images)",
    )
    p_post.add_argument(
        "--alt-text",
        help="Optional comma-separated alt text values matching --media order",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="Print final tweet text (and media intent) without posting",
    )

    p_oc_auto = sub.add_parser(
        "engine-autopost",
        aliases=["openclaw-autopost"],
        help="Twitter Engine integration post",
    )
    p_oc_auto.add_argument("--text", help="Base tweet text")
    p_oc_auto.add_argument("--file", help="Path to base text file")
    p_oc_auto.add_argument(
        "--dry-run",
        action="store_true",
        help="Print final unique tweet text instead of posting",
    )
    p_openclaw = sub.add_parser(
        "engine-check",
        aliases=["openclaw"],
        help="Run Twitter Engine readiness check for posting",
    )
    p_openclaw.add_argument(
        "--json",
        action="store_true",
        help="Print readiness data as JSON",
    )

    p_thread = sub.add_parser("thread", help="Post a thread from file")
    p_thread.add_argument("--file", required=True, help="Path to thread text file")

    p_run = sub.add_parser(
        "twitter-engine",
        aliases=["run-twitter-helper"],
        help="One-command Twitter Engine flow: check, repair auth if needed, then post unique tweet",
    )
    p_run.add_argument(
        "--mode",
        choices=["auto", "post", "reply", "diagnose"],
        default="auto",
        help="Decision mode (default: auto)",
    )
    p_run.add_argument("--text", help="Base tweet text")
    p_run.add_argument("--file", help="Path to base text file")
    p_run.add_argument(
        "--reply-handle",
        default="OpenClawAI",
        help="Handle to inspect for reply opportunities in auto/reply mode",
    )
    p_run.add_argument(
        "--reply-since-id",
        help="Only consider mentions newer than this tweet ID in auto/reply mode",
    )
    p_run.add_argument(
        "--no-repair-auth",
        action="store_true",
        help="Disable interactive OAuth2 repair attempt during diagnose mode",
    )
    p_run.add_argument(
        "--json-diagnose",
        action="store_true",
        help="When mode triggers diagnosis, print diagnostic report as JSON",
    )
    p_run.add_argument("--log-path", default="data/replies.jsonl")
    p_run.add_argument("--report-path", default="data/mentions_report.json")
    p_run.add_argument(
        "--gather-when-no-reply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In auto mode, gather data snapshot when no reply candidate is found",
    )
    p_run.add_argument(
        "--gather-query",
        default='openclaw OR "local ai agent" lang:en -is:retweet',
        help="Query used for gather-data decision step in auto mode",
    )
    p_run.add_argument("--gather-limit", type=int, default=20)
    p_run.add_argument("--gather-max-pages", type=int, default=1)
    p_run.add_argument("--gather-output", default="data/engine_data_snapshot.json")
    p_run.add_argument(
        "--no-post",
        action="store_true",
        help="Only verify/repair readiness; do not post",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print final tweet text instead of posting",
    )
    p_run.add_argument(
        "--web-inspiration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use internet headlines/search signals to inspire generated posts",
    )
    p_run.add_argument(
        "--web-query",
        default="ai agents automation reliability",
        help="Query used for internet inspiration in post mode",
    )
    p_run.add_argument(
        "--web-items",
        type=int,
        default=8,
        help="Max inspiration items pulled from internet sources",
    )
    p_run.add_argument(
        "--style",
        choices=["auto", "contrarian", "operator", "story"],
        default="auto",
        help="Autonomous post style when mode resolves to post",
    )
    p_run.add_argument(
        "--voice",
        choices=["auto", "chaotic", "degen", "based", "savage", "operator", "sage", "shitposter"],
        default="auto",
        help="Autonomous voice profile when mode resolves to post",
    )
    p_run.add_argument(
        "--viral-pack",
        choices=["auto", "light", "medium", "nuclear", "alpha", "chaos", "infinite"],
        default="",
        help="Single-flag preset for voice/style/ensemble/judge/viral settings",
    )
    p_run.add_argument(
        "--anti-boring",
        action="store_true",
        help="Enable frog-chaos anti-boring generation constraints for autonomous posts",
    )
    p_run.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply high-signal lexicon sharpening to autonomous post generation",
    )
    p_run.add_argument(
        "--judge-threshold",
        type=float,
        default=82.0,
        help="Frog Judge minimum score (0-100) required for autonomous post acceptance",
    )
    p_run.add_argument(
        "--max-attempts",
        type=int,
        default=7,
        help="Max autonomous regeneration attempts before returning best draft",
    )
    p_run.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="Generate N variants per attempt and pick the top-scoring draft (max 8)",
    )
    p_run.add_argument(
        "--viral-boost",
        action="store_true",
        help="Increase viral-potential weighting in Frog Judge",
    )

    p_kit = sub.add_parser(
        "kit",
        help="All-in-one operator kit (diagnose/reply/gather/post with one command)",
    )
    p_kit.add_argument(
        "--mode",
        choices=["auto", "post", "reply", "diagnose"],
        default="auto",
    )
    p_kit.add_argument("--text")
    p_kit.add_argument("--file")
    p_kit.add_argument("--dry-run", action="store_true")
    p_kit.add_argument("--json", action="store_true")
    p_kit.add_argument("--reply-handle", default="OpenClawAI")
    p_kit.add_argument("--reply-since-id")
    p_kit.add_argument("--log-path", default="data/replies.jsonl")
    p_kit.add_argument("--report-path", default="data/mentions_report.json")
    p_kit.add_argument(
        "--gather-when-no-reply",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_kit.add_argument(
        "--gather-query",
        default='openclaw OR "local ai agent" lang:en -is:retweet',
    )
    p_kit.add_argument("--gather-limit", type=int, default=20)
    p_kit.add_argument("--gather-max-pages", type=int, default=1)
    p_kit.add_argument("--gather-output", default="data/engine_data_snapshot.json")
    p_kit.add_argument(
        "--web-inspiration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use internet headlines/search signals to inspire generated posts",
    )
    p_kit.add_argument(
        "--web-query",
        default="ai agents automation reliability",
    )
    p_kit.add_argument("--web-items", type=int, default=8)
    p_kit.add_argument(
        "--style",
        choices=["auto", "contrarian", "operator", "story"],
        default="auto",
    )
    p_kit.add_argument(
        "--voice",
        choices=["auto", "chaotic", "degen", "based", "savage", "operator", "sage", "shitposter"],
        default="auto",
    )
    p_kit.add_argument(
        "--viral-pack",
        choices=["auto", "light", "medium", "nuclear", "alpha", "chaos", "infinite"],
        default="",
        help="Single-flag preset for voice/style/ensemble/judge/viral settings",
    )
    p_kit.add_argument("--anti-boring", action="store_true")
    p_kit.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply high-signal lexicon sharpening to autonomous post generation",
    )
    p_kit.add_argument(
        "--judge-threshold",
        type=float,
        default=82.0,
        help="Frog Judge minimum score (0-100) required for autonomous post acceptance",
    )
    p_kit.add_argument(
        "--max-attempts",
        type=int,
        default=7,
        help="Max autonomous regeneration attempts before returning best draft",
    )
    p_kit.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="Generate N variants per attempt and pick the top-scoring draft (max 8)",
    )
    p_kit.add_argument(
        "--viral-boost",
        action="store_true",
        help="Increase viral-potential weighting in Frog Judge",
    )

    p_seamless = sub.add_parser(
        "seamless",
        help="One-command reliable flow: diagnose/repair, then lean single reply run",
    )
    p_seamless.add_argument("--handle", default="OpenClawAI")
    p_seamless.add_argument("--since-id", default=None)
    p_seamless.add_argument("--min-confidence", type=int, default=70)
    p_seamless.add_argument("--cooldown-minutes", type=int, default=15)
    p_seamless.add_argument("--dry-run", action="store_true")
    p_seamless.add_argument("--skip-network", action="store_true")
    p_seamless.add_argument("--no-repair-auth", action="store_true")
    p_seamless.add_argument("--log-path", default="data/replies.jsonl")
    p_seamless.add_argument("--report-path", default="data/mentions_report.json")

    sub.add_parser(
        "restart-setup",
        help="Restart recovery: repair setup/auth after reboot without posting",
    )

    p_gather = sub.add_parser(
        "gather-data",
        help="Collect mentions + search snapshot for autonomous decision context",
    )
    p_gather.add_argument("--handle", default="OpenClawAI")
    p_gather.add_argument(
        "--query",
        default='openclaw OR "local ai agent" lang:en -is:retweet',
    )
    p_gather.add_argument("--limit", type=int, default=20)
    p_gather.add_argument("--max-pages", type=int, default=1)
    p_gather.add_argument("--output", default="data/engine_data_snapshot.json")
    p_gather.add_argument("--json", action="store_true")

    p_browse = sub.add_parser(
        "browse-twitter",
        help="Browse Twitter: mentions/search/user timeline, with pagination and incremental scan",
    )
    p_browse.add_argument(
        "--mode",
        choices=["search", "user"],
        default="search",
        help="search (mentions/custom query) or user timeline mode",
    )
    p_browse.add_argument("--tweet", help="Fetch one tweet by tweet ID")
    p_browse.add_argument("--query", help="Custom recent-search query")
    p_browse.add_argument("--handle", default="OpenClawAI", help="Handle for default mentions query")
    p_browse.add_argument("--username", help="Username for --mode user (defaults to --handle)")
    p_browse.add_argument("--limit", type=int, default=20, help="Number of results (5-100)")
    p_browse.add_argument("--max-pages", type=int, default=1, help="Pagination pages to fetch (1-10)")
    p_browse.add_argument("--since-id", help="Only return tweets newer than this tweet ID")
    p_browse.add_argument("--until-id", help="Only return tweets older than this tweet ID")
    p_browse.add_argument("--save", help="Optional file path to save output JSON")
    p_browse.add_argument(
        "--with-replies",
        action="store_true",
        help="In user mode, include replies in timeline results",
    )
    p_browse.add_argument("--json", action="store_true", help="Print raw JSON response")

    p_mentions = sub.add_parser(
        "mentions",
        help="Fetch mentions via native /2/users/:id/mentions endpoint",
    )
    p_mentions.add_argument("--limit", type=int, default=20, help="Results per page (5-100)")
    p_mentions.add_argument("--max-pages", type=int, default=1, help="Pagination pages to fetch (1-10)")
    p_mentions.add_argument("--since-id", help="Only include mentions newer than this tweet ID")
    p_mentions.add_argument("--save", help="Optional file path to save output JSON")
    p_mentions.add_argument("--preview", type=int, default=5, help="Rows to print in non-JSON mode")
    p_mentions.add_argument("--json", action="store_true", help="Print raw JSON output")

    p_search = sub.add_parser(
        "search",
        help="Proactive discovery across X recent search",
    )
    p_search.add_argument("--query", required=True, help="Recent search query")
    p_search.add_argument("--limit", type=int, default=20, help="Results per page (5-100)")
    p_search.add_argument("--max-pages", type=int, default=1, help="Pagination pages to fetch (1-10)")
    p_search.add_argument("--since-id", help="Only include tweets newer than this tweet ID")
    p_search.add_argument("--save", help="Optional file path to save output JSON")
    p_search.add_argument("--preview", type=int, default=5, help="Rows to print in non-JSON mode")
    p_search.add_argument("--json", action="store_true", help="Print raw JSON output")

    p_discover_run = sub.add_parser(
        "reply-discover-run",
        help="Proactive discovery + draft/auto-reply pipeline",
    )
    p_discover_run.add_argument("--watchlist", default="default", help="Watchlist name from ~/.config/twitter-engine/watchlists.json")
    p_discover_run.add_argument("--query", help="One-off query (overrides --watchlist)")
    p_discover_run.add_argument("--since-id", help="Override checkpoint and only include newer tweets than this ID")
    p_discover_run.add_argument("--max-tweets", type=int, default=1, help="Max tweets to fetch per query")
    p_discover_run.add_argument("--max-pages", type=int, default=1, help="Pagination pages per query")
    p_discover_run.add_argument("--min-score", type=int, default=20, help="Minimum engagement score")
    p_discover_run.add_argument("--min-confidence", type=int, default=75, help="Minimum confidence required for auto-post")
    p_discover_run.add_argument("--auto-post", action="store_true", help="Auto-post qualified replies")
    p_discover_run.add_argument(
        "--approval-queue",
        action="store_true",
        help="Queue qualified replies for manual approval instead of posting immediately",
    )
    p_discover_run.add_argument("--dry-run", action="store_true", help="Generate drafts only, never post")
    p_discover_run.add_argument("--output", default="data/discovery_latest.json", help="Where to save JSON output")
    p_discover_run.add_argument("--preview", type=int, default=5, help="Rows to print in non-JSON mode")
    p_discover_run.add_argument("--json", action="store_true", help="Print raw JSON output")

    p_reply_approve = sub.add_parser(
        "reply-approve",
        help="Review approval queue and post selected queued replies",
    )
    p_reply_approve.add_argument("--list", action="store_true", help="List queued replies")
    p_reply_approve.add_argument(
        "--approve",
        nargs="*",
        help="Queue IDs to approve/post (accepts with or without q_ prefix)",
    )
    p_reply_approve.add_argument("--dry-run", action="store_true", help="Preview approve actions without posting")
    p_reply_approve.add_argument("--json", action="store_true", help="Print queue list as JSON")

    p_inspire = sub.add_parser(
        "inspire-tweets",
        help="Browse Twitter and generate inspiration drafts from current conversation themes",
    )
    p_inspire.add_argument("--topic", default="OpenClaw", help="Topic label for drafted tweets")
    p_inspire.add_argument("--query", help="Custom search query (defaults from --topic)")
    p_inspire.add_argument("--limit", type=int, default=20, help="Results per page (5-100)")
    p_inspire.add_argument("--max-pages", type=int, default=2, help="Pages to fetch (1-10)")
    p_inspire.add_argument("--since-id", help="Only include tweets newer than this tweet ID")
    p_inspire.add_argument("--until-id", help="Only include tweets older than this tweet ID")
    p_inspire.add_argument("--sample-size", type=int, default=12, help="Sample tweets used to derive themes")
    p_inspire.add_argument("--draft-count", type=int, default=5, help="Number of inspiration drafts")
    p_inspire.add_argument(
        "--web-inspiration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also pull internet headlines/search trends for inspiration",
    )
    p_inspire.add_argument("--web-query", help="Optional internet query override")
    p_inspire.add_argument("--web-items", type=int, default=8, help="Max internet inspiration items")
    p_inspire.add_argument("--save", help="Optional file path to save JSON output")
    p_inspire.add_argument("--json", action="store_true", help="Print raw JSON output")

    p_reply_discover = sub.add_parser(
        "reply-discover",
        help="Reply engine: discover candidate conversations",
    )
    p_reply_discover.add_argument("--keywords", required=True, help="comma-separated keywords")
    p_reply_discover.add_argument("--limit", type=int, default=30)
    p_reply_discover.add_argument("--local-input", default=None)
    p_reply_discover.add_argument("--output", required=True)

    p_reply_rank = sub.add_parser(
        "reply-rank",
        help="Reply engine: score and rank candidates",
    )
    p_reply_rank.add_argument("--input", required=True)
    p_reply_rank.add_argument("--keywords", required=True, help="comma-separated keywords")
    p_reply_rank.add_argument("--include-weak", action="store_true")
    p_reply_rank.add_argument("--output", required=True)

    p_reply_ideas = sub.add_parser(
        "reply-ideas",
        help="Reply engine: generate markdown reply ideas",
    )
    p_reply_ideas.add_argument("--input", required=True)
    p_reply_ideas.add_argument("--top", type=int, default=20)
    p_reply_ideas.add_argument("--output", required=True)

    p_reply_run = sub.add_parser(
        "reply-run",
        help="Reply engine: end-to-end discover/rank/ideas",
    )
    p_reply_run.add_argument("--keywords", required=True, help="comma-separated keywords")
    p_reply_run.add_argument("--limit", type=int, default=30)
    p_reply_run.add_argument("--local-input", default=None)
    p_reply_run.add_argument("--include-weak", action="store_true")
    p_reply_run.add_argument("--output", required=True)

    p_reply_tw = sub.add_parser(
        "reply-twitter-helper",
        help="Reply engine: draft/post reply to one target tweet",
    )
    p_reply_tw.add_argument("--tweet", required=True, help="tweet URL or tweet ID")
    p_reply_tw.add_argument("--draft-count", type=int, default=5)
    p_reply_tw.add_argument("--pick", type=int, default=1)
    p_reply_tw.add_argument("--dry-run", action="store_true")
    p_reply_tw.add_argument("--log-path", default="data/replies.jsonl")

    p_reply_e2e = sub.add_parser(
        "reply-twitter-e2e",
        help="Reply engine: mentions workflow with optional posting",
    )
    p_reply_e2e.add_argument("--handle", default="OpenClawAI")
    p_reply_e2e.add_argument("--mention-limit", type=int, default=1)
    p_reply_e2e.add_argument("--since-id", default=None, help="Only include mentions newer than this tweet ID")
    p_reply_e2e.add_argument("--draft-count", type=int, default=1)
    p_reply_e2e.add_argument("--pick", type=int, default=1)
    p_reply_e2e.add_argument("--post", action="store_true", help="Post replies")
    p_reply_e2e.add_argument("--auto-post", action="store_true", help="Alias for --post")
    p_reply_e2e.add_argument("--max-posts", type=int, default=1)
    p_reply_e2e.add_argument(
        "--approval-queue",
        action="store_true",
        help="Queue qualified replies instead of posting immediately",
    )
    p_reply_e2e.add_argument("--min-confidence", type=int, default=70)
    p_reply_e2e.add_argument("--web-enrich", action="store_true")
    p_reply_e2e.add_argument("--web-context-items", type=int, default=2)
    p_reply_e2e.add_argument("--log-path", default="data/replies.jsonl")
    p_reply_e2e.add_argument("--report-path", default="data/mentions_report.json")

    p_reply_quick = sub.add_parser(
        "reply-quick",
        help="Reply engine: one-shot efficient flow (1 mention -> 1 draft -> post)",
    )
    p_reply_quick.add_argument("--handle", default="OpenClawAI")
    p_reply_quick.add_argument("--since-id", default=None, help="Only include mentions newer than this tweet ID")
    p_reply_quick.add_argument("--min-confidence", type=int, default=70)
    p_reply_quick.add_argument("--cooldown-minutes", type=int, default=15, help="Skip run if last reply was newer than this")
    p_reply_quick.add_argument("--dry-run", action="store_true", help="Generate one draft without posting")
    p_reply_quick.add_argument("--log-path", default="data/replies.jsonl")
    p_reply_quick.add_argument("--report-path", default="data/mentions_report.json")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    global ACTIVE_ACCOUNT
    parser = build_parser()
    args = parser.parse_args(argv)
    ACTIVE_ACCOUNT = args.account or "default"
    os.environ["OPENCLAW_TWITTER_ACCOUNT"] = ACTIVE_ACCOUNT
    os.environ["TWITTER_ENGINE_ACCOUNT"] = ACTIVE_ACCOUNT

    env_path = Path(args.env_file)

    try:
        if args.command == "setup":
            return cmd_setup(env_path, args)
        if args.command == "app-settings":
            return cmd_app_settings(env_path)
        if args.command == "walkthrough":
            return cmd_walkthrough()
        if args.command == "set-bearer-token":
            return cmd_set_bearer_token(env_path, args)
        if args.command == "memory":
            return cmd_memory(args)
        if args.command in {"engine-status", "openclaw-status"}:
            return cmd_engine_status(env_path)
        if args.command == "kit":
            return cmd_kit(env_path, args)
        if args.command in {"twitter-engine", "run-twitter-helper"}:
            return cmd_twitter_engine(env_path, args)
        if args.command == "seamless":
            return cmd_seamless(env_path, args)
        if args.command == "restart-setup":
            return cmd_restart_setup(env_path, args)
        if args.command == "gather-data":
            return cmd_gather_data(env_path, args)
        if args.command == "browse-twitter":
            return cmd_browse_twitter(env_path, args)
        if args.command == "mentions":
            return cmd_mentions(env_path, args)
        if args.command == "search":
            return cmd_search(env_path, args)
        if args.command == "reply-discover-run":
            return cmd_reply_discover_run(env_path, args)
        if args.command == "reply-approve":
            return cmd_reply_approve(env_path, args)
        if args.command == "inspire-tweets":
            return cmd_inspire_tweets(env_path, args)
        if args.command in {
            "reply-discover",
            "reply-rank",
            "reply-ideas",
            "reply-run",
            "reply-twitter-helper",
            "reply-twitter-e2e",
            "reply-quick",
        }:
            try:
                return cmd_reply_engine(args)
            except Exception as exc:
                raise TwitterHelperError(str(exc)) from exc
        if args.command == "auth-login":
            return cmd_auth_login(env_path, args)
        if args.command == "doctor":
            return cmd_doctor(env_path)
        if args.command == "auto-diagnose":
            return cmd_auto_diagnose(env_path, args)

        cfg, env_values = resolve_config(env_path)

        if args.command == "check-auth":
            return cmd_check_auth(cfg, env_path, env_values)
        if args.command == "post":
            return cmd_post(cfg, env_path, env_values, args)
        if args.command in {"engine-autopost", "openclaw-autopost"}:
            return cmd_engine_autopost(cfg, env_path, env_values, args)
        if args.command in {"engine-check", "openclaw"}:
            return cmd_engine_check(env_path, args)
        if args.command == "thread":
            return cmd_thread(cfg, env_path, env_values, args)

        raise TwitterHelperError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except TwitterHelperError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Try: `setup` then `auth-login` then `doctor`", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
