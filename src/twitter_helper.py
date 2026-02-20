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
import secrets
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from email.message import Message
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

API_BASE = "https://api.twitter.com/2"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
MEDIA_UPLOAD_URL = f"{API_BASE}/media/upload"
MEDIA_METADATA_URL = f"{API_BASE}/media/metadata"
MAX_TWEET_LEN = 280
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8080/callback"
DEFAULT_SCOPES = "tweet.read tweet.write users.read offline.access media.write"
OPENCLAW_SUFFIX_RE = re.compile(
    r"\s*\[openclaw-\d{8}-\d{6}-[a-z0-9]{4}\]\s*$", re.IGNORECASE
)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
TOOL_ROOT = Path(__file__).resolve().parents[1]
TOKEN_SERVICE_NAME = "openclaw-twitter-helper"
TOKEN_CONFIG_DIR = Path.home() / ".config" / "openclaw-twitter-helper"
TOKEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TOKENS_JSON_FALLBACK = TOKEN_CONFIG_DIR / "tokens.json"
ACTIVE_ACCOUNT = "default"
USER_ID_CACHE_TTL_SECONDS = 86400
APPROVAL_DIR = TOKEN_CONFIG_DIR / "approval_queue"
APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_SIZE_MB = 5
MAX_IMAGES = 5

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

REPLY_POST_KEYS = [
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_SECRET",
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
        try:
            raw = keyring.get_password(self.key_prefix, "oauth_tokens")
            if not raw:
                return None
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k: str(v) if v is not None else "" for k, v in parsed.items()}
            return None
        except Exception:
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


def http_json_with_headers(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, object]] = None,
    form_payload: Optional[Dict[str, str]] = None,
    max_retries: int = 5,
) -> Tuple[int, Dict[str, object], Dict[str, str]]:
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
                print(
                    f"Rate limit hit (429). Sleeping {int(sleep_seconds)}s before retry "
                    f"({attempt + 1}/{max_retries - 1})..."
                )
                time.sleep(sleep_seconds)
                continue
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
    token = get_env_value(env, "TWITTER_BEARER_TOKEN")
    if not token and env_path is not None:
        token = token_manager(env_path).get_access_token(env)
    if not token:
        token = get_env_value(env, "TWITTER_OAUTH2_ACCESS_TOKEN")
    if not token:
        raise TwitterHelperError(
            "Missing TWITTER_BEARER_TOKEN (or TWITTER_OAUTH2_ACCESS_TOKEN) for browse-twitter."
        )
    return token


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

    active_run_tag = run_tag or unique_marker("openclaw")
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
    return OPENCLAW_SUFFIX_RE.sub("", text).strip()


def unique_marker(prefix: str = "openclaw") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(2)
    return f"{prefix}-{stamp}-{token}"


def make_unique_public_tweet(base_text: str) -> str:
    base = sanitize_public_text(base_text)
    if not base:
        raise TwitterHelperError("No tweet text provided.")

    suffix = datetime.now(timezone.utc).strftime(" • %Y-%m-%d %H:%M:%SZ")
    allowed = MAX_TWEET_LEN - len(suffix)
    if allowed < 1:
        raise TwitterHelperError("Tweet too long to append uniqueness suffix.")
    if len(base) > allowed:
        if allowed <= 3:
            base = base[:allowed]
        else:
            base = base[: allowed - 3].rstrip() + "..."
    return base + suffix


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
        return 0

    if args.command == "reply-twitter-e2e":
        result = run_mentions_workflow(
            handle=args.handle,
            mention_limit=args.mention_limit,
            since_id=getattr(args, "since_id", None),
            draft_count=args.draft_count,
            pick=args.pick,
            post=args.post,
            max_posts=args.max_posts,
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
    for row in queue:
        qid = str(row.get("id", "")).strip()
        if qid not in wanted:
            continue
        text = str(row.get("text", "")).strip()
        reply_to = str(row.get("in_reply_to", "")).strip() or None
        if not text:
            print(f"[WARN] q_{qid} missing text, skipping")
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
        )
        if status < 200 or status >= 300:
            print(f"[FAIL] q_{qid} post failed: {json.dumps(body, ensure_ascii=False)}")
            continue
        data = body.get("data") if isinstance(body, dict) else None
        posted_id = str(data.get("id", "")) if isinstance(data, dict) else ""
        if posted_id:
            _, posted_url = verify_post_visible(fresh, posted_id)
            print(f"Posted q_{qid}: {posted_url}")
        path = Path(str(row.get("_path", "")))
        if path.exists():
            path.unlink()
        posted += 1

    print(f"Approved and posted: {posted}")
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
    per_query_results: List[Dict[str, object]] = []

    for query in queries:
        effective_since = args.since_id or load_query_since_id(query, ACTIVE_ACCOUNT)
        rows, users, _ = fetch_search_rows(
            bearer=bearer,
            query=query,
            limit=max(5, min(args.max_tweets, 100)),
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
            total_candidates += 1
            aid = str(row.get("author_id", ""))
            author = users.get(aid, "unknown")
            text = str(row.get("text", "")).strip()

            context_chain = fetch_conversation_chain(bearer=bearer, tweet_id=tid, max_depth=6)
            context_text = "\n".join(
                [str(x.get("text", "")).strip() for x in context_chain[:-1] if x.get("text")]
            ).strip()

            draft_input = text if not context_text else f"{text}\n\nThread context:\n{context_text}"
            drafts = generate_reply_drafts(author=author, text=draft_input, draft_count=3)
            draft = drafts[0] if drafts else ""
            confidence = max(40, min(95, 50 + int(score / 8)))

            item = {
                "tweet_id": tid,
                "author": author,
                "score": score,
                "confidence": confidence,
                "tweet_text": text,
                "thread_context": context_text,
                "draft_reply": draft,
                "action": "draft",
            }

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
                        "reason": "discover_run_threshold",
                        "source": "reply-discover-run",
                        "query": query,
                        "tweet_text": text,
                    }
                )
                item["action"] = "queued"
                item["queue_id"] = f"q_{qid}"
            elif post_enabled and cfg is not None and draft and confidence >= args.min_confidence:
                fresh, (status, body) = post_with_retry(
                    cfg,
                    env_path,
                    env_values,
                    draft,
                    reply_to_id=tid,
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
    print(f"Seen: {total_seen} | candidates: {total_candidates} | posted: {total_posted}")
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


def make_inspiration_drafts(topic: str, sample_texts: List[str], draft_count: int) -> List[str]:
    terms = [t for t, _ in top_terms(sample_texts, top_n=10)]
    anchors = terms[: min(3, len(terms))]
    anchor_text = ", ".join(anchors) if anchors else topic

    templates = [
        f"Seeing strong momentum around {anchor_text}. My takeaway on {topic}: consistency beats hype when you ship every week.",
        f"{topic} feels noisy right now. Useful lens: pick one measurable KPI, iterate daily, and publish the deltas.",
        f"Hot take on {topic}: distribution compounds when message stays stable and examples stay concrete.",
        f"Question for builders in {topic}: what changed your outcomes most in the last 30 days?",
        f"{topic} trend check: signal is rising, but execution quality still decides winners.",
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
    terms = top_terms(sample_texts, top_n=8)
    drafts = make_inspiration_drafts(args.topic or "this space", sample_texts, args.draft_count)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": query,
            "count": len(rows),
            "sample_size": len(sample),
            "top_terms": terms,
            "drafts": drafts,
            "sample": sample,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved inspiration output -> {out}")

    if args.json:
        print(
            json.dumps(
                {
                    "query": query,
                    "count": len(rows),
                    "top_terms": terms,
                    "drafts": drafts,
                    "sample": sample,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"Query: {query}")
    print(f"Fetched: {len(rows)} tweets")
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
        next_steps.append("run post --text \"hello from Open Claw\"")

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
    print("Open Claw Posting Walkthrough")
    print("1) Run `setup` and enter OAuth 2.0 Client ID + Client Secret.")
    print("2) Run `app-settings` and mirror those values in Twitter Developer Portal.")
    print("3) Run `auth-login` to open browser consent and generate OAuth2 tokens.")
    print("4) Paste callback URL back into the CLI.")
    print("5) `doctor` runs automatically to validate readiness.")
    print("6) Post with: `post --text \"hello from Open Claw\"`")
    return 0


def cmd_openclaw_status(env_path: Path) -> int:
    status = config_status(env_path)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def cmd_openclaw(env_path: Path, args: argparse.Namespace) -> int:
    status = config_status(env_path)

    if getattr(args, "json", False):
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    print("Open Claw Readiness Check")
    print(f"Env file: {status['env_file']}")
    print(f"Env exists: {status['env_exists']}")
    print(f"Has client id: {status['has_client_id']}")
    print(f"Has client secret: {status['has_client_secret']}")
    print(f"Has app-only bearer token: {status['has_bearer_token']}")
    print(f"Has OAuth2 access token: {status['has_oauth2_access_token']}")
    print(f"Has OAuth2 refresh token: {status['has_oauth2_refresh_token']}")

    doctor_rc = cmd_doctor(env_path)
    if doctor_rc != 0:
        print("Open Claw is not ready to post yet.")
        return doctor_rc

    print("Open Claw is ready to post.")
    print("Try: openclaw-autopost --text \"Open Claw status update\"")
    return 0


def cmd_run_twitter_helper(env_path: Path, args: argparse.Namespace) -> int:
    helper_path = TOOL_ROOT / "src" / "twitter_helper.py"
    wrapper_path = TOOL_ROOT / "run-twitter-helper"
    print("Run Twitter Helper")
    print(f"Workspace: {TOOL_ROOT}")
    print(f"Helper: {helper_path}")
    print(f"Wrapper: {wrapper_path}")
    print(f"Env file: {env_path.resolve()}")

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
                "Run `auth-login` once in an interactive terminal, then rerun `run-twitter-helper`."
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

    if args.no_post:
        print("Readiness is healthy. Skipping post (--no-post).")
        return 0

    cfg, env_values = resolve_config(env_path)
    base_text = args.text
    if args.file:
        base_text = Path(args.file).read_text(encoding="utf-8").strip()
    if not base_text:
        base_text = "Open Claw is online and posting."
    unique_text = make_unique_public_tweet(base_text)
    post_args = argparse.Namespace(text=unique_text, file=None, dry_run=args.dry_run)
    return cmd_openclaw_autopost(cfg, env_path, env_values, post_args)


def cmd_restart_setup(env_path: Path, args: argparse.Namespace) -> int:
    print("Twitter Helper Restart Recovery")
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
) -> Tuple[Config, Tuple[int, Dict[str, object]]]:
    run_tag = unique_marker("openclaw")
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


def cmd_openclaw_autopost(
    cfg: Config, env_path: Path, env_values: Dict[str, str], args: argparse.Namespace
) -> int:
    base_text = read_text_from_args(args)
    text = sanitize_public_text(base_text)
    validate_tweet_len(text)

    if args.dry_run:
        print(text)
        return 0

    print(f"Posting Open Claw auto-tweet ({len(text)}/{MAX_TWEET_LEN} chars)...")
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
    print(f"Open Claw auto-tweet posted and verified: id={tweet_id}")
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
        else "Open Claw auth complete and posting pipeline is live."
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
    return cmd_openclaw_autopost(cfg, env_path, env_values, auto_args)


def cmd_doctor(env_path: Path) -> int:
    print("Twitter Helper Doctor")
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
        print("You're ready. Try: post --text \"hello world\"")
        return 0
    except TwitterHelperError as exc:
        print(f"[FAIL] {exc}")
        return 1


def diagnose_openclaw(
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
        "posting": {"ready": False, "issues": []},
        "reply_scan": {"ready": False, "issues": []},
        "reply_post": {"ready": False, "issues": []},
        "actions": [],
    }

    posting_issues: List[str] = []
    reply_scan_issues: List[str] = []
    reply_post_issues: List[str] = []
    actions: List[str] = []

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
        if scan_status >= 400:
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

    missing_reply_keys = [k for k in REPLY_POST_KEYS if not get_env_value(env, k)]
    if missing_reply_keys:
        reply_post_issues.append(f"Missing OAuth1 reply-post keys: {', '.join(missing_reply_keys)}")
        actions.append("add OAuth1 keys/tokens to .env for reply posting")

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


def cmd_auto_diagnose(env_path: Path, args: argparse.Namespace) -> int:
    print("Open Claw Auto Diagnose")
    print(f"Env file: {env_path}")
    report = diagnose_openclaw(
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
                report = diagnose_openclaw(
                    env_path=env_path,
                    skip_network=args.skip_network,
                    reply_target_id=args.reply_target_id,
                )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
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
    text = read_text_from_args(args)
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
        print(f"Posted and verified {idx}/{len(tweets)}: id={tweet_id}")
        print(f"URL: {tweet_url}")

    print("Thread posted successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open Claw Twitter helper",
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
    sub.add_parser("openclaw-status", help="Print machine-readable readiness JSON")
    sub.add_parser("check-auth", help="Validate auth and print current account")

    p_post = sub.add_parser("post", help="Post a single tweet")
    p_post.add_argument("--text", help="Tweet text")
    p_post.add_argument("--file", help="Path to text file")
    p_post.add_argument(
        "--unique",
        action="store_true",
        help="Append UTC timestamp suffix to avoid duplicate-content errors",
    )
    p_post.add_argument(
        "--in-reply-to",
        help="Optional tweet ID to post as a reply",
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
        "openclaw-autopost",
        help="Open Claw integration post",
    )
    p_oc_auto.add_argument("--text", help="Base tweet text")
    p_oc_auto.add_argument("--file", help="Path to base text file")
    p_oc_auto.add_argument(
        "--dry-run",
        action="store_true",
        help="Print final unique tweet text instead of posting",
    )
    p_openclaw = sub.add_parser(
        "openclaw",
        help="Run Open Claw readiness check for posting",
    )
    p_openclaw.add_argument(
        "--json",
        action="store_true",
        help="Print readiness data as JSON",
    )

    p_thread = sub.add_parser("thread", help="Post a thread from file")
    p_thread.add_argument("--file", required=True, help="Path to thread text file")

    p_run = sub.add_parser(
        "run-twitter-helper",
        help="One-command Open Claw flow: check, repair auth if needed, then post unique tweet",
    )
    p_run.add_argument("--text", help="Base tweet text")
    p_run.add_argument("--file", help="Path to base text file")
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

    sub.add_parser(
        "restart-setup",
        help="Restart recovery: repair setup/auth after reboot without posting",
    )

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
    p_discover_run.add_argument("--watchlist", default="default", help="Watchlist name from ~/.config/openclaw-twitter-helper/watchlists.json")
    p_discover_run.add_argument("--query", help="One-off query (overrides --watchlist)")
    p_discover_run.add_argument("--since-id", help="Override checkpoint and only include newer tweets than this ID")
    p_discover_run.add_argument("--max-tweets", type=int, default=10, help="Max tweets to fetch per query")
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
    p_reply_e2e.add_argument("--mention-limit", type=int, default=20)
    p_reply_e2e.add_argument("--since-id", default=None, help="Only include mentions newer than this tweet ID")
    p_reply_e2e.add_argument("--draft-count", type=int, default=5)
    p_reply_e2e.add_argument("--pick", type=int, default=1)
    p_reply_e2e.add_argument("--post", action="store_true")
    p_reply_e2e.add_argument("--max-posts", type=int, default=3)
    p_reply_e2e.add_argument("--log-path", default="data/replies.jsonl")
    p_reply_e2e.add_argument("--report-path", default="data/mentions_report.json")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    global ACTIVE_ACCOUNT
    parser = build_parser()
    args = parser.parse_args(argv)
    ACTIVE_ACCOUNT = args.account or "default"
    os.environ["OPENCLAW_TWITTER_ACCOUNT"] = ACTIVE_ACCOUNT

    env_path = Path(args.env_file)

    try:
        if args.command == "setup":
            return cmd_setup(env_path, args)
        if args.command == "app-settings":
            return cmd_app_settings(env_path)
        if args.command == "walkthrough":
            return cmd_walkthrough()
        if args.command == "openclaw-status":
            return cmd_openclaw_status(env_path)
        if args.command == "run-twitter-helper":
            return cmd_run_twitter_helper(env_path, args)
        if args.command == "restart-setup":
            return cmd_restart_setup(env_path, args)
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
        if args.command == "openclaw-autopost":
            return cmd_openclaw_autopost(cfg, env_path, env_values, args)
        if args.command == "openclaw":
            return cmd_openclaw(env_path, args)
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
