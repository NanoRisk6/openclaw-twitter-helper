#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import sys
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

API_BASE = "https://api.twitter.com/2"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
MAX_TWEET_LEN = 280
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8080/callback"
DEFAULT_SCOPES = "tweet.read tweet.write users.read offline.access"
OPENCLAW_SUFFIX_RE = re.compile(
    r"\s*\[openclaw-\d{8}-\d{6}-[a-z0-9]{4}\]\s*$", re.IGNORECASE
)
TOOL_ROOT = Path(__file__).resolve().parents[1]

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


def resolve_config(env_path: Path) -> Tuple[Config, Dict[str, str]]:
    env = load_env_file(env_path)

    def get(name: str) -> str:
        return os.getenv(name) or env.get(name, "")

    cfg = Config(
        client_id=get("TWITTER_CLIENT_ID"),
        client_secret=get("TWITTER_CLIENT_SECRET"),
        access_token=get("TWITTER_OAUTH2_ACCESS_TOKEN"),
        refresh_token=get("TWITTER_OAUTH2_REFRESH_TOKEN"),
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


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, object]] = None,
    form_payload: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, object]]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
    elif form_payload is not None:
        data = urllib.parse.urlencode(form_payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}

    req = urllib.request.Request(url=url, method=method, headers=headers, data=data)

    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body


def get_basic_auth_header(client_id: str, client_secret: str) -> str:
    creds = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(creds).decode("utf-8")


def get_env_value(env: Dict[str, str], key: str, default: str = "") -> str:
    return os.getenv(key) or env.get(key, default)


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
    env_values["TWITTER_OAUTH2_ACCESS_TOKEN"] = new_access
    env_values["TWITTER_OAUTH2_REFRESH_TOKEN"] = new_refresh
    write_env_file(env_path, env_values)

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


def fetch_tweet(cfg: Config, tweet_id: str) -> Tuple[int, Dict[str, object]]:
    return http_json(
        "GET",
        f"{API_BASE}/tweets/{tweet_id}?tweet.fields=author_id,created_at",
        {"Authorization": f"Bearer {cfg.access_token}"},
    )


def post_tweet(
    cfg: Config,
    text: str,
    reply_to_id: Optional[str] = None,
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


def config_status(env_path: Path) -> Dict[str, object]:
    env = load_env_file(env_path)
    exists = env_path.exists()

    has_client_id = bool(get_env_value(env, "TWITTER_CLIENT_ID"))
    has_client_secret = bool(get_env_value(env, "TWITTER_CLIENT_SECRET"))
    has_bearer_token = bool(get_env_value(env, "TWITTER_BEARER_TOKEN"))
    has_access_token = bool(get_env_value(env, "TWITTER_OAUTH2_ACCESS_TOKEN"))
    has_refresh_token = bool(get_env_value(env, "TWITTER_OAUTH2_REFRESH_TOKEN"))

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
) -> Tuple[Config, Tuple[int, Dict[str, object]]]:
    run_tag = unique_marker("openclaw")
    fresh = ensure_auth(cfg, env_path, env_values)
    status, body = post_tweet(fresh, text, reply_to_id=reply_to_id, run_tag=run_tag)
    if status in (401, 403):
        fresh = refresh_tokens(fresh, env_path, env_values)
        status, body = post_tweet(fresh, text, reply_to_id=reply_to_id, run_tag=run_tag)
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
    _, (status, body) = post_with_retry(cfg, env_path, env_values, text)
    if status < 200 or status >= 300:
        raise TwitterHelperError(
            f"Auto-post failed ({status}): {json.dumps(body, ensure_ascii=False)}"
        )
    data = body.get("data") if isinstance(body, dict) else None
    tweet_id = data.get("id") if isinstance(data, dict) else None
    print(f"Open Claw auto-tweet posted: id={tweet_id}")
    return 0


def cmd_auth_login(env_path: Path, args: argparse.Namespace) -> int:
    env = load_env_file(env_path)
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

    env["TWITTER_OAUTH2_ACCESS_TOKEN"] = access_token
    env["TWITTER_OAUTH2_REFRESH_TOKEN"] = refresh_token
    env["TWITTER_CLIENT_ID"] = client_id
    env["TWITTER_CLIENT_SECRET"] = client_secret
    env["TWITTER_REDIRECT_URI"] = redirect_uri
    env["TWITTER_WEBSITE_URL"] = get_env_value(
        env, "TWITTER_WEBSITE_URL", infer_website_url(redirect_uri)
    )
    env["TWITTER_SCOPES"] = scopes
    write_env_file(env_path, env)

    print("OAuth login complete. Tokens saved.")
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

    missing_tokens = [
        k
        for k in ["TWITTER_OAUTH2_ACCESS_TOKEN", "TWITTER_OAUTH2_REFRESH_TOKEN"]
        if not get_env_value(env, k)
    ]
    if missing_tokens:
        print(f"[FAIL] Missing OAuth tokens: {', '.join(missing_tokens)}")
        print("Run: auth-login (this opens the OAuth2 browser flow to generate tokens)")
        return 1

    print("[PASS] Config values are present.")
    if not get_env_value(env, "TWITTER_BEARER_TOKEN"):
        print(
            "[WARN] TWITTER_BEARER_TOKEN is missing. "
            "App-only read/scan reply workflows may not work."
        )

    try:
        cfg, env_values = resolve_config(env_path)
        cfg = ensure_auth(cfg, env_path, env_values)
        status, body = me(cfg)
        if status != 200:
            print(f"[FAIL] Auth check failed with status {status}")
            print(json.dumps(body, ensure_ascii=False))
            return 1

        data = body.get("data") if isinstance(body, dict) else None
        username = data.get("username") if isinstance(data, dict) else None
        user_id = data.get("id") if isinstance(data, dict) else None
        print(f"[PASS] API auth works as @{username} (id={user_id}).")
        print("You're ready. Try: post --text \"hello world\"")
        return 0
    except TwitterHelperError as exc:
        print(f"[FAIL] {exc}")
        return 1


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

    validate_tweet_len(text)
    print(f"Posting tweet ({len(text)}/{MAX_TWEET_LEN} chars)...")

    if args.in_reply_to:
        fresh = ensure_auth(cfg, env_path, env_values)
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

    _, (status, body) = post_with_retry(
        cfg,
        env_path,
        env_values,
        text,
        reply_to_id=args.in_reply_to,
    )

    if status < 200 or status >= 300:
        raise TwitterHelperError(
            f"Post failed ({status}): {json.dumps(body, ensure_ascii=False)}"
        )

    data = body.get("data") if isinstance(body, dict) else None
    tweet_id = data.get("id") if isinstance(data, dict) else None
    print(f"Tweet posted successfully: id={tweet_id}")
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
        print(f"Posted {idx}/{len(tweets)}: id={tweet_id}")

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
    sub.add_parser("app-settings", help="Print exact Twitter app settings to use")
    sub.add_parser("walkthrough", help="Print end-to-end setup/posting walkthrough")
    sub.add_parser("openclaw-status", help="Print machine-readable readiness JSON")
    sub.add_parser("check-auth", help="Validate auth and print current account")

    p_post = sub.add_parser("post", help="Post a single tweet")
    p_post.add_argument("--text", help="Tweet text")
    p_post.add_argument("--file", help="Path to text file")
    p_post.add_argument(
        "--in-reply-to",
        help="Optional tweet ID to post as a reply",
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
    p_reply_e2e.add_argument("--draft-count", type=int, default=5)
    p_reply_e2e.add_argument("--pick", type=int, default=1)
    p_reply_e2e.add_argument("--post", action="store_true")
    p_reply_e2e.add_argument("--max-posts", type=int, default=3)
    p_reply_e2e.add_argument("--log-path", default="data/replies.jsonl")
    p_reply_e2e.add_argument("--report-path", default="data/mentions_report.json")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
