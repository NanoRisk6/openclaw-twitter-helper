import importlib.util
import io
import sys
import tempfile
import types
from pathlib import Path
import contextlib
from email.message import Message
import urllib.error
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "twitter_helper.py"
spec = importlib.util.spec_from_file_location("twitter_helper", MODULE_PATH)
twitter_helper = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(twitter_helper)


class PostSanitizeTests(unittest.TestCase):
    def test_post_tweet_strips_openclaw_suffix_before_payload(self) -> None:
        captured = {}

        def fake_http_json(method, url, headers, payload=None, form_payload=None):
            captured["method"] = method
            captured["url"] = url
            captured["payload"] = payload
            return 201, {"data": {"id": "123"}}

        cfg = twitter_helper.Config(
            client_id="cid",
            client_secret="secret",
            access_token="token",
            refresh_token="refresh",
        )

        original_http_json = twitter_helper.http_json
        twitter_helper.http_json = fake_http_json
        try:
            status, _ = twitter_helper.post_tweet(
                cfg,
                "Hello world [openclaw-20260220-025047-a4d2]   ",
                run_tag="openclaw-20260220-025047-a4d2",
            )
        finally:
            twitter_helper.http_json = original_http_json

        self.assertEqual(status, 201)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["payload"]["text"], "Hello world")

    def test_make_unique_public_tweet_adds_visible_timestamp_suffix(self) -> None:
        text = twitter_helper.make_unique_public_tweet("Open Claw status update")
        self.assertTrue(text.startswith("Open Claw status update"))
        self.assertRegex(text, r" • \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z$")
        self.assertNotRegex(text, r"\[openclaw-\d{8}-\d{6}-[a-z0-9]{4}\]$")

    def test_cmd_post_passes_reply_id(self) -> None:
        captured = {}

        def fake_post_with_retry(cfg, env_path, env_values, text, reply_to_id=None, media_ids=None):
            captured["reply_to_id"] = reply_to_id
            captured["text"] = text
            captured["media_ids"] = media_ids
            return cfg, (201, {"data": {"id": "tweet123"}})

        cfg = twitter_helper.Config(
            client_id="cid",
            client_secret="secret",
            access_token="token",
            refresh_token="refresh",
        )
        args = type(
            "Args",
            (),
            {
                "text": "Thanks!",
                "file": None,
                "in_reply_to": None,
                "unique": False,
                "dry_run": False,
                "media": None,
                "alt_text": None,
            },
        )()

        original_post_with_retry = twitter_helper.post_with_retry
        original_verify_post_visible = twitter_helper.verify_post_visible
        original_ensure_auth = twitter_helper.ensure_auth
        twitter_helper.post_with_retry = fake_post_with_retry
        twitter_helper.verify_post_visible = lambda cfg, tweet_id, attempts=3, delay_seconds=1.0: (
            "",
            f"https://x.com/i/web/status/{tweet_id}",
        )
        twitter_helper.ensure_auth = lambda cfg, env_path, env_values: cfg
        try:
            rc = twitter_helper.cmd_post(cfg, Path("."), {}, args)
        finally:
            twitter_helper.post_with_retry = original_post_with_retry
            twitter_helper.verify_post_visible = original_verify_post_visible
            twitter_helper.ensure_auth = original_ensure_auth

        self.assertEqual(rc, 0)
        self.assertIsNone(captured["reply_to_id"])
        self.assertIsNone(captured["media_ids"])

    def test_verify_post_visible_success(self) -> None:
        cfg = twitter_helper.Config(
            client_id="cid",
            client_secret="secret",
            access_token="token",
            refresh_token="refresh",
        )
        original_fetch = twitter_helper.fetch_tweet_with_author
        twitter_helper.fetch_tweet_with_author = lambda cfg, tweet_id: (
            200,
            {
                "data": {"id": tweet_id, "author_id": "42"},
                "includes": {"users": [{"id": "42", "username": "OpenClawAI"}]},
            },
        )
        try:
            username, url = twitter_helper.verify_post_visible(cfg, "123")
        finally:
            twitter_helper.fetch_tweet_with_author = original_fetch

        self.assertEqual(username, "OpenClawAI")
        self.assertEqual(url, "https://x.com/OpenClawAI/status/123")

    def test_verify_post_visible_failure(self) -> None:
        cfg = twitter_helper.Config(
            client_id="cid",
            client_secret="secret",
            access_token="token",
            refresh_token="refresh",
        )
        original_fetch = twitter_helper.fetch_tweet_with_author
        original_sleep = twitter_helper.time.sleep
        twitter_helper.fetch_tweet_with_author = lambda cfg, tweet_id: (
            404,
            {"title": "Not Found"},
        )
        twitter_helper.time.sleep = lambda x: None
        try:
            with self.assertRaises(twitter_helper.TwitterHelperError):
                twitter_helper.verify_post_visible(cfg, "123", attempts=2, delay_seconds=0)
        finally:
            twitter_helper.fetch_tweet_with_author = original_fetch
            twitter_helper.time.sleep = original_sleep

    def test_cmd_post_in_reply_to_preflight_not_visible(self) -> None:
        cfg = twitter_helper.Config(
            client_id="cid",
            client_secret="secret",
            access_token="token",
            refresh_token="refresh",
        )
        args = type(
            "Args",
            (),
            {
                "text": "Thanks!",
                "file": None,
                "in_reply_to": "2024832368729463259",
                "unique": False,
                "dry_run": False,
                "media": None,
                "alt_text": None,
            },
        )()

        original_ensure_auth = twitter_helper.ensure_auth
        original_fetch_tweet = twitter_helper.fetch_tweet
        twitter_helper.ensure_auth = lambda cfg, env_path, env_values: cfg
        twitter_helper.fetch_tweet = lambda cfg, tid: (
            200,
            {
                "errors": [
                    {
                        "detail": "Could not find tweet",
                        "resource_id": tid,
                    }
                ]
            },
        )
        try:
            with self.assertRaises(twitter_helper.TwitterHelperError):
                twitter_helper.cmd_post(cfg, Path("."), {}, args)
        finally:
            twitter_helper.ensure_auth = original_ensure_auth
            twitter_helper.fetch_tweet = original_fetch_tweet

    def test_diagnose_openclaw_reports_missing_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("", encoding="utf-8")

            report = twitter_helper.diagnose_openclaw(env_path, skip_network=True)

        self.assertFalse(report["overall_ready"])
        self.assertIn("run setup", report["actions"])
        posting_issues = report["posting"]["issues"]
        self.assertTrue(any("Missing app config" in issue for issue in posting_issues))

    def test_diagnose_openclaw_healthy_paths_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWITTER_CLIENT_ID=cid",
                        "TWITTER_CLIENT_SECRET=secret",
                        "TWITTER_BEARER_TOKEN=bearer",
                        "TWITTER_OAUTH2_ACCESS_TOKEN=access",
                        "TWITTER_OAUTH2_REFRESH_TOKEN=refresh",
                        "TWITTER_API_KEY=api",
                        "TWITTER_API_SECRET=api_secret",
                        "TWITTER_ACCESS_TOKEN=oauth1_access",
                        "TWITTER_ACCESS_SECRET=oauth1_secret",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            original_find_spec = twitter_helper.importlib.util.find_spec
            twitter_helper.importlib.util.find_spec = lambda name: object()
            try:
                report = twitter_helper.diagnose_openclaw(env_path, skip_network=True)
            finally:
                twitter_helper.importlib.util.find_spec = original_find_spec

        self.assertTrue(report["posting"]["ready"])
        self.assertTrue(report["reply_scan"]["ready"])
        self.assertTrue(report["reply_post"]["ready"])
        self.assertTrue(report["overall_ready"])

    def test_diagnose_openclaw_reply_target_skipped_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWITTER_CLIENT_ID=cid",
                        "TWITTER_CLIENT_SECRET=secret",
                        "TWITTER_BEARER_TOKEN=bearer",
                        "TWITTER_OAUTH2_ACCESS_TOKEN=access",
                        "TWITTER_OAUTH2_REFRESH_TOKEN=refresh",
                        "TWITTER_API_KEY=api",
                        "TWITTER_API_SECRET=api_secret",
                        "TWITTER_ACCESS_TOKEN=oauth1_access",
                        "TWITTER_ACCESS_SECRET=oauth1_secret",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            original_find_spec = twitter_helper.importlib.util.find_spec
            twitter_helper.importlib.util.find_spec = lambda name: object()
            try:
                report = twitter_helper.diagnose_openclaw(
                    env_path,
                    skip_network=True,
                    reply_target_id="2024820748980748765",
                )
            finally:
                twitter_helper.importlib.util.find_spec = original_find_spec

        issues = report["reply_post"]["issues"]
        self.assertTrue(any("--skip-network" in issue for issue in issues))

    def test_cmd_auto_diagnose_json_exit_code_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "json": True,
                    "skip_network": True,
                    "reply_target_id": None,
                    "no_repair_auth": True,
                },
            )()

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = twitter_helper.cmd_auto_diagnose(env_path, args)

        self.assertEqual(rc, 1)
        self.assertIn('"overall_ready": false', output.getvalue().lower())

    def test_cmd_auto_diagnose_json_exit_code_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWITTER_CLIENT_ID=cid",
                        "TWITTER_CLIENT_SECRET=secret",
                        "TWITTER_BEARER_TOKEN=bearer",
                        "TWITTER_OAUTH2_ACCESS_TOKEN=access",
                        "TWITTER_OAUTH2_REFRESH_TOKEN=refresh",
                        "TWITTER_API_KEY=api",
                        "TWITTER_API_SECRET=api_secret",
                        "TWITTER_ACCESS_TOKEN=oauth1_access",
                        "TWITTER_ACCESS_SECRET=oauth1_secret",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "json": True,
                    "skip_network": True,
                    "reply_target_id": None,
                    "no_repair_auth": True,
                },
            )()

            original_find_spec = twitter_helper.importlib.util.find_spec
            twitter_helper.importlib.util.find_spec = lambda name: object()
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    rc = twitter_helper.cmd_auto_diagnose(env_path, args)
            finally:
                twitter_helper.importlib.util.find_spec = original_find_spec

        self.assertEqual(rc, 0)
        self.assertIn('"overall_ready": true', output.getvalue().lower())

    def test_token_manager_migrates_and_scrubs_env_tokens(self) -> None:
        class FakeKeyring:
            def __init__(self):
                self.data = {}

            def set_password(self, service, username, password):
                self.data[(service, username)] = password

            def get_password(self, service, username):
                return self.data.get((service, username))

            def get_keyring(self):
                return self

        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWITTER_CLIENT_ID=cid",
                        "TWITTER_CLIENT_SECRET=secret",
                        "TWITTER_OAUTH2_ACCESS_TOKEN=access_token_value",
                        "TWITTER_OAUTH2_REFRESH_TOKEN=refresh_token_value",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            original_keyring = twitter_helper.keyring
            twitter_helper.keyring = FakeKeyring()
            token_from_tm = ""
            try:
                tm = twitter_helper.TokenManager(env_path=env_path, account="default")
                env = twitter_helper.load_env_file(env_path)
                migrated = tm.migrate_from_env(env)
                env_after = twitter_helper.load_env_file(env_path)
                token_from_tm = tm.get_access_token(env_after)
            finally:
                twitter_helper.keyring = original_keyring

        self.assertTrue(migrated)
        self.assertEqual(env_after.get("TWITTER_OAUTH2_ACCESS_TOKEN", ""), "")
        self.assertEqual(env_after.get("TWITTER_OAUTH2_REFRESH_TOKEN", ""), "")
        self.assertEqual(token_from_tm, "access_token_value")

    def test_http_json_with_headers_retries_429(self) -> None:
        class FakeResponse:
            def __init__(self, status, payload, headers):
                self.status = status
                self._payload = payload
                msg = Message()
                for k, v in headers.items():
                    msg[k] = str(v)
                self.headers = msg

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return self._payload.encode("utf-8")

        class DummyReq:
            full_url = "https://example.com"

        calls = {"n": 0, "slept": 0}

        def fake_urlopen(req, timeout=25):
            calls["n"] += 1
            if calls["n"] == 1:
                fp = io.BytesIO(b'{"title":"rate_limited"}')
                hdr = Message()
                hdr["x-rate-limit-reset"] = str(int(1))
                raise urllib.error.HTTPError(
                    url=req.full_url,
                    code=429,
                    msg="Too Many Requests",
                    hdrs=hdr,
                    fp=fp,
                )
            return FakeResponse(200, '{"ok": true}', {"x-rate-limit-remaining": "99"})

        original_urlopen = twitter_helper.urllib.request.urlopen
        original_sleep = twitter_helper.time.sleep
        twitter_helper.urllib.request.urlopen = fake_urlopen
        twitter_helper.time.sleep = lambda s: calls.__setitem__("slept", calls["slept"] + 1)
        try:
            status, body, headers = twitter_helper.http_json_with_headers(
                method="GET",
                url="https://example.com/test",
                headers={},
                max_retries=2,
            )
        finally:
            twitter_helper.urllib.request.urlopen = original_urlopen
            twitter_helper.time.sleep = original_sleep

        self.assertEqual(status, 200)
        self.assertEqual(body.get("ok"), True)
        self.assertEqual(headers.get("x-rate-limit-remaining"), "99")
        self.assertGreaterEqual(calls["slept"], 1)

    def test_cmd_mentions_json_uses_native_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("TWITTER_BEARER_TOKEN=bearer\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "limit": 20,
                    "max_pages": 1,
                    "since_id": None,
                    "save": None,
                    "preview": 5,
                    "json": True,
                },
            )()

            calls = {"urls": []}
            original_resolve_user_id = twitter_helper.resolve_current_user_id
            original_api_get_with_token = twitter_helper.api_get_with_token
            try:
                twitter_helper.resolve_current_user_id = lambda env_path, env: "u123"

                def fake_api_get_with_token(url, bearer):
                    calls["urls"].append(url)
                    return 200, {"data": [], "meta": {}}

                twitter_helper.api_get_with_token = fake_api_get_with_token
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    rc = twitter_helper.cmd_mentions(env_path, args)
            finally:
                twitter_helper.resolve_current_user_id = original_resolve_user_id
                twitter_helper.api_get_with_token = original_api_get_with_token

        self.assertEqual(rc, 0)
        self.assertTrue(any("/users/u123/mentions?" in u for u in calls["urls"]))

    def test_score_tweet_for_discovery(self) -> None:
        row = {
            "public_metrics": {
                "like_count": 10,
                "retweet_count": 2,
                "reply_count": 3,
                "quote_count": 1,
            }
        }
        self.assertEqual(twitter_helper.score_tweet_for_discovery(row), 25)

    def test_cmd_search_uses_recent_search(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("TWITTER_BEARER_TOKEN=bearer\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "query": "openclaw lang:en",
                    "limit": 20,
                    "max_pages": 1,
                    "since_id": None,
                    "save": None,
                    "preview": 5,
                    "json": True,
                },
            )()
            original_fetch = twitter_helper.fetch_search_rows
            twitter_helper.fetch_search_rows = lambda **kwargs: ([], {}, {})
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    rc = twitter_helper.cmd_search(env_path, args)
            finally:
                twitter_helper.fetch_search_rows = original_fetch
        self.assertEqual(rc, 0)
        self.assertIn('"query": "openclaw lang:en"', output.getvalue())

    def test_cmd_reply_discover_run_draft_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("TWITTER_BEARER_TOKEN=bearer\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "watchlist": "default",
                    "query": "openclaw lang:en",
                    "since_id": None,
                    "max_tweets": 10,
                    "max_pages": 1,
                    "min_score": 1,
                    "min_confidence": 75,
                    "auto_post": False,
                    "dry_run": True,
                    "output": None,
                    "preview": 5,
                    "json": True,
                },
            )()

            original_fetch_rows = twitter_helper.fetch_search_rows
            original_fetch_conv = twitter_helper.fetch_conversation_chain
            original_load_since = twitter_helper.load_query_since_id
            original_save_since = twitter_helper.save_query_since_id

            def fake_fetch_rows(**kwargs):
                return (
                    [
                        {
                            "id": "123",
                            "author_id": "42",
                            "text": "hello",
                            "public_metrics": {"like_count": 2, "retweet_count": 0, "reply_count": 0, "quote_count": 0},
                        }
                    ],
                    {"42": "alice"},
                    {},
                )

            fake_reply_mod = types.ModuleType("reply_engine.twitter_helper")
            fake_reply_mod.generate_reply_drafts = lambda author, text, draft_count: ["draft reply"]

            twitter_helper.fetch_search_rows = fake_fetch_rows
            twitter_helper.fetch_conversation_chain = lambda bearer, tweet_id, max_depth=6: []
            twitter_helper.load_query_since_id = lambda query, account: None
            twitter_helper.save_query_since_id = lambda query, account, tweet_id: None
            original_reply_mod = sys.modules.get("reply_engine.twitter_helper")
            sys.modules["reply_engine.twitter_helper"] = fake_reply_mod
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    rc = twitter_helper.cmd_reply_discover_run(env_path, args)
            finally:
                twitter_helper.fetch_search_rows = original_fetch_rows
                twitter_helper.fetch_conversation_chain = original_fetch_conv
                twitter_helper.load_query_since_id = original_load_since
                twitter_helper.save_query_since_id = original_save_since
                if original_reply_mod is None:
                    sys.modules.pop("reply_engine.twitter_helper", None)
                else:
                    sys.modules["reply_engine.twitter_helper"] = original_reply_mod

        self.assertEqual(rc, 0)
        self.assertIn('"total_candidates": 1', output.getvalue())

    def test_cmd_post_with_media_attaches_media_id(self) -> None:
        captured = {}
        cfg = twitter_helper.Config(
            client_id="cid",
            client_secret="secret",
            access_token="token",
            refresh_token="refresh",
        )
        args = type(
            "Args",
            (),
            {
                "text": "With image",
                "file": None,
                "in_reply_to": None,
                "unique": False,
                "dry_run": False,
                "media": "https://example.com/image.png",
                "alt_text": "chart image",
            },
        )()

        def fake_post_with_retry(cfg, env_path, env_values, text, reply_to_id=None, media_ids=None):
            captured["media_ids"] = media_ids
            return cfg, (201, {"data": {"id": "tweet999"}})

        original_ensure_auth = twitter_helper.ensure_auth
        original_upload_media = twitter_helper.upload_media
        original_post_with_retry = twitter_helper.post_with_retry
        original_verify_post_visible = twitter_helper.verify_post_visible
        twitter_helper.ensure_auth = lambda cfg, env_path, env_values: cfg
        twitter_helper.upload_media = lambda access_token, media_input, alt_text=None: "media123"
        twitter_helper.post_with_retry = fake_post_with_retry
        twitter_helper.verify_post_visible = lambda cfg, tweet_id, attempts=3, delay_seconds=1.0: (
            "",
            f"https://x.com/i/web/status/{tweet_id}",
        )
        try:
            rc = twitter_helper.cmd_post(cfg, Path("."), {}, args)
        finally:
            twitter_helper.ensure_auth = original_ensure_auth
            twitter_helper.upload_media = original_upload_media
            twitter_helper.post_with_retry = original_post_with_retry
            twitter_helper.verify_post_visible = original_verify_post_visible

        self.assertEqual(rc, 0)
        self.assertEqual(captured["media_ids"], ["media123"])


if __name__ == "__main__":
    unittest.main()
