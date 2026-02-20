import importlib.util
import json
import os
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "reply_engine" / "twitter_helper.py"
spec = importlib.util.spec_from_file_location("reply_engine_twitter_helper", MODULE_PATH)
reply_helper = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(reply_helper)


class _User:
    def __init__(self, user_id: str, username: str):
        self.id = user_id
        self.username = username


class _Tweet:
    def __init__(self, author_id: str):
        self.author_id = author_id


class _Resp:
    def __init__(self, data, users):
        self.data = data
        self.includes = {"users": users}


class VerifyReplyVisibleTests(unittest.TestCase):
    def test_verify_reply_visible_success(self) -> None:
        class Client:
            def get_tweet(self, **kwargs):
                return _Resp(_Tweet("42"), [_User("42", "OpenClawAI")])

        out = reply_helper.verify_reply_visible(
            client=Client(),
            reply_id="999",
            expected_username="OpenClawAI",
            attempts=1,
            delay_seconds=0,
        )
        self.assertEqual(out["url"], "https://x.com/OpenClawAI/status/999")

    def test_verify_reply_visible_failure(self) -> None:
        class Client:
            def get_tweet(self, **kwargs):
                raise RuntimeError("not found")

        original_sleep = reply_helper.time.sleep
        reply_helper.time.sleep = lambda x: None
        try:
            with self.assertRaises(RuntimeError):
                reply_helper.verify_reply_visible(
                    client=Client(),
                    reply_id="999",
                    attempts=2,
                    delay_seconds=0,
                )
        finally:
            reply_helper.time.sleep = original_sleep

    def test_required_env_uses_keyring_access_token_fallback(self) -> None:
        class FakeKeyring:
            def get_password(self, service, username):
                if service == "openclaw-twitter-helper:default" and username == "oauth_tokens":
                    return '{"access_token":"keyring_access"}'
                return None

        original_keyring = reply_helper.keyring
        original_env = dict(os.environ)
        try:
            os.environ.pop("TWITTER_BEARER_TOKEN", None)
            os.environ.pop("TWITTER_OAUTH2_ACCESS_TOKEN", None)
            os.environ["OPENCLAW_TWITTER_ACCOUNT"] = "default"
            reply_helper.keyring = FakeKeyring()
            env = reply_helper._required_env()
        finally:
            reply_helper.keyring = original_keyring
            os.environ.clear()
            os.environ.update(original_env)

        self.assertEqual(env["TWITTER_BEARER_TOKEN"], "keyring_access")

    def test_fetch_mentions_native_from_users_mentions(self) -> None:
        class _MeData:
            id = "99"
            username = "OpenClawAI"

        class _Me:
            data = _MeData()

        class _T:
            def __init__(self, tid, aid, text):
                self.id = tid
                self.author_id = aid
                self.text = text
                self.created_at = None
                self.public_metrics = {}
                self.conversation_id = tid

        class _UserObj:
            def __init__(self, uid, username):
                self.id = uid
                self.username = username

        class _Resp2:
            def __init__(self):
                self.data = [_T("123", "42", "hello mention")]
                self.includes = {"users": [_UserObj("42", "alice")]}
                self.meta = {}

        class Client:
            def get_me(self, **kwargs):
                return _Me()

            def get_users_mentions(self, **kwargs):
                return _Resp2()

        rows = reply_helper.fetch_mentions_native(Client(), handle="OpenClawAI", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tweet_id"], "123")
        self.assertEqual(rows[0]["author"], "alice")

    def test_run_mentions_workflow_fallback_source(self) -> None:
        class Client:
            pass

        original_build_client = reply_helper.build_client
        original_fetch_native = reply_helper.fetch_mentions_native
        original_fetch_fallback = reply_helper.fetch_mentions_search_fallback
        original_get_auth_user = reply_helper.get_authenticated_username
        original_generate = reply_helper.generate_reply_drafts
        try:
            reply_helper.build_client = lambda require_write=False: Client()
            reply_helper.fetch_mentions_native = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
            reply_helper.fetch_mentions_search_fallback = lambda client, handle, limit: [
                {
                    "tweet_id": "t1",
                    "author": "alice",
                    "text": "hi",
                    "created_at": "",
                    "metrics": {},
                    "conversation_id": "t1",
                    "url": "https://x.com/alice/status/t1",
                }
            ]
            reply_helper.get_authenticated_username = lambda client: "OpenClawAI"
            reply_helper.generate_reply_drafts = lambda author, text, draft_count: ["reply one"]

            result = reply_helper.run_mentions_workflow(
                handle="OpenClawAI",
                mention_limit=5,
                draft_count=1,
                pick=1,
                post=False,
                log_path="data/test_replies.jsonl",
                report_path="data/test_mentions_report.json",
            )
        finally:
            reply_helper.build_client = original_build_client
            reply_helper.fetch_mentions_native = original_fetch_native
            reply_helper.fetch_mentions_search_fallback = original_fetch_fallback
            reply_helper.get_authenticated_username = original_get_auth_user
            reply_helper.generate_reply_drafts = original_generate

        self.assertEqual(result["source"], "search_fallback")
        self.assertEqual(result["fetched_mentions"], 1)

    def test_get_full_conversation_chain(self) -> None:
        class _Ref:
            def __init__(self, tid, rtype="replied_to"):
                self.id = tid
                self.type = rtype

        class _T:
            def __init__(self, tid, aid, text, refs=None):
                self.id = tid
                self.author_id = aid
                self.text = text
                self.created_at = None
                self.conversation_id = "conv1"
                self.referenced_tweets = refs or []

        class _User:
            def __init__(self, uid, username):
                self.id = uid
                self.username = username

        class _Resp:
            def __init__(self, data, users):
                self.data = data
                self.includes = {"users": users}

        class Client:
            def get_tweet(self, id=None, **kwargs):
                if str(id) == "2":
                    return _Resp(_T("2", "a2", "child", refs=[_Ref("1")]), [_User("a2", "bob")])
                return _Resp(_T("1", "a1", "root"), [_User("a1", "alice")])

        out = reply_helper.get_full_conversation(Client(), "2")
        self.assertEqual(out["main"]["tweet_id"], "2")
        self.assertEqual(len(out["parents"]), 1)
        self.assertEqual(out["parents"][0]["tweet_id"], "1")

    def test_last_mention_checkpoint_roundtrip(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            original_config = reply_helper.CONFIG_DIR
            try:
                reply_helper.CONFIG_DIR = Path(td)
                reply_helper.save_last_mention_id("12345", account="default")
                got = reply_helper.load_last_mention_id(account="default")
            finally:
                reply_helper.CONFIG_DIR = original_config
        self.assertEqual(got, "12345")

    def test_replied_log_roundtrip(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            original_config = reply_helper.CONFIG_DIR
            try:
                reply_helper.CONFIG_DIR = Path(td)
                reply_helper.record_replied("123", "555", "unit")
                self.assertTrue(reply_helper.has_replied_to("123"))
                self.assertFalse(reply_helper.has_replied_to("999"))
            finally:
                reply_helper.CONFIG_DIR = original_config

    def test_score_discovery_candidate(self) -> None:
        score = reply_helper.score_discovery_candidate(
            {"metrics": {"like_count": 10, "retweet_count": 2, "reply_count": 3, "quote_count": 1}}
        )
        self.assertEqual(score, 25)

    def test_run_discovery_workflow_skips_already_replied(self) -> None:
        class Client:
            pass

        original_build_client = reply_helper.build_client
        original_fetch_discovery = reply_helper.fetch_discovery_search
        original_generate = reply_helper.generate_reply_drafts
        original_has_replied = reply_helper.has_replied_to
        original_get_auth_user = reply_helper.get_authenticated_username
        original_post_reply = reply_helper.post_reply
        original_verify = reply_helper.verify_reply_visible
        try:
            reply_helper.build_client = lambda require_write=False: Client()
            reply_helper.fetch_discovery_search = lambda **kwargs: [
                {
                    "tweet_id": "123",
                    "author": "alice",
                    "text": "openclaw local ai",
                    "url": "https://x.com/alice/status/123",
                    "metrics": {"like_count": 10, "retweet_count": 0, "reply_count": 0, "quote_count": 0},
                    "score": 30,
                }
            ]
            reply_helper.generate_reply_drafts = lambda author, text, draft_count: ["reply one"]
            reply_helper.has_replied_to = lambda tweet_id, account=None: True
            reply_helper.get_authenticated_username = lambda client: "OpenClawAI"
            reply_helper.post_reply = lambda client, tweet_id, text: "r1"
            reply_helper.verify_reply_visible = lambda client, reply_id, expected_username=None, attempts=3, delay_seconds=1.0: {
                "url": "https://x.com/OpenClawAI/status/r1"
            }

            result = reply_helper.run_discovery_workflow(
                query="openclaw",
                limit=5,
                post=True,
                max_posts=1,
                log_path="data/test_replies.jsonl",
                report_path="data/test_discovery_report.json",
            )
        finally:
            reply_helper.build_client = original_build_client
            reply_helper.fetch_discovery_search = original_fetch_discovery
            reply_helper.generate_reply_drafts = original_generate
            reply_helper.has_replied_to = original_has_replied
            reply_helper.get_authenticated_username = original_get_auth_user
            reply_helper.post_reply = original_post_reply
            reply_helper.verify_reply_visible = original_verify

        self.assertEqual(result["posted_replies"], 0)
        self.assertEqual(result["results"][0]["status"], "skipped_already_replied")

    def test_generate_reply_many_ways_fallback_modes(self) -> None:
        modes = ["direct", "curious", "technical"]
        result = reply_helper.generate_reply_many_ways(
            author="alice",
            text="OpenClaw runs local models and avoids lock-in.",
            modes=modes,
        )
        self.assertEqual(set(result.keys()), set(modes))
        self.assertIn("@alice", result["direct"])

    def test_fetch_web_context_parses_abstract(self) -> None:
        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return None
            def read(self):
                return b'{"AbstractText":"Local AI summary.","RelatedTopics":[]}'

        original_urlopen = reply_helper.urlrequest.urlopen
        try:
            reply_helper.urlrequest.urlopen = lambda url, timeout=6.0: _Resp()
            rows = reply_helper.fetch_web_context("local ai agents", max_items=1)
        finally:
            reply_helper.urlrequest.urlopen = original_urlopen
        self.assertEqual(len(rows), 1)
        self.assertIn("Local AI summary", rows[0])


if __name__ == "__main__":
    unittest.main()
