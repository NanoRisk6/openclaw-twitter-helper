import importlib.util
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


if __name__ == "__main__":
    unittest.main()
