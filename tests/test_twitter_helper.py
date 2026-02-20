import importlib.util
from pathlib import Path
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

        def fake_post_with_retry(cfg, env_path, env_values, text, reply_to_id=None):
            captured["reply_to_id"] = reply_to_id
            captured["text"] = text
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
            {"text": "Thanks!", "file": None, "in_reply_to": "2024820748980748765"},
        )()

        original_post_with_retry = twitter_helper.post_with_retry
        twitter_helper.post_with_retry = fake_post_with_retry
        try:
            rc = twitter_helper.cmd_post(cfg, Path("."), {}, args)
        finally:
            twitter_helper.post_with_retry = original_post_with_retry

        self.assertEqual(rc, 0)
        self.assertEqual(captured["reply_to_id"], "2024820748980748765")


if __name__ == "__main__":
    unittest.main()
