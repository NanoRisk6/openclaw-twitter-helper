# Open Claw Twitter Helper

[![Tests](https://github.com/NanoRisk6/openclaw-twitter-helper/actions/workflows/test.yml/badge.svg)](https://github.com/NanoRisk6/openclaw-twitter-helper/actions/workflows/test.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

OAuth2 CLI for OpenClaw Twitter auto-posts/replies.

## Install

```bash
git clone https://github.com/NanoRisk6/openclaw-twitter-helper.git
cd openclaw-twitter-helper
python3 -m pip install --upgrade pip
pip install -e .
```

Optional reply-engine extras:

```bash
pip install -r requirements-reply-engine.txt
```

Set helper dir once for easier commands:

```bash
export HELPER_DIR="$(pwd)"
```

## Fast Start

```bash
$HELPER_DIR/run-twitter-helper restart
$HELPER_DIR/run-twitter-helper openclaw-autopost --text "Open Claw is online"
```

## Core Commands

Setup + OAuth:

```bash
python src/twitter_helper.py setup
python src/twitter_helper.py auth-login
python src/twitter_helper.py doctor
```

`setup` now also asks for `TWITTER_BEARER_TOKEN` (App-Only Authentication).  
This token is optional for basic posting, but recommended for reply scan/mentions workflows.

One-command mode (check/repair + post unique tweet):

```bash
python src/twitter_helper.py run-twitter-helper --text "Open Claw status update"
```

Restart recovery (no post):

```bash
python src/twitter_helper.py restart-setup
# wrapper aliases:
./run-twitter-helper restart
./run-twitter-helper recover
./run-twitter-helper fix
```

Post single tweet:

```bash
python src/twitter_helper.py post --text "hello from Open Claw"
```

Post reply tweet:

```bash
python src/twitter_helper.py post --text "Thanks for the feedback" --in-reply-to 2024820748980748765
```

Post from file:

```bash
python src/twitter_helper.py post --file examples/tweet.txt
```

Post thread (`---` separators):

```bash
python src/twitter_helper.py thread --file examples/thread.txt
```

Open Claw integration post:

```bash
python src/twitter_helper.py openclaw-autopost --text "Open Claw status update"
```

Dry-run:

```bash
python src/twitter_helper.py openclaw-autopost --text "Open Claw status update" --dry-run
```

Readiness check:

```bash
python src/twitter_helper.py openclaw
python src/twitter_helper.py openclaw --json
```

Browse Twitter (default: recent mentions query for OpenClawAI):

```bash
python src/twitter_helper.py browse-twitter --handle OpenClawAI --limit 20
```

Browse with custom query:

```bash
python src/twitter_helper.py browse-twitter --query "openclaw OR xaetbgoad" --limit 20
```

Fetch one tweet by ID:

```bash
python src/twitter_helper.py browse-twitter --tweet 2024835587052613989 --json
```

## Integrated Reply Engine

Discover:

```bash
python src/twitter_helper.py reply-discover --keywords "open source,ai agents" --limit 30 --output data/discovered.json
```

Rank:

```bash
python src/twitter_helper.py reply-rank --input data/discovered.json --keywords "open source,ai agents" --output data/ranked.json
```

Ideas markdown:

```bash
python src/twitter_helper.py reply-ideas --input data/ranked.json --top 20 --output data/reply_ideas.md
```

End-to-end:

```bash
python src/twitter_helper.py reply-run --keywords "open source,ai agents,twitter growth" --limit 30 --output data/reply_ideas_step.md
```

Single tweet helper:

```bash
python src/twitter_helper.py reply-twitter-helper --tweet "https://twitter.com/OpenClawAI/status/2024820748980748765" --dry-run
```

Mentions workflow:

```bash
python src/twitter_helper.py reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1
```

For `reply-twitter-e2e` and other scan/read-heavy reply flows, set `TWITTER_BEARER_TOKEN` in `.env`.

## Examples

- `examples/tweet.txt`
- `examples/thread.txt`
- `examples/reply.txt`
- `examples/thread_reply.txt`

Wizard flow diagram: `docs/wizard-flow.md`

## CI

GitHub Actions workflow: `.github/workflows/test.yml`

- runs `pytest`
- runs `doctor` smoke check

## Notes

- OAuth1 keys/tokens are not used for primary posting flow.
- OAuth2 callback URI must exactly match your app settings.
- Public tweet text is sanitized to remove trailing `[openclaw-YYYYMMDD-HHMMSS-xxxx]` suffixes.
