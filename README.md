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

Secure token storage:

```bash
pip install keyring
```

Set helper dir once for easier commands:

```bash
export HELPER_DIR="$(pwd)"
```

## OpenClaw Native Skill Install

```bash
git clone https://github.com/NanoRisk6/openclaw-twitter-helper.git ~/.openclaw/workspace/skills/x-twitter-helper
cd ~/.openclaw/workspace/skills/x-twitter-helper
pip install -r requirements.txt
./run-twitter-helper --account default auth-login
./run-twitter-helper --account default diagnose
```

Skill file:

- `SKILL.md`

## Fast Start

```bash
$HELPER_DIR/run-twitter-helper restart
$HELPER_DIR/run-twitter-helper diagnose
$HELPER_DIR/run-twitter-helper openclaw-autopost --text "Open Claw is online"
```

Dedicated reply engine tool:

```bash
./reply-engine -h
./reply-engine twitter-discovery --query "openclaw OR \"local ai\" lang:en -is:retweet min_faves:5" --approval-queue
./reply-engine many-ways --tweet "https://x.com/OpenClawAI/status/2024820748980748765"
./reply-engine twitter-discovery --query "openclaw OR local ai" --web-enrich --web-context-items 3 --approval-queue
```

## Open Claw Docs

- Operations guide: `docs/OPENCLAW_OPERATIONS.md`
- Wizard flow: `docs/wizard-flow.md`
- Update log: `docs/OPENCLAW_UPDATE_LOG.md`


## Core Commands

Setup + OAuth:

```bash
python src/twitter_helper.py setup
python src/twitter_helper.py auth-login
python src/twitter_helper.py doctor
python src/twitter_helper.py auto-diagnose
```

Multi-account token namespace:

```bash
python src/twitter_helper.py --account default doctor
python src/twitter_helper.py --account brand2 auth-login
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

Post with media (comma-separated image paths/URLs, max 5):

```bash
python src/twitter_helper.py post --text "OpenClaw update with chart 🦞" --media ./examples/chart.png --alt-text "Performance chart"
python src/twitter_helper.py post --text "OpenClaw update with charts 🦞" --media ./examples/chart.png,https://picsum.photos/800/600 --alt-text "Chart one,Chart two"
python src/twitter_helper.py post --text "OpenClaw update with chart 🦞" --media https://picsum.photos/800/600 --dry-run
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

Claude/Open Claw auto-diagnose:

```bash
./run-twitter-helper diagnose
python src/twitter_helper.py auto-diagnose --json
```

Browse Twitter (default: recent mentions query for OpenClawAI):

```bash
python src/twitter_helper.py browse-twitter --handle OpenClawAI --limit 20
```

Native mentions endpoint:

```bash
python src/twitter_helper.py mentions --limit 20 --json
python src/twitter_helper.py mentions --since-id 2024835587052613989 --limit 20 --max-pages 2 --save data/mentions_latest.json
```

Browse with custom query:

```bash
python src/twitter_helper.py browse-twitter --query "openclaw OR xaetbgoad" --limit 20
```

Fetch one tweet by ID:

```bash
python src/twitter_helper.py browse-twitter --tweet 2024835587052613989 --json
```

Browse user timeline:

```bash
python src/twitter_helper.py browse-twitter --mode user --username OpenClawAI --limit 20 --max-pages 2
```

Incremental scan + save:

```bash
python src/twitter_helper.py browse-twitter --handle OpenClawAI --since-id 2024835587052613989 --limit 20 --max-pages 2 --save data/browse_latest.json
```

Generate tweet inspiration from live Twitter browsing:

```bash
python src/twitter_helper.py inspire-tweets --topic "OpenClaw" --max-pages 2 --draft-count 5 --save data/inspiration_latest.json
```

Proactive discovery search:

```bash
python src/twitter_helper.py search --query "openclaw OR \"local ai agent\" lang:en -is:retweet min_faves:5" --limit 20
python src/twitter_helper.py search --query "openclaw OR clawdbot lang:en" --since-id 2024835587052613989 --json
```

Proactive reply discovery run:

```bash
python src/twitter_helper.py reply-discover-run --watchlist default --max-tweets 15 --min-score 25 --dry-run
python src/twitter_helper.py reply-discover-run --query "openclaw lang:en min_faves:10" --auto-post --min-confidence 80
python src/twitter_helper.py reply-discover-run --watchlist default --approval-queue --max-tweets 8 --min-score 25
python src/twitter_helper.py reply-approve --list
python src/twitter_helper.py reply-approve --approve q_ab12cd34 q_ef56gh78
```

Unique applicable replies:
- `reply-discover-run` now filters for specificity + 24h phrasing uniqueness before queue/post.
- Recent reply prefixes are tracked in `~/.config/openclaw-twitter-helper/recent_replies.jsonl`.
- Optional persona file: `~/.config/openclaw-twitter-helper/persona/openclaw.md`.
- Double-reply protection: replied target IDs are tracked in `~/.config/openclaw-twitter-helper/replied_targets_<account>.json`.
- Persistent dedupe log: `~/.config/openclaw-twitter-helper/replied_to_<account>.jsonl` (90-day check window).
- To intentionally override manual reply guard: `post --in-reply-to <ID> --force-reply-target`.

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
python src/twitter_helper.py reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --since-id 2024835587052613989 --draft-count 5 --pick 1
```

For `reply-twitter-e2e` and other scan/read-heavy reply flows, set `TWITTER_BEARER_TOKEN` in `.env`.

Reply-engine native CLI (fully built workflows):

```bash
python -m src.reply_engine.cli twitter-e2e --handle OpenClawAI --mention-limit 20 --approval-queue --min-confidence 75
python -m src.reply_engine.cli twitter-discovery --query "openclaw OR \"local ai\" lang:en -is:retweet min_faves:5" --approval-queue --min-score 20 --min-confidence 75
python -m src.reply_engine.cli queue-list
python -m src.reply_engine.cli queue-approve --ids q_12345678 --dry-run
```

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
- Anti-hallucination guard: after post/reply create, the helper re-fetches the tweet by ID and only reports success after verification.
- OAuth2 access/refresh tokens are saved to OS keyring when available; `.env` is fallback-only.
- API calls retry on HTTP 429 using `x-rate-limit-reset` before failing.
