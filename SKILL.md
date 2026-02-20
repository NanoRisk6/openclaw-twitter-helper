---
name: x-twitter-helper
description: Post, reply, browse mentions, and run reply workflows on X/Twitter via official API. Uses keyring-stored OAuth2 tokens, rate-limit aware retries, dry-run safety, and account namespacing.
version: 0.2.0
author: NanoRisk6
tags: [twitter, x, post, reply, mentions, engagement, social-media, api]
emoji: 🦞
requires:
  binaries:
    - python3
    - twitter-engine
    - reply-engine
  env: []
homepage: https://github.com/NanoRisk6/twitter-engine
---

# X/Twitter Helper For OpenClaw

Official API bridge for posting, replying, browsing, and reply workflows.

## One-Time Setup

```bash
git clone https://github.com/NanoRisk6/twitter-engine.git ~/.openclaw/workspace/skills/x-twitter-helper
cd ~/.openclaw/workspace/skills/x-twitter-helper
pip install -r requirements.txt
./twitter-engine --account default auth-login
./twitter-engine --account default diagnose
```

## Commands (via `system.run`)

| Action | Command |
|---|---|
| Post | `twitter-engine --account <name> post --text "..."` |
| Post with media | `twitter-engine --account <name> post --text "..." --media <path-or-url[,path-or-url...]> [--alt-text "a,b"] [--dry-run]` |
| Reply | `twitter-engine --account <name> post --text "..." --in-reply-to <tweet_id>` |
| Mentions | `twitter-engine --account <name> mentions --limit 20 [--since-id ID] [--json]` |
| Search | `twitter-engine --account <name> search --query "openclaw lang:en min_faves:5"` |
| Discover & reply | `twitter-engine --account <name> reply-discover-run --watchlist default [--auto-post] [--dry-run]` |
| Approval queue | `twitter-engine --account <name> reply-discover-run --approval-queue ...` then `twitter-engine --account <name> reply-approve --list|--approve q_xxx` |
| Browse mentions/search | `twitter-engine --account <name> browse-twitter --handle OpenClawAI --limit 20 --json` |
| Diagnose | `twitter-engine --account <name> diagnose --json` |
| Readiness check only | `twitter-engine --account <name> engine-check --json` |
| Mentions reply workflow (draft) | `twitter-engine --account <name> reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1` |

## Agent Examples

```bash
system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default post --text 'OpenClaw helper is live 🦞'"
```

```bash
system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default post --text 'Great point — appreciate the thread.' --in-reply-to 1892345678901234567"
```

```bash
system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default browse-twitter --handle OpenClawAI --limit 10 --json"
```

## Best Practices

- Use `engine-autopost --dry-run` before posting if confidence is low.
- Run `diagnose` before automated posting loops.
- Use `--account` for persona/account separation.
- Respect platform policy and avoid spammy duplicate posting.
