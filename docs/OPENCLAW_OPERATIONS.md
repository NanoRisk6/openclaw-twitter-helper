# Open Claw Operations Guide

This guide documents what Open Claw can do with this helper, how to run each capability, and what credentials are required.

## Full Command Index

All commands exposed by `python src/twitter_helper.py`:

- `setup`: interactive wizard for app credentials and base config
- `auth-login`: OAuth2 browser flow to obtain/refresh access tokens
- `doctor`: guided diagnostics for config/auth/posting readiness
- `auto-diagnose`: combined posting + reply diagnosis with optional OAuth2 self-heal
- `app-settings`: prints exact Twitter Developer Portal settings to copy
- `walkthrough`: prints full setup/post flow in one place
- `openclaw-status`: machine-readable status output for automation
- `check-auth`: validates current auth and prints current account identity
- `post`: posts one tweet, file-based tweet, or reply tweet
- `openclaw-autopost`: Open Claw-oriented single post command
- `openclaw`: readiness check command for Open Claw orchestration
- `thread`: posts multi-part thread from file
- `run-twitter-helper`: one-command repair + unique post flow
- `restart-setup`: reboot-safe setup/auth repair flow without posting
- `browse-twitter`: read/search/user timeline/tweet fetch command
- `mentions`: native mentions endpoint fetch via `/2/users/:id/mentions`
- `inspire-tweets`: browse + generate concise draft tweet ideas
- `reply-discover`: discover candidate tweets for replies
- `reply-rank`: rank discovered candidates by relevance/quality
- `reply-ideas`: turn ranked candidates into drafted replies
- `reply-run`: end-to-end discover/rank/ideas pipeline
- `reply-twitter-helper`: draft/post to one specific tweet target
- `reply-twitter-e2e`: mentions workflow with optional posting cap

## Capability Map

| Capability | Command | Credentials Required | Output |
|---|---|---|---|
| Setup app + env | `python src/twitter_helper.py setup` | Client ID/Secret | `.env` updated |
| OAuth2 login | `python src/twitter_helper.py auth-login` | Client ID/Secret + browser | OAuth2 access/refresh tokens saved |
| Health check | `python src/twitter_helper.py doctor` | OAuth2 tokens | PASS/FAIL diagnostics |
| Auto diagnose (posting + reply) | `python src/twitter_helper.py auto-diagnose` | env + optional network | combined PASS/FAIL + fix steps |
| Print portal settings | `python src/twitter_helper.py app-settings` | Client ID/Secret recommended | exact values to copy into Twitter app |
| Print walkthrough | `python src/twitter_helper.py walkthrough` | none | setup/post instructions |
| Readiness status JSON | `python src/twitter_helper.py openclaw --json` | none (reads `.env`) | machine-readable readiness |
| Automation status JSON | `python src/twitter_helper.py openclaw-status` | none (reads `.env`) | compact JSON status for orchestration |
| Validate auth/account | `python src/twitter_helper.py check-auth` | OAuth2 tokens | account identity + validation result |
| One-command post flow | `python src/twitter_helper.py run-twitter-helper --text "..."` | OAuth2 tokens | one unique tweet posted |
| Restart recovery (no post) | `python src/twitter_helper.py restart-setup` | OAuth2/browser if needed | repaired auth/setup |
| Post single tweet | `python src/twitter_helper.py post --text "..."` | OAuth2 tokens | tweet posted |
| Reply to tweet | `python src/twitter_helper.py post --text "..." --in-reply-to <ID>` | OAuth2 tokens + visible target | reply posted |
| Thread post | `python src/twitter_helper.py thread --file examples/thread.txt` | OAuth2 tokens | thread posted |
| Open Claw autopost | `python src/twitter_helper.py openclaw-autopost --text "..."` | OAuth2 tokens | tweet posted |
| Browse Twitter search/user/tweet | `python src/twitter_helper.py browse-twitter ...` | Bearer token (or OAuth2 fallback) | console/JSON browse output |
| Native mentions endpoint | `python src/twitter_helper.py mentions ...` | Bearer token (or OAuth2 fallback) | mentions JSON + console summary |
| Generate inspiration drafts | `python src/twitter_helper.py inspire-tweets ...` | Bearer token (or OAuth2 fallback) | theme summary + drafts |
| Reply discovery | `python src/twitter_helper.py reply-discover ...` | internet only | `data/discovered.json` |
| Reply ranking | `python src/twitter_helper.py reply-rank ...` | none | `data/ranked.json` |
| Reply ideas markdown | `python src/twitter_helper.py reply-ideas ...` | optional OpenAI key | `data/reply_ideas.md` |
| Reply single target (draft/post) | `python src/twitter_helper.py reply-twitter-helper ...` | Bearer + OAuth1 keys for post | drafts + optional posted reply |
| Mentions workflow (draft/post) | `python src/twitter_helper.py reply-twitter-e2e ...` | Bearer + OAuth1 keys for post | `data/mentions_report.json` + optional posts |

## Required Environment Variables

Core posting flow:

- `TWITTER_CLIENT_ID`
- `TWITTER_CLIENT_SECRET`
- `TWITTER_REDIRECT_URI`
- `TWITTER_WEBSITE_URL`
- `TWITTER_SCOPES`

OAuth2 access/refresh tokens are stored in OS keyring when available (account namespace via `--account`).  
`.env` token fields are fallback-only when keyring is unavailable.

Recommended for browsing/reply scanning:

- `TWITTER_BEARER_TOKEN`

Required for reply-engine post via tweepy:

- `TWITTER_API_KEY`
- `TWITTER_API_SECRET`
- `TWITTER_ACCESS_TOKEN`
- `TWITTER_ACCESS_SECRET`

Optional for LLM draft generation:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

## Open Claw Wrapper Usage

From repo root:

- `./run-twitter-helper restart`
- `./run-twitter-helper recover`
- `./run-twitter-helper fix`
- `./run-twitter-helper reboot`
- `./run-twitter-helper openclaw`
- `./run-twitter-helper diagnose`
- `./run-twitter-helper openclaw-autopost --text "Open Claw update"`
- `./run-twitter-helper browse-twitter --mode search --query "openclaw" --limit 20`
- `./run-twitter-helper inspire-tweets --topic "OpenClaw" --max-pages 2 --draft-count 5 --save data/inspiration_latest.json`
- `./run-twitter-helper --account default diagnose`

Wrapper passes through all subcommands and has aliases:

- `restart`, `recover`, `fix`, `reboot` -> `restart-setup`

## Open Claw Intent Mapping

Use these direct intents from Open Claw:

- "Run Twitter Helper setup" -> `./run-twitter-helper restart`
- "Check if Twitter posting is healthy" -> `./run-twitter-helper openclaw`
- "Auto-diagnose posting/replying issues" -> `./run-twitter-helper diagnose`
- "Post this tweet" -> `./run-twitter-helper openclaw-autopost --text "<text>"`
- "Post a unique tweet" -> `python src/twitter_helper.py post --text "<text>" --unique`
- "Reply to this tweet ID" -> `python src/twitter_helper.py post --text "<text>" --in-reply-to <ID>`
- "Browse mentions" -> `python src/twitter_helper.py browse-twitter --handle OpenClawAI --limit 20`
- "Fetch mentions natively" -> `python src/twitter_helper.py mentions --limit 20 --json`
- "Find inspiration from recent posts" -> `python src/twitter_helper.py inspire-tweets --topic "OpenClaw" --draft-count 5`
- "Run mentions reply workflow draft-only" -> `python src/twitter_helper.py reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1`
- "Run mentions reply workflow and post one" -> `python src/twitter_helper.py reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1 --post --max-posts 1`

## OpenClaw `system.run` Examples

Use absolute skill path for reliable execution:

- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./run-twitter-helper --account default diagnose --json"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./run-twitter-helper --account default post --text 'OpenClaw helper is live 🦞'"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./run-twitter-helper --account default post --text 'Great point, thanks for sharing.' --in-reply-to 1892345678901234567"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./run-twitter-helper --account default browse-twitter --handle OpenClawAI --limit 10 --json"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./run-twitter-helper --account default mentions --limit 20 --json"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./run-twitter-helper --account default reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1"`

## Recommended Workflows

### 1. First-time setup

1. `python src/twitter_helper.py setup`
2. `python src/twitter_helper.py app-settings`
3. `python src/twitter_helper.py auth-login`
4. `python src/twitter_helper.py doctor`

### 2. After reboot

1. `./run-twitter-helper restart`
2. `./run-twitter-helper openclaw`

### 3. Inspiration + post loop

1. `./run-twitter-helper inspire-tweets --topic "OpenClaw" --max-pages 2 --draft-count 5 --save data/inspiration_latest.json`
2. Select one draft
3. `./run-twitter-helper openclaw-autopost --text "<draft>"`

### 4. Mentions reply loop (safe rollout)

Draft-only:

- `./run-twitter-helper reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1`

Post-enabled (capped):

- `./run-twitter-helper reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1 --post --max-posts 1`

## Output Locations

- Config template: `.env.example`
- Active config: `.env`
- Browse output: `data/browse_latest.json` (if `--save` used)
- Inspiration output: `data/inspiration_latest.json` (if `--save` used)
- Reply discovery: `data/discovered.json`
- Reply ranking: `data/ranked.json`
- Reply ideas: `data/reply_ideas.md`
- Mentions report: `data/mentions_report.json`
- Reply log: `data/replies.jsonl`

## Troubleshooting

401 unauthorized on browse/inspire:

- Verify `TWITTER_BEARER_TOKEN` in `.env`
- Ensure token has v2 read access
- Re-run `auth-login` and `doctor`

429 rate limit:

- The helper auto-retries using `x-rate-limit-reset`
- If retries are exhausted, wait for reset window and rerun

403 reply not visible/deleted:

- The target tweet ID is not visible to this account
- Use a visible tweet ID and retry

403 duplicate content:

- Use `post --unique` to append UTC suffix

Posted link is missing / page does not exist:

- The helper now verifies visibility by tweet ID before reporting success
- If verification fails, treat it as not posted and retry with `post --unique` or a new reply target

Non-interactive auth repair failure:

- Run `auth-login` once in interactive terminal

Missing tweepy for reply engine:

- `pip install -r requirements-reply-engine.txt`
