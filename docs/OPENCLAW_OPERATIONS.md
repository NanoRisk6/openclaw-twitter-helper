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
- `engine-status`: machine-readable status output for automation
- `check-auth`: validates current auth and prints current account identity
- `post`: posts one tweet, file-based tweet, or reply tweet
- `engine-autopost`: Open Claw-oriented single post command
- `engine-check`: readiness check command for Open Claw orchestration
- `thread`: posts multi-part thread from file
- `twitter-engine`: one-command repair + unique post flow
- `restart-setup`: reboot-safe setup/auth repair flow without posting
- `browse-twitter`: read/search/user timeline/tweet fetch command
- `mentions`: native mentions endpoint fetch via `/2/users/:id/mentions`
- `search`: proactive recent search across X using full query operators
- `inspire-tweets`: browse + generate concise draft tweet ideas
- `reply-discover`: discover candidate tweets for replies
- `reply-rank`: rank discovered candidates by relevance/quality
- `reply-ideas`: turn ranked candidates into drafted replies
- `reply-run`: end-to-end discover/rank/ideas pipeline
- `reply-discover-run`: proactive discover + draft/auto-reply pipeline
- `reply-approve`: review and approve queued replies for posting
- `reply-twitter-helper`: draft/post to one specific tweet target
- `reply-twitter-e2e`: mentions workflow with optional posting cap
- `reply-engine`: dedicated tool wrapper for `src.reply_engine.cli`

## Capability Map

| Capability | Command | Credentials Required | Output |
|---|---|---|---|
| Setup app + env | `python src/twitter_helper.py setup` | Client ID/Secret | `.env` updated |
| OAuth2 login | `python src/twitter_helper.py auth-login` | Client ID/Secret + browser | OAuth2 access/refresh tokens saved |
| Health check | `python src/twitter_helper.py doctor` | OAuth2 tokens | PASS/FAIL diagnostics |
| Auto diagnose (posting + reply) | `python src/twitter_helper.py auto-diagnose` | env + optional network | combined PASS/FAIL + fix steps |
| Print portal settings | `python src/twitter_helper.py app-settings` | Client ID/Secret recommended | exact values to copy into Twitter app |
| Print walkthrough | `python src/twitter_helper.py walkthrough` | none | setup/post instructions |
| Readiness status JSON | `python src/twitter_helper.py engine-check --json` | none (reads `.env`) | machine-readable readiness |
| Automation status JSON | `python src/twitter_helper.py engine-status` | none (reads `.env`) | compact JSON status for orchestration |
| Validate auth/account | `python src/twitter_helper.py check-auth` | OAuth2 tokens | account identity + validation result |
| One-command post flow | `python src/twitter_helper.py twitter-engine --text "..."` | OAuth2 tokens | one unique tweet posted |
| Restart recovery (no post) | `python src/twitter_helper.py restart-setup` | OAuth2/browser if needed | repaired auth/setup |
| Post single tweet | `python src/twitter_helper.py post --text "..."` | OAuth2 tokens | tweet posted |
| Post with media | `python src/twitter_helper.py post --text "..." --media <path-or-url[,path-or-url...]>` | OAuth2 tokens + `media.write` scope | tweet + 1-5 images posted |
| Reply to tweet | `python src/twitter_helper.py post --text "..." --in-reply-to <ID>` | OAuth2 tokens + visible target | reply posted |
| Thread post | `python src/twitter_helper.py thread --file examples/thread.txt` | OAuth2 tokens | thread posted |
| Open Claw autopost | `python src/twitter_helper.py engine-autopost --text "..."` | OAuth2 tokens | tweet posted |
| Browse Twitter search/user/tweet | `python src/twitter_helper.py browse-twitter ...` | Bearer token (or OAuth2 fallback) | console/JSON browse output |
| Native mentions endpoint | `python src/twitter_helper.py mentions ...` | Bearer token (or OAuth2 fallback) | mentions JSON + console summary |
| Proactive search | `python src/twitter_helper.py search --query "..." ...` | Bearer token (or OAuth2 fallback) | ranked discovery feed |
| Generate inspiration drafts | `python src/twitter_helper.py inspire-tweets ...` | Bearer token (or OAuth2 fallback) | theme summary + drafts |
| Reply discovery | `python src/twitter_helper.py reply-discover ...` | internet only | `data/discovered.json` |
| Reply ranking | `python src/twitter_helper.py reply-rank ...` | none | `data/ranked.json` |
| Reply ideas markdown | `python src/twitter_helper.py reply-ideas ...` | optional OpenAI key | `data/reply_ideas.md` |
| Reply single target (draft/post) | `python src/twitter_helper.py reply-twitter-helper ...` | Bearer for read + OAuth2 helper tokens for post | drafts + optional posted reply |
| Mentions workflow (draft/post) | `python src/twitter_helper.py reply-twitter-e2e ...` | Bearer for read + OAuth2 helper tokens for post | `data/mentions_report.json` + optional posts |
| Proactive discover+reply | `python src/twitter_helper.py reply-discover-run ...` | Bearer, optional OAuth2 post token | drafts or auto-replies + JSON report |
| Approval queue review/post | `python src/twitter_helper.py reply-approve --list` / `--approve q_xxx` | OAuth2 tokens for final post | queued replies listed or posted |

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

Optional for LLM draft generation:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

## Open Claw Wrapper Usage

From repo root:

- `./twitter-engine restart`
- `./twitter-engine recover`
- `./twitter-engine fix`
- `./twitter-engine reboot`
- `./twitter-engine engine-check`
- `./twitter-engine diagnose`
- `./twitter-engine engine-autopost --text "Open Claw update"`
- `./twitter-engine browse-twitter --mode search --query "openclaw" --limit 20`
- `./twitter-engine inspire-tweets --topic "OpenClaw" --max-pages 2 --draft-count 5 --save data/inspiration_latest.json`
- `./twitter-engine --account default diagnose`
- `./twitter-engine --account default reply-discover-run --watchlist default --approval-queue --max-tweets 8`
- `./twitter-engine --account default reply-approve --list`
- `./reply-engine twitter-discovery --query "openclaw OR local ai lang:en -is:retweet min_faves:5" --approval-queue`

Wrapper passes through all subcommands and has aliases:

- `restart`, `recover`, `fix`, `reboot` -> `restart-setup`

## Open Claw Intent Mapping

Use these direct intents from Open Claw:

- "Run Twitter Helper setup" -> `./twitter-engine restart`
- "Check if Twitter posting is healthy" -> `./twitter-engine engine-check`
- "Auto-diagnose posting/replying issues" -> `./twitter-engine diagnose`
- "Post this tweet" -> `./twitter-engine engine-autopost --text "<text>"`
- "Post a unique tweet" -> `python src/twitter_helper.py post --text "<text>" --unique`
- "Reply to this tweet ID" -> `python src/twitter_helper.py post --text "<text>" --in-reply-to <ID>`
- "Browse mentions" -> `python src/twitter_helper.py browse-twitter --handle OpenClawAI --limit 20`
- "Fetch mentions natively" -> `python src/twitter_helper.py mentions --limit 20 --json`
- "Find inspiration from recent posts" -> `python src/twitter_helper.py inspire-tweets --topic "OpenClaw" --draft-count 5`
- "Run mentions reply workflow draft-only" -> `python src/twitter_helper.py reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1`
- "Run mentions reply workflow and post one" -> `python src/twitter_helper.py reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1 --post --max-posts 1`
- "Queue proactive replies for manual review" -> `python src/twitter_helper.py reply-discover-run --watchlist default --approval-queue --max-tweets 8`
- "Approve queued replies" -> `python src/twitter_helper.py reply-approve --approve q_ab12cd34`

## OpenClaw `system.run` Examples

Use absolute skill path for reliable execution:

- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default diagnose --json"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default post --text 'OpenClaw helper is live 🦞'"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default post --text 'OpenClaw chart update 🦞' --media ./chart.png --dry-run"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default post --text 'Great point, thanks for sharing.' --in-reply-to 1892345678901234567"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default browse-twitter --handle OpenClawAI --limit 10 --json"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default mentions --limit 20 --json"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default reply-discover-run --watchlist default --approval-queue --max-tweets 8"`
- `system.run command:"cd ~/.openclaw/workspace/skills/x-twitter-helper && ./twitter-engine --account default reply-approve --list"`

## Recommended Workflows

### 1. First-time setup

1. `python src/twitter_helper.py setup`
2. `python src/twitter_helper.py app-settings`
3. `python src/twitter_helper.py auth-login`
4. `python src/twitter_helper.py doctor`

### 2. After reboot

1. `./twitter-engine restart`
2. `./twitter-engine engine-check`

### 3. Inspiration + post loop

1. `./twitter-engine inspire-tweets --topic "OpenClaw" --max-pages 2 --draft-count 5 --save data/inspiration_latest.json`
2. Select one draft
3. `./twitter-engine engine-autopost --text "<draft>"`

### 4. Mentions reply loop (safe rollout)

Draft-only:

- `./twitter-engine reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1`

Post-enabled (capped):

- `./twitter-engine reply-twitter-e2e --handle OpenClawAI --mention-limit 20 --draft-count 5 --pick 1 --post --max-posts 1`

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

Missing reply engine dependencies:

- `pip install -r requirements-reply-engine.txt`
