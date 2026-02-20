# OpenClaw Update Log

This file tracks major OpenClaw-facing helper updates in reverse chronological order.

## 2026-02-20

### `8ca3eba` - JSONL dedupe + discovery tone
- Added persistent reply dedupe log: `~/.config/twitter-engine/replied_to_<account>.jsonl` (90-day window).
- Added discovery-intent generation mode (`is_discovery=True`) so proactive replies read as intentional, not reactive spam.
- Integrated with existing double-reply guards.

### `79188ab` - Double-reply prevention + intentional discovery guardrails
- Added no-double-reply checks across:
  - `reply-discover-run`
  - `reply-approve`
  - `post --in-reply-to`
- Added manual override for intentional second reply:
  - `post --in-reply-to <ID> --force-reply-target`

### `6fa5e36` - Unique applicable reply selection
- Added reply selection layer requiring specificity to tweet/thread context.
- Added 24h phrase-prefix dedupe memory:
  - `~/.config/twitter-engine/recent_replies.jsonl`
- Added optional persona file support:
  - `~/.config/twitter-engine/persona/twitter-engine.md`

### `7c65092` - Duplicate-content auto-retry for replies
- Added automatic one-time retry with unique visible suffix when reply post fails with duplicate-content `403`.

### `8ad20ec` - Media hardening + approval queue workflow
- Hardened media posting with multi-image validation.
- Added approval queue commands:
  - `reply-discover-run --approval-queue`
  - `reply-approve --list`
  - `reply-approve --approve ...`

### `0082754` - Proactive discovery pipeline
- Added proactive search/discovery flow:
  - `search`
  - `reply-discover-run`
- Added context-aware reply drafting loop for discovery candidates.

### `aba01df` - Native mentions endpoint
- Added official `/2/users/:id/mentions` integration for reliable mention fetches.

### `af22b0a` - Secure token storage + retry hardening
- Added keyring-backed OAuth token storage.
- Added rate-limit aware retries.
- Added account-aware wrapper behavior.

## Notes
- For operational commands and runbooks, see `docs/OPENCLAW_OPERATIONS.md`.
- For skill usage, see `SKILL.md`.
