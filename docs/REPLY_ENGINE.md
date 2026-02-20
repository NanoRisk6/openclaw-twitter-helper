# OpenClaw Twitter Helper Reply Engine

The Reply Engine is designed for autonomous-but-safe engagement on X:

- unique and context-aware replies
- no accidental double-replies
- intentional discovery workflows
- queue-first human review when needed

## Core Principles

- Unique + Applicable: replies reference concrete details from target tweet/thread.
- No Double-Replies: persistent dedupe by tweet ID.
- Intentional Discovery: proactive search replies are filtered and confidence-gated.
- Topical Relevance: off-topic/non-sequitur drafts are skipped.
- Human-in-the-Loop: approval queue mode for safe review before posting.

## Architecture

```text
reply engine
├── Mentions workflow
│   └── /2/users/:id/mentions (native) + since_id checkpoints
├── Discovery workflow
│   └── /2/tweets/search/recent + scoring + watchlist/query controls
├── Context assembly
│   └── full conversation parent chain + optional web enrichment
├── Draft generation
│   └── unique/applicable filtering + confidence scoring
├── Safety
│   ├── dedupe (replied_to_<account>.jsonl)
│   ├── approval queue (q_*.json)
│   └── confidence/score thresholds
└── Output
    ├── draft
    ├── queue
    └── post (with visibility verification + logging)
```

## Tool Entry Points

- Dedicated tool: `./reply-engine`
- Helper wrapper: `./twitter-engine`

## High-Value Commands

### 1) Mentions workflow

```bash
./reply-engine twitter-e2e \
  --handle OpenClawAI \
  --mention-limit 20 \
  --approval-queue \
  --min-confidence 75
```

Useful flags:
- `--since-id <tweet_id>`
- `--post --max-posts 1`
- `--web-enrich --web-context-items 2`

### 2) Proactive discovery workflow

```bash
./reply-engine twitter-discovery \
  --query "openclaw OR \"local ai\" OR \"open source ai\" lang:en -is:retweet min_faves:5" \
  --limit 15 \
  --min-score 20 \
  --min-confidence 78 \
  --web-enrich \
  --web-context-items 2 \
  --approval-queue
```

### 3) Approval queue review/post

```bash
./reply-engine queue-list
./reply-engine queue-approve --ids q_12345678
./reply-engine queue-approve --ids q_12345678 --dry-run
```

### 4) Multi-style reply generation for one tweet

```bash
./reply-engine many-ways --tweet "https://x.com/OpenClawAI/status/2024820748980748765"
./reply-engine many-ways --tweet "2024820748980748765" --modes direct,curious,technical --json
```

### 5) Reply-engine doctor (health + readiness)

```bash
./reply-engine doctor
./reply-engine doctor --skip-network
./reply-engine doctor --json
```

Checks in one command:
- Queue health (invalid entries, duplicate targets, already-replied targets)
- State-file integrity (`replied_to_*.jsonl`, `reply_engine_runs.jsonl`, mention checkpoint format)
- OAuth2 posting readiness via shared helper auth path

## Dedupe + State Files

- Replied log: `~/.config/twitter-engine/replied_to_<account>.jsonl`
- Queue files: `~/.config/twitter-engine/approval_queue/q_<id>.json`
- Mentions checkpoint: `~/.config/twitter-engine/last_mention_id_<account>.txt`
- Run audit log: `~/.config/twitter-engine/reply_engine_runs.jsonl`
- Watchlists: `~/.config/twitter-engine/watchlists.json`
- Persona: `~/.config/twitter-engine/persona/twitter-engine.md`

## Continuous Mode (Queue-First)

```bash
*/20 * * * * cd /path/to/twitter-engine && ./reply-engine twitter-discovery --query "openclaw OR local ai lang:en -is:retweet min_faves:5" --limit 10 --min-score 20 --min-confidence 78 --approval-queue >> logs/reply.log 2>&1
```

Then periodically:

```bash
./reply-engine queue-list
./reply-engine queue-approve --ids q_12345678 q_23456789 --max-posts 2
```

## Recommended Operational Sequence

1. `./twitter-engine --account default diagnose`
2. `./reply-engine twitter-discovery ... --approval-queue`
3. `./reply-engine queue-list`
4. `./reply-engine queue-approve --ids ...`

## Notes

- `--approval-queue` and `--post` are mutually exclusive in intent; queue-first is safer.
- Use `--web-enrich` for additional factual context when threads are thin.
- Dedupe protections prevent accidental repeat replies to the same target tweet.
