from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    load_candidates,
    load_scored,
    run_discovery,
    run_ideas,
    run_rank,
    save_candidates,
    save_scored,
)
from .twitter_helper import (
    approve_queue,
    cleanup_queue,
    DEFAULT_REPLY_MODES,
    list_approval_queue,
    run_reply_many_ways,
    run_discovery_workflow,
    run_mentions_workflow,
    run_twitter_helper,
)


def _parse_keywords(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenClaw reply engine helper")
    sub = p.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="Discover candidate conversations")
    discover.add_argument("--keywords", required=True, help="comma-separated keywords")
    discover.add_argument("--limit", type=int, default=30)
    discover.add_argument("--local-input", default=None)
    discover.add_argument("--output", required=True)

    rank = sub.add_parser("rank", help="Score and rank candidates")
    rank.add_argument("--input", required=True)
    rank.add_argument("--keywords", required=True, help="comma-separated keywords")
    rank.add_argument("--include-weak", action="store_true", help="include low-relevance matches")
    rank.add_argument("--output", required=True)

    ideas = sub.add_parser("ideas", help="Generate reply ideas markdown")
    ideas.add_argument("--input", required=True)
    ideas.add_argument("--top", type=int, default=20)
    ideas.add_argument("--output", required=True)

    run = sub.add_parser("run", help="End-to-end pipeline")
    run.add_argument("--keywords", required=True, help="comma-separated keywords")
    run.add_argument("--limit", type=int, default=30)
    run.add_argument("--local-input", default=None)
    run.add_argument("--include-weak", action="store_true", help="include low-relevance matches")
    run.add_argument("--output", required=True, help="markdown output path")

    tw = sub.add_parser("twitter-helper", help="Generate and optionally post a Twitter reply")
    tw.add_argument("--tweet", required=True, help="tweet URL or tweet ID")
    tw.add_argument("--draft-count", type=int, default=5)
    tw.add_argument("--pick", type=int, default=1, help="1-based draft index to post")
    tw.add_argument("--dry-run", action="store_true", help="generate drafts but do not post")
    tw.add_argument("--log-path", default="data/replies.jsonl", help="jsonl log output path")

    many = sub.add_parser("many-ways", help="Generate multiple stylistic reply options for one tweet")
    many.add_argument("--tweet", required=True, help="tweet URL or tweet ID")
    many.add_argument(
        "--modes",
        default=",".join(DEFAULT_REPLY_MODES),
        help="comma-separated modes (direct,curious,witty,technical,supportive,question)",
    )
    many.add_argument("--json", action="store_true", help="print JSON output")

    e2e = sub.add_parser("twitter-e2e", help="End-to-end mentions workflow for OpenClaw")
    e2e.add_argument("--handle", default="OpenClawAI", help="target account handle")
    e2e.add_argument("--mention-limit", type=int, default=20)
    e2e.add_argument("--since-id", default=None, help="Only include mentions newer than this tweet ID")
    e2e.add_argument("--draft-count", type=int, default=5)
    e2e.add_argument("--pick", type=int, default=1, help="1-based draft index")
    e2e.add_argument("--post", action="store_true", help="post replies (default is draft-only)")
    e2e.add_argument("--max-posts", type=int, default=3, help="max replies to post in one run")
    e2e.add_argument("--approval-queue", action="store_true", help="queue qualified replies instead of posting")
    e2e.add_argument("--min-confidence", type=int, default=70, help="minimum confidence to queue/post")
    e2e.add_argument("--web-enrich", action="store_true", help="enrich draft context with lightweight web snippets")
    e2e.add_argument("--web-context-items", type=int, default=2, help="max web snippets per candidate")
    e2e.add_argument("--log-path", default="data/replies.jsonl")
    e2e.add_argument("--report-path", default="data/mentions_report.json")

    d2e = sub.add_parser("twitter-discovery", help="End-to-end discovery workflow with queue/post controls")
    d2e.add_argument("--query", required=True, help="recent search query")
    d2e.add_argument("--limit", type=int, default=20)
    d2e.add_argument("--since-id", default=None)
    d2e.add_argument("--draft-count", type=int, default=5)
    d2e.add_argument("--pick", type=int, default=1, help="1-based draft index")
    d2e.add_argument("--post", action="store_true", help="post replies")
    d2e.add_argument("--approval-queue", action="store_true", help="queue qualified replies")
    d2e.add_argument("--min-score", type=int, default=20)
    d2e.add_argument("--min-confidence", type=int, default=70)
    d2e.add_argument("--max-posts", type=int, default=3)
    d2e.add_argument("--web-enrich", action="store_true", help="enrich draft context with lightweight web snippets")
    d2e.add_argument("--web-context-items", type=int, default=2, help="max web snippets per candidate")
    d2e.add_argument("--log-path", default="data/replies.jsonl")
    d2e.add_argument("--report-path", default="data/discovery_report.json")

    qlist = sub.add_parser("queue-list", help="List pending approval queue items")
    qlist.add_argument("--json", action="store_true")

    qapprove = sub.add_parser("queue-approve", help="Approve queued replies and post")
    qapprove.add_argument("--ids", nargs="*", default=[], help="queue ids (q_xxx or xxx); empty means all")
    qapprove.add_argument("--dry-run", action="store_true")
    qapprove.add_argument("--max-posts", type=int, default=None)
    qapprove.add_argument("--log-path", default="data/replies.jsonl")

    qclean = sub.add_parser("queue-clean", help="Remove invalid/duplicate/already-replied queue items")
    qclean.add_argument("--keep-duplicates", action="store_true", help="do not remove duplicate target queue entries")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"twitter-e2e", "twitter-discovery"}:
        if args.post and args.approval_queue:
            parser.error("Use either --post or --approval-queue, not both.")
        if args.max_posts is not None and args.max_posts < 1:
            parser.error("--max-posts must be >= 1")
        if args.min_confidence < 0 or args.min_confidence > 100:
            parser.error("--min-confidence must be between 0 and 100")
    if args.command == "twitter-discovery":
        if args.min_score < 0:
            parser.error("--min-score must be >= 0")

    if args.command == "discover":
        keywords = _parse_keywords(args.keywords)
        candidates = run_discovery(keywords, limit=args.limit, local_input=args.local_input)
        save_candidates(candidates, args.output)
        print(f"discovered {len(candidates)} candidates -> {args.output}")
        return

    if args.command == "rank":
        keywords = _parse_keywords(args.keywords)
        candidates = load_candidates(args.input)
        scored = run_rank(candidates, keywords, include_weak=args.include_weak)
        save_scored(scored, args.output)
        print(f"ranked {len(scored)} candidates -> {args.output}")
        return

    if args.command == "ideas":
        scored = load_scored(args.input)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ideas(scored, top=args.top, out_path=str(out))
        print(f"wrote ideas -> {out}")
        return

    if args.command == "run":
        keywords = _parse_keywords(args.keywords)
        candidates = run_discovery(keywords, limit=args.limit, local_input=args.local_input)
        scored = run_rank(candidates, keywords, include_weak=args.include_weak)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ideas(scored, top=min(20, len(scored)), out_path=str(out))
        print(f"discovered {len(candidates)} | wrote ideas -> {out}")
        return

    if args.command == "twitter-helper":
        result = run_twitter_helper(
            tweet=args.tweet,
            draft_count=args.draft_count,
            pick=args.pick,
            dry_run=args.dry_run,
            log_path=args.log_path,
        )
        print(f"tweet: {result['tweet_id']} (@{result['author']})")
        print("drafts:")
        for i, draft in enumerate(result["drafts"], start=1):
            print(f"{i}. {draft}")
        print(f"picked: #{result['picked_index']}")
        if result["dry_run"]:
            print("dry-run: no reply posted")
        else:
            print(f"posted: {result['reply_url']}")
            print(f"log: {result['log_path']}")
        return

    if args.command == "many-ways":
        result = run_reply_many_ways(
            tweet=args.tweet,
            modes=[x.strip() for x in args.modes.split(",") if x.strip()],
        )
        if args.json:
            import json
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        print(f"tweet: {result['tweet_id']} (@{result['author']})")
        print("reply variants:")
        for mode, text in result["replies"].items():
            print(f"- {mode}: {text}")
        return

    if args.command == "twitter-e2e":
        result = run_mentions_workflow(
            handle=args.handle,
            mention_limit=args.mention_limit,
            since_id=args.since_id,
            draft_count=args.draft_count,
            pick=args.pick,
            post=args.post,
            max_posts=args.max_posts,
            approval_queue=args.approval_queue,
            min_confidence=args.min_confidence,
            web_enrich=args.web_enrich,
            web_context_items=args.web_context_items,
            log_path=args.log_path,
            report_path=args.report_path,
        )
        print(f"handle: @{result['handle']}")
        print(f"mentions fetched: {result['fetched_mentions']}")
        print(f"replies posted: {result['posted_replies']}")
        print(f"report: {result['report_path']}")
        if result["post"]:
            print(f"log: {result['log_path']}")
        for item in result["results"]:
            print(f"- {item['status']} | {item['tweet_id']} | @{item['author']}")
        return

    if args.command == "twitter-discovery":
        result = run_discovery_workflow(
            query=args.query,
            limit=args.limit,
            since_id=args.since_id,
            draft_count=args.draft_count,
            pick=args.pick,
            post=args.post,
            approval_queue=args.approval_queue,
            min_score=args.min_score,
            min_confidence=args.min_confidence,
            max_posts=args.max_posts,
            web_enrich=args.web_enrich,
            web_context_items=args.web_context_items,
            log_path=args.log_path,
            report_path=args.report_path,
        )
        print(f"query: {result['query']}")
        print(f"fetched: {result['fetched_tweets']}")
        print(f"posted: {result['posted_replies']}")
        print(f"queued: {result['queued_replies']}")
        print(f"report: {result['report_path']}")
        for item in result["results"]:
            print(f"- {item['status']} | {item['tweet_id']} | @{item.get('author', 'unknown')}")
        return

    if args.command == "queue-list":
        rows = list_approval_queue()
        if args.json:
            import json
            print(json.dumps({"count": len(rows), "items": rows}, ensure_ascii=False, indent=2))
        else:
            print(f"pending: {len(rows)}")
            for row in rows:
                qid = str(row.get("id", ""))
                conf = row.get("confidence", "n/a")
                tid = str(row.get("tweet_id", "") or row.get("in_reply_to", ""))
                txt = str(row.get("text", "")).replace("\n", " ").strip()
                print(f"- q_{qid} | tweet={tid} | conf={conf} | {txt[:120]}")
        return

    if args.command == "queue-approve":
        result = approve_queue(
            ids=args.ids,
            dry_run=args.dry_run,
            max_posts=args.max_posts,
            log_path=args.log_path,
        )
        print(f"posted: {result['posted']} | skipped: {result['skipped']}")
        for row in result["results"]:
            print(f"- {row['status']} | {row.get('id', '')} | {row.get('tweet_id', '')}")
        return

    if args.command == "queue-clean":
        result = cleanup_queue(remove_duplicates=not args.keep_duplicates)
        print(
            f"kept: {result['kept']} | removed: {result['removed']} | "
            f"invalid: {result['reasons']['invalid']} | "
            f"already_replied: {result['reasons']['already_replied']} | "
            f"duplicate_target: {result['reasons']['duplicate_target']}"
        )
        return


if __name__ == "__main__":
    main()
