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
from .twitter_helper import run_mentions_workflow, run_twitter_helper


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

    e2e = sub.add_parser("twitter-e2e", help="End-to-end mentions workflow for OpenClaw")
    e2e.add_argument("--handle", default="OpenClawAI", help="target account handle")
    e2e.add_argument("--mention-limit", type=int, default=20)
    e2e.add_argument("--draft-count", type=int, default=5)
    e2e.add_argument("--pick", type=int, default=1, help="1-based draft index")
    e2e.add_argument("--post", action="store_true", help="post replies (default is draft-only)")
    e2e.add_argument("--max-posts", type=int, default=3, help="max replies to post in one run")
    e2e.add_argument("--log-path", default="data/replies.jsonl")
    e2e.add_argument("--report-path", default="data/mentions_report.json")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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

    if args.command == "twitter-e2e":
        result = run_mentions_workflow(
            handle=args.handle,
            mention_limit=args.mention_limit,
            draft_count=args.draft_count,
            pick=args.pick,
            post=args.post,
            max_posts=args.max_posts,
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


if __name__ == "__main__":
    main()
