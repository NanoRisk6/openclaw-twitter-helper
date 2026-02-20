from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reply_engine.twitter_helper import run_twitter_helper
else:
    from .twitter_helper import run_twitter_helper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Legacy Twitter reply entrypoint")
    p.add_argument("--tweet", default="2024820748980748765", help="tweet URL or ID")
    p.add_argument("--draft-count", type=int, default=5)
    p.add_argument("--pick", type=int, default=4, help="1-based draft index")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-path", default="data/replies.jsonl")
    return p


def main() -> None:
    args = build_parser().parse_args()
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


if __name__ == "__main__":
    main()
