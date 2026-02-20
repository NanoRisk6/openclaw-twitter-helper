from __future__ import annotations

import sys
from typing import List, Optional

import twitter_helper
from reply_engine import cli as reply_cli


TWITTER_HELPER_REPLY_COMMANDS = {
    "reply-quick",
    "reply-twitter-e2e",
    "reply-discover-run",
    "reply-approve",
    "reply-discover",
    "reply-rank",
    "reply-ideas",
    "reply-run",
    "reply-twitter-helper",
}

REPLY_CLI_COMMANDS = {
    "doctor",
    "discover",
    "rank",
    "ideas",
    "run",
    "twitter-helper",
    "many-ways",
    "twitter-e2e",
    "twitter-discovery",
    "queue-list",
    "queue-approve",
    "queue-clean",
}


def _print_help() -> None:
    print("Reply Engine")
    print("Usage: python src/reply_engine.py <command> [args]")
    print("")
    print("One-shot / helper commands:")
    for cmd in sorted(TWITTER_HELPER_REPLY_COMMANDS):
        print(f"  - {cmd}")
    print("")
    print("Dedicated reply-engine CLI commands:")
    for cmd in sorted(REPLY_CLI_COMMANDS):
        print(f"  - {cmd}")
    print("")
    print("For posting/setup workflows, use: python src/post_engine.py ...")


def _run_reply_cli(args: List[str]) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["reply_engine.py", *args]
        reply_cli.main()
    finally:
        sys.argv = old_argv
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    cmd = args[0]
    if cmd in TWITTER_HELPER_REPLY_COMMANDS:
        return int(twitter_helper.main(args))
    if cmd in REPLY_CLI_COMMANDS:
        return _run_reply_cli(args)

    print(f"Unknown reply-engine command: {cmd}")
    print("Run with --help to see supported commands.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
