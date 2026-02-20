from __future__ import annotations

import sys
from typing import List, Optional

import twitter_helper


POST_COMMANDS = {
    "setup",
    "auth-login",
    "doctor",
    "auto-diagnose",
    "app-settings",
    "walkthrough",
    "openclaw-status",
    "check-auth",
    "post",
    "openclaw-autopost",
    "openclaw",
    "thread",
    "run-twitter-helper",
    "restart-setup",
}

POST_ALIASES = {
    "login": "auth-login",
    "diag": "auto-diagnose",
    "status": "openclaw-status",
    "publish": "post",
    "publish-thread": "thread",
    "quickstart": "restart-setup",
}

REPLY_SIDE_COMMANDS = {
    "reply-quick",
    "reply-twitter-e2e",
    "reply-discover-run",
    "reply-approve",
    "reply-discover",
    "reply-rank",
    "reply-ideas",
    "reply-run",
    "reply-twitter-helper",
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
    print("Post Engine")
    print("Usage: python src/post_engine.py <command> [args]")
    print("")
    print("Commands:")
    for cmd in sorted(POST_COMMANDS):
        print(f"  - {cmd}")
    print("")
    print("Aliases:")
    for alias, target in sorted(POST_ALIASES.items()):
        print(f"  - {alias} -> {target}")
    print("")
    print("For reply workflows, use: python src/reply_engine.py ...")
    print("For supply workflows alias, use: python src/supply_engine.py ...")


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    cmd = POST_ALIASES.get(args[0], args[0])
    if cmd != args[0]:
        args = [cmd, *args[1:]]

    if cmd in REPLY_SIDE_COMMANDS:
        print(f"`{cmd}` is a reply/supply-engine command, not post-engine.")
        print("Use: python src/reply_engine.py ...  or  python src/supply_engine.py ...")
        return 2

    if cmd not in POST_COMMANDS:
        print(f"Unknown post-engine command: {cmd}")
        print("Run with --help to see supported commands.")
        return 2

    return int(twitter_helper.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
