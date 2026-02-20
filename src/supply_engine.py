from __future__ import annotations

import sys
from typing import List, Optional

import reply_engine as reply_entry


def _print_help() -> None:
    print("Supply Engine")
    print("Supply engine is an alias for the Reply Engine.")
    print("Usage: python src/supply_engine.py <command> [args]")
    print("")
    print("Examples:")
    print("  - python src/supply_engine.py reply-quick --handle OpenClawAI")
    print("  - python src/supply_engine.py doctor --skip-network")


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    return int(reply_entry.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
