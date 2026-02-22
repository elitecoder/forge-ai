# Copyright 2026. Unified CLI entry point for the Architect package.

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="architect",
        description="Plan deeply, execute reliably.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("plan", help="Adversarial dual-architecture planner", add_help=False)
    subparsers.add_parser("execute", help="Deterministic pipeline executor (CLI)", add_help=False)
    subparsers.add_parser("drive", help="Deterministic pipeline driver (plan → PR)", add_help=False)

    # Parse only the first arg to route to the right sub-CLI
    args, remaining = parser.parse_known_args()

    if args.command == "plan":
        sys.argv = ["architect plan"] + remaining
        from architect.planner.commands import main as planner_main
        planner_main()

    elif args.command == "execute":
        sys.argv = ["architect execute"] + remaining
        from architect.executor.commands import main as executor_main
        executor_main()

    elif args.command == "drive":
        sys.argv = ["architect drive"] + remaining
        from architect.executor.driver import main as driver_main
        driver_main()

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
