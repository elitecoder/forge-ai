# Copyright 2026. Unified CLI entry point for the Forge package.

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Plan deeply, execute reliably.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("plan", help="Adversarial dual-architecture planner", add_help=False)
    subparsers.add_parser("execute", help="Deterministic pipeline executor (CLI)", add_help=False)
    subparsers.add_parser("drive", help="Deterministic pipeline driver (plan → PR)", add_help=False)
    status_parser = subparsers.add_parser("status", help="Show all sessions")
    status_parser.add_argument("--active", action="store_true", help="Only show running sessions")
    status_parser.add_argument("--limit", type=int, default=20, help="Max sessions to show")
    setup_parser = subparsers.add_parser("setup", help="Install required tools and configure environment")
    setup_parser.add_argument("--force", action="store_true", help="Re-run setup even if already completed")
    setup_parser.add_argument("--preset", help="Also run preset-specific setup (e.g. npm-ts)")
    dash_parser = subparsers.add_parser("dashboard", help="Open web dashboard")
    dash_parser.add_argument("--port", type=int, default=8765, help="Server port")
    dash_parser.add_argument("--no-browser", action="store_true", help="Start server without opening browser")

    # Parse only the first arg to route to the right sub-CLI
    args, remaining = parser.parse_known_args()

    if args.command == "plan":
        sys.argv = ["forge plan"] + remaining
        from forge.planner.commands import main as planner_main
        planner_main()

    elif args.command == "execute":
        sys.argv = ["forge execute"] + remaining
        from forge.executor.commands import main as executor_main
        executor_main()

    elif args.command == "drive":
        sys.argv = ["forge drive"] + remaining
        from forge.executor.driver import main as driver_main
        driver_main()

    elif args.command == "status":
        from forge.core.events import cmd_status
        return cmd_status(args)

    elif args.command == "setup":
        from forge.setup import run_setup, run_preset_setup
        try:
            run_setup(force=args.force)
            if args.preset:
                from forge.executor.engine.pipeline_ops import resolve_preset_dir
                preset_path = str(resolve_preset_dir(args.preset))
                run_preset_setup(preset_path, force=args.force)
            print("Setup complete.")
        except RuntimeError as e:
            print(f"Setup failed: {e}", file=sys.stderr)
            return 1

    elif args.command == "dashboard":
        from forge.dashboard.commands import cmd_dashboard
        return cmd_dashboard(args)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
