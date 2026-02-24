#!/usr/bin/env python3
"""Standalone pipeline CLI entry point.

AI agents call: python3 pipeline_cli.py <pass|fail|reset|...> <step> [args]
Delegates to forge.executor.commands.main().
"""

from forge.executor.commands import main

if __name__ == "__main__":
    main()
