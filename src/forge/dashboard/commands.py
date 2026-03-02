# Copyright 2026. Dashboard CLI command.

import os
import signal
import subprocess
import sys
import webbrowser
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
SERVER_SCRIPT = DASHBOARD_DIR / "server.py"
DASHBOARD_HTML = DASHBOARD_DIR / "dashboard.html"


def _find_running_server() -> int | None:
    """Return PID of a running dashboard server, or None."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "forge/dashboard/server.py"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            return int(pids[0]) if pids[0] else None
    except (OSError, ValueError):
        pass
    return None


def cmd_dashboard(args) -> int:
    port = getattr(args, "port", 8765)
    no_browser = getattr(args, "no_browser", False)

    pid = _find_running_server()
    if pid:
        print(f"Dashboard server already running (pid {pid})")
    else:
        print(f"Starting dashboard server on http://localhost:{port}")
        proc = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            env={**os.environ, "DASHBOARD_PORT": str(port)},
            stdout=open("/tmp/forge-dashboard-server.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Server started (pid {proc.pid})")

    if not no_browser:
        url = DASHBOARD_HTML.as_uri()
        print(f"Opening {url}")
        webbrowser.open(url)

    print(f"\nAPI: http://localhost:{port}/sessions")
    print(f"Stop: pkill -f 'forge/dashboard/server.py'")
    return 0
