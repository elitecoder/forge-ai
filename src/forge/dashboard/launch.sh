#!/bin/bash

# Architect Dashboard Launcher
# Starts the session server and opens the web dashboard.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/server.py"
DASHBOARD_HTML="$SCRIPT_DIR/dashboard.html"

echo "Architect Dashboard Launcher"
echo "============================"
echo ""

# Check if server is already running
if pgrep -f "forge/dashboard/server.py" > /dev/null; then
    echo "Server is already running"
else
    echo "Starting session server..."
    python3 "$SERVER_SCRIPT" > /tmp/forge-dashboard-server.log 2>&1 &
    sleep 2
    echo "Server started on http://localhost:8765"
fi

echo ""
echo "Opening dashboard in browser..."
open "$DASHBOARD_HTML"

echo ""
echo "Available endpoints:"
echo "  Dashboard: file://$DASHBOARD_HTML"
echo "  API Sessions: http://localhost:8765/sessions"
echo "  API Session Detail: http://localhost:8765/session/<session-id>"
echo "  API Project Detail: http://localhost:8765/project?planner=<id>"
echo ""
echo "To stop the server: pkill -f 'forge/dashboard/server.py'"
echo ""
