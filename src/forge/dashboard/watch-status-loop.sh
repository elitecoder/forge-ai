#!/bin/bash

# Self-refreshing terminal status monitor
# Press Ctrl+C to exit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Press Ctrl+C to exit..."
echo ""

while true; do
    "$SCRIPT_DIR/watch-status.sh"
    sleep 2
done
