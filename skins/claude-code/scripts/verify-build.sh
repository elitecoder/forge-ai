#!/bin/bash
# Stop hook: verify build passes before allowing agent to stop.
# Only runs the build when the repo state changed since the last successful check.
# Exit 0 = allow stop, exit 2 = block stop (agent continues with stderr feedback).

set -euo pipefail

INPUT=$(cat)

# Note: we intentionally do NOT check stop_hook_active here.
# Broken code must never be allowed to proceed. The agent is bounded
# by max_turns/timeout, so infinite loops are not a real risk.

# Build command: explicit from driver, or derived from changed files
BUILD_CMD="${FORGE_BUILD_CMD:-}"

if [ -z "$BUILD_CMD" ]; then
  # Derive Bazel build targets from changed + new files
  TARGETS=$( (git diff HEAD --name-only 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null) | while read -r file; do
    dir=$(dirname "$file")
    while [ "$dir" != "." ]; do
      if [ -f "$dir/BUILD.bazel" ] || [ -f "$dir/BUILD" ]; then
        echo "//$dir/..."
        break
      fi
      dir=$(dirname "$dir")
    done
  done | sort -u | tr '\n' ' ')

  if [ -n "$TARGETS" ]; then
    BUILD_CMD="bazel build $TARGETS"
  else
    exit 0
  fi
fi

# ── Hash-based change detection ──────────────────────────────────
# Compute a hash of tracked diffs + working tree status.
# Skip the build if nothing changed since the last successful check.

SESSION_DIR="${FORGE_SESSION_DIR:-}"
if [ -z "$SESSION_DIR" ]; then
  exit 0
fi

compute_hash() {
  (git diff HEAD 2>/dev/null; git status --porcelain 2>/dev/null) \
    | shasum -a 256 | cut -d' ' -f1
}

HASH_FILE="${SESSION_DIR}/build-hash"

CURRENT_HASH=$(compute_hash)

if [ -f "$HASH_FILE" ]; then
  STORED_HASH=$(cat "$HASH_FILE")
  if [ "$CURRENT_HASH" = "$STORED_HASH" ]; then
    exit 0
  fi
fi

# State changed (or first run) — run the build
BUILD_OUTPUT=$(eval "$BUILD_CMD" 2>&1) && BUILD_EXIT=0 || BUILD_EXIT=$?

if [ "$BUILD_EXIT" -eq 0 ]; then
  echo "$CURRENT_HASH" > "$HASH_FILE"
  exit 0
fi

# Build failed — extract the most useful lines for the agent
TAIL=$(echo "$BUILD_OUTPUT" | tail -40)

cat >&2 <<EOF
Build verification failed (exit code $BUILD_EXIT). Fix the build errors before stopping.

$TAIL
EOF

exit 2
