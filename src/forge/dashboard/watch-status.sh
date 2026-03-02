#!/bin/bash

# Find the most recent session (planner or executor)
PLANNER_SESSION=$(ls -td ~/.forge/sessions/planner/*/ 2>/dev/null | head -1)
EXECUTOR_SESSION=$(ls -td ~/.forge/sessions/executor/*/ 2>/dev/null | head -1)

# Determine which session is more recent
if [ -n "$EXECUTOR_SESSION" ] && [ -n "$PLANNER_SESSION" ]; then
    EXECUTOR_TIME=$(stat -f %m "$EXECUTOR_SESSION" 2>/dev/null || stat -c %Y "$EXECUTOR_SESSION" 2>/dev/null)
    PLANNER_TIME=$(stat -f %m "$PLANNER_SESSION" 2>/dev/null || stat -c %Y "$PLANNER_SESSION" 2>/dev/null)
    if [ "$EXECUTOR_TIME" -gt "$PLANNER_TIME" ]; then
        SESSION_DIR="$EXECUTOR_SESSION"
        SESSION_TYPE="executor"
    else
        SESSION_DIR="$PLANNER_SESSION"
        SESSION_TYPE="planner"
    fi
elif [ -n "$EXECUTOR_SESSION" ]; then
    SESSION_DIR="$EXECUTOR_SESSION"
    SESSION_TYPE="executor"
elif [ -n "$PLANNER_SESSION" ]; then
    SESSION_DIR="$PLANNER_SESSION"
    SESSION_TYPE="planner"
else
    echo "No active forge sessions found."
    exit 1
fi

clear

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                   ARCHITECT STATUS MONITOR${NC}"
echo -e "${BOLD}                      (${SESSION_TYPE})${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Parse state file based on session type
if [ "$SESSION_TYPE" = "planner" ]; then
    STATE_FILE="${SESSION_DIR}.planner-state.json"
    if [ -f "$STATE_FILE" ]; then
        PROBLEM=$(grep '"problem_statement"' "$STATE_FILE" | sed 's/.*": "//;s/",$//')
        PRESET=$(grep '"preset"' "$STATE_FILE" | sed 's/.*": "//;s/",$//')
        FAST=$(grep '"fast_mode"' "$STATE_FILE" | sed 's/.*": //;s/,$//')

        echo -e "${BLUE}Problem:${NC} ${PROBLEM}"
        echo -e "${BLUE}Preset:${NC} ${PRESET} ${GRAY}(fast: ${FAST})${NC}"
        echo -e "${BLUE}Session:${NC} $(basename "$SESSION_DIR")"
        echo ""
    fi
elif [ "$SESSION_TYPE" = "executor" ]; then
    STATE_FILE="${SESSION_DIR}agent-state.json"
    if [ -f "$STATE_FILE" ]; then
        PRESET=$(grep '"preset"' "$STATE_FILE" | sed 's/.*": "//;s/",$//' | head -1)
        REPO=$(grep '"repo_dir"' "$STATE_FILE" | sed 's/.*": "//;s/",$//')

        echo -e "${BLUE}Repository:${NC} $(basename "$REPO")"
        echo -e "${BLUE}Preset:${NC} ${PRESET}"
        echo -e "${BLUE}Session:${NC} $(basename "$SESSION_DIR")"
        echo ""
    fi
fi

echo -e "${BOLD}Phase Progress:${NC}"
echo ""

if [ "$SESSION_TYPE" = "planner" ] && [ -f "$STATE_FILE" ]; then
    # Helper function to show phase status
    show_phase() {
        local phase_name=$1
        local display_name=$2
        local status=$(grep -A 3 "\"$phase_name\":" "$STATE_FILE" | grep '"status"' | sed 's/.*": "//;s/",$//')
        local started=$(grep -A 3 "\"$phase_name\":" "$STATE_FILE" | grep '"started_at"' | sed 's/.*": "//;s/"$//')
        local completed=$(grep -A 3 "\"$phase_name\":" "$STATE_FILE" | grep '"completed_at"' | sed 's/.*": "//;s/"$//')

        if [ "$status" = "complete" ]; then
            echo -e "  ${GREEN}✓${NC} ${display_name}: ${GREEN}Complete${NC} ${GRAY}(${started} → ${completed})${NC}"
        elif [ "$status" = "in_progress" ]; then
            echo -e "  ${YELLOW}●${NC} ${display_name}: ${YELLOW}In Progress${NC} ${GRAY}(started: ${started})${NC}"
        elif [ "$status" = "skipped" ]; then
            echo -e "  ${GRAY}⊘${NC} ${display_name}: ${GRAY}Skipped${NC}"
        else
            echo -e "  ${GRAY}○${NC} ${display_name}: ${GRAY}Pending${NC}"
        fi
    }

    show_phase "recon" "Recon"
    show_phase "forges" "Architects"
    show_phase "critics" "Critics"
    show_phase "refiners" "Refiners"
    show_phase "judge" "Judge"
    show_phase "enrichment" "Enrichment"

elif [ "$SESSION_TYPE" = "executor" ]; then
    # Show executor pipeline steps
    PIPELINE_STATUS="${SESSION_DIR}pipeline-status.md"
    if [ -f "$PIPELINE_STATUS" ]; then
        # Parse the table and show status
        grep "^|" "$PIPELINE_STATUS" | tail -n +2 | while IFS='|' read -r _ step status duration _ _; do
            step=$(echo "$step" | xargs)
            status=$(echo "$status" | xargs)
            # Skip separator lines
            if [ "$step" = "------" ] || [ "$step" = "Step" ]; then
                continue
            fi
            duration=$(echo "$duration" | xargs)

            if [ "$status" = "COMPLETE" ] || [ "$status" = "complete" ]; then
                echo -e "  ${GREEN}✓${NC} ${step}: ${GREEN}Complete${NC} ${GRAY}(${duration})${NC}"
            elif [ "$status" = "RUNNING" ] || [ "$status" = "in_progress" ]; then
                echo -e "  ${YELLOW}●${NC} ${step}: ${YELLOW}Running${NC} ${GRAY}(${duration})${NC}"
            elif [ "$status" = "FAILED" ] || [ "$status" = "failed" ]; then
                echo -e "  ${RED}✗${NC} ${step}: ${RED}Failed${NC} ${GRAY}(${duration})${NC}"
            else
                echo -e "  ${GRAY}○${NC} ${step}: ${GRAY}Pending${NC}"
            fi
        done
    fi
fi

echo ""
echo -e "${BOLD}Recent Activity:${NC}"
echo ""

# Show last 5 lines of activity log
if [ "$SESSION_TYPE" = "planner" ] && [ -f "${SESSION_DIR}planner-activity.log" ]; then
    tail -5 "${SESSION_DIR}planner-activity.log" | while IFS= read -r line; do
        echo -e "  ${GRAY}${line}${NC}"
    done
elif [ "$SESSION_TYPE" = "executor" ] && [ -f "${SESSION_DIR}pipeline-activity.log" ]; then
    tail -5 "${SESSION_DIR}pipeline-activity.log" | while IFS= read -r line; do
        echo -e "  ${GRAY}${line}${NC}"
    done
fi

echo ""
echo -e "${BOLD}Design Documents:${NC}"
echo ""

# Show file sizes to indicate progress
for file in design-a.md design-b.md critique-a.md critique-b.md final-plan.md; do
    if [ -f "${SESSION_DIR}${file}" ]; then
        size=$(ls -lh "${SESSION_DIR}${file}" | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} ${file} ${GRAY}(${size})${NC}"
    fi
done

echo ""
echo -e "${GRAY}Session directory: ${SESSION_DIR}${NC}"
echo -e "${GRAY}Updated: $(date '+%H:%M:%S')${NC}"
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
