# Architect Execute

Deterministic pipeline executor. Takes a plan and runs it through a multi-step
pipeline: code, build, lint, test, code review, visual test, PR creation.

## Usage

```bash
# Start a new pipeline
architect drive --plan final-plan.md --preset hz-web

# Resume an interrupted pipeline
architect drive --resume

# Pipeline CLI for manual step management
architect execute init full --preset hz-web --plan plan.md
architect execute status
architect execute next
```

The driver spawns fresh AI agents per step with zero context rot.
State lives on disk, not in LLM memory.
