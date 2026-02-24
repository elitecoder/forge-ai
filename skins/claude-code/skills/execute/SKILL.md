# Forge Execute

Deterministic pipeline executor. Takes a plan and runs it through a multi-step
pipeline: code, build, lint, test, code review, visual test, PR creation.

## Usage

```bash
# Start a new pipeline
forge drive --plan final-plan.md --preset <your-preset>

# Resume an interrupted pipeline
forge drive --resume

# Pipeline CLI for manual step management
forge execute init full --preset <your-preset> --plan plan.md
forge execute status
forge execute next
```

The driver spawns fresh AI agents per step with zero context rot.
State lives on disk, not in LLM memory.
