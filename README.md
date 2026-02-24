# Forge

**Adversarial multi-agent planning and deterministic pipeline execution for any AI coding tool.**

Forge brings rigorous, adversarial architectural planning and reliable step-by-step execution to AI-assisted development. It works with Claude Code, Codex CLI, Cursor, or directly via API — the provider is pluggable.

Two subcommands. One idea: *think deeply, then execute reliably.*

```
pip install forgeai
forge plan "add user authentication"
forge execute --preset npm-ts
```

## How It Works

**Planning** — Two independent AI architects design competing solutions. Adversarial critics tear each design apart. Refiners address the critiques. A judge synthesizes the best elements into a final plan. The result: plans that have been stress-tested before a single line of code is written.

**Execution** — A deterministic pipeline runs your plan through configurable steps (code, build, lint, test, review, PR) with evidence-based validation at each stage. State lives on disk, not in LLM memory — every step gets a fresh agent with zero context rot.

## Installation

```bash
# From PyPI
pip install forgeai

# From source
git clone https://github.com/elitecoder/forge.git
cd forge
pip install -e .
```

Requires Python 3.12+ and an AI coding tool (Claude Code, Codex CLI, or Cursor).

## Quick Start

```bash
# Plan a feature with adversarial multi-agent planning
forge plan "add role-based access control"

# Execute the plan through a deterministic pipeline
forge execute --preset npm-ts

# Or run the full plan-to-PR workflow
forge drive --plan path/to/plan.md --preset npm-ts
```

### Fast Mode

Use `--fast` to run all phases with faster models (lower cost, less rigor):

```bash
forge plan "add a 404 page" --fast
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                  forge CLI                   │
│           plan | execute | drive             │
├──────────────────┬──────────────────────────┤
│     Planner      │       Executor           │
│  ┌────────────┐  │  ┌────────────────────┐  │
│  │ Recon      │  │  │ Pipeline DAG       │  │
│  │ Architects │  │  │ Evidence Checker   │  │
│  │ Critics    │  │  │ Judge Engine       │  │
│  │ Refiners   │  │  │ Step Agents        │  │
│  │ Judge      │  │  │ Checkpoint/Resume  │  │
│  │ Enrichment │  │  │ Pre-PR Gate        │  │
│  └────────────┘  │  └────────────────────┘  │
├──────────────────┴──────────────────────────┤
│              Provider Layer                  │
│     Claude | Codex | Cursor | Direct API     │
├─────────────────────────────────────────────┤
│           Presets (npm-ts, python-uv, ...)   │
└─────────────────────────────────────────────┘
```

## Presets

Presets define how Forge interacts with your tech stack — build commands, lint tools, test runners, pipeline steps, and evidence rules. Create a preset for your project:

```
my-preset/
├── manifest.json    # Pipeline definition, steps, evidence rules
└── skills/          # Step-specific prompt templates
    ├── code-review.md
    └── create-pr.md
```

Pass your preset with `--preset`:

```bash
forge execute --preset ./my-preset
```

## Provider Protocol

Forge orchestrates AI tools through a pluggable provider interface:

```python
class Provider(Protocol):
    def run_agent(self, prompt, model, max_turns, cwd, ...) -> AgentResult: ...
    def run_judge(self, prompt, model, ...) -> str: ...
```

Model tiers abstract away vendor-specific model names:

| Tier | Purpose | Claude | OpenAI |
|------|---------|--------|--------|
| `reasoning` | Planning, judging | Opus | o3 |
| `balanced` | Implementation | Sonnet | GPT-4.1 |
| `fast` | Quick fixes | Haiku | GPT-4.1-mini |

## License

Apache 2.0 — see [LICENSE](LICENSE).
