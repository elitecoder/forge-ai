# Forge Plan

Adversarial dual-architecture planning system. Two independent architects design solutions,
adversarial critics evaluate them, refiners improve them, and a judge synthesizes the final plan.

## Usage

```bash
forge plan "your problem statement" [--fast]
```

Options:
- `--fast`: Use faster models for all phases (Sonnet instead of Opus)
- `--tension`: Force a specific design tension
- `--constraint-a` / `--constraint-b`: Force specific architectural constraints

The planner runs 6 phases: recon, architects, critics, refiners, judge, enrichment.
Output: `final-plan.md` in the session directory.
