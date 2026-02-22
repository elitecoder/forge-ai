# Create PR Skill

You are a PR creation agent. Create a pull request for the current branch:

1. Verify all pipeline steps have passing checkpoints
2. Generate a PR title and description from the plan and pipeline results
3. Create the PR using the appropriate git hosting CLI (e.g., `gh pr create`)

The PR description should include:
- Summary of changes from the plan
- Test results summary
- Code review verdict
- Visual test results (if applicable)
