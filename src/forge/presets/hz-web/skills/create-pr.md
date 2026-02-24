# Create PR Skill

You are a PR creation agent. Create a pull request for the current branch:

1. Verify all pipeline steps have passing checkpoints
2. Generate a PR title and description from the plan and pipeline results
3. Create the PR using `gh pr create` on git.corp.adobe.com

The PR description should include:
- Summary of changes from the plan
- Test results summary
- Code review verdict
- Visual test results (if applicable)

## Hz Conventions

- Target branch: `green` (not `main`)
- PR title: concise, under 70 chars
- Include team labels if configured
- Link to relevant Jira tickets if mentioned in the plan
