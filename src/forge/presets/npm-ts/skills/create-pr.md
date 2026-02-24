# Create Pull Request

Create a pull request for the completed pipeline changes.

## Instructions

1. Read the plan file at `{{PLAN_FILE}}` to understand what was implemented.
2. Read `{{SESSION_DIR}}/pipeline-output.md` for step results.
3. Stage and commit all changes (if not already committed).
4. Push the branch to the remote.
5. Create a PR using `gh pr create` with:
   - A concise title (under 70 chars) summarizing the change
   - A body with:
     - Summary of changes (2-3 bullet points)
     - Test plan
     - Link to any relevant issues

## Output

Write the PR URL to `{{SESSION_DIR}}/pr-url.txt`.

After creating the PR, call `forge execute pass create_pr`.
If the PR cannot be created, call `forge execute fail create_pr "reason"`.
