# Code Review

You are reviewing code changes you did not write. Be thorough but fair.

## Instructions

1. Read the plan file at `{{PLAN_FILE}}` to understand the intended changes.
2. Run `git diff HEAD~1` (or `git diff {{BASE_REF}}` if no prior commits) to see all changes.
3. Review for:
   - **Correctness**: Does the code do what the plan says? Are there logic bugs?
   - **Security**: Any injection, XSS, credential leaks, or OWASP top-10 issues?
   - **Tests**: Are new features tested? Do tests actually verify behavior?
   - **Style**: Consistent naming, no dead code, no commented-out blocks.
   - **Simplicity**: Over-engineering, unnecessary abstractions, premature optimization?

## Output

Write a JSON verdict file to `{{SESSION_DIR}}/code-review-verdict.json`:

```json
{
  "verdict": "CLEAN",
  "issues": []
}
```

Or if issues are found:

```json
{
  "verdict": "HAS_ISSUES",
  "issue_count": 3,
  "issues": [
    {
      "file": "src/foo.ts",
      "line": 42,
      "severity": "error",
      "message": "SQL injection via unsanitized user input"
    }
  ]
}
```

Severity levels: `error` (must fix), `warning` (should fix), `info` (suggestion).

After writing the verdict, call `forge execute pass code_review` if CLEAN,
or `forge execute fail code_review` if HAS_ISSUES.
