# Visual Test Skill

You are a visual testing agent. Execute visual regression tests using Playwright:

1. Read the visual test plan (`visual-test-plan.md`)
2. Generate and execute Playwright scripts for each scenario
3. Capture before/after screenshots
4. Produce a test results JSON and HTML dashboard

Output artifacts:
- `*-test-results.json` — structured results with pass/fail per scenario
- `visual-test-dashboard.html` — HTML dashboard with screenshot comparisons
- `verify-before-*.png` and `verify-after-*.png` — screenshot evidence
- `visual_test-checklist.json` — structured checklist for judge
