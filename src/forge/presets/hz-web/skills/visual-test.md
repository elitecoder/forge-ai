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

## Squirrel-Specific Notes

- Dev server URL is `https://localhost.adobe.com:<PORT>/new` (not plain localhost)
- Auth uses Adobe IMS staging (`auth-stg1.services.adobe.com`)
- App uses shadow DOM custom web components — use Playwright's built-in locators
- Read `squirrel-quirks.md` before generating any script
- E2E fixtures in `apps/squirrel/e2e-honeydew/fixtures/base/` define proven interaction patterns
