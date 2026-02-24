# Auto-Fix Retry Strategy

Detailed rules for the visual test executor's auto-fix retry loop.

## Overview

When a visual test fails, the executor attempts to automatically fix the issue and retry, up to 3 total attempts.

## Retry Rules

### Rule 1: Max 3 Attempts

Each test execution gets at most 3 attempts:

```
Attempt 1 → FAIL → Fix → Attempt 2 → FAIL → Fix → Attempt 3 → PASS/FAIL(escalate)
```

### Rule 2: Different Root Cause Each Time

Each retry must identify a _different_ root cause. If the same assertion fails 3 times with the same measurement (e.g., same wrong width), stop — it's not a transient issue.

**Track what was tried:**

```
Attempt 1: FAIL — clip width didn't decrease
  → Root cause: keyboard shortcut Q didn't fire
  → Fix: added timeline click before keyboard press

Attempt 2: FAIL — wrong clip was trimmed
  → Root cause: wrong selector (used gapless instead of freeform)
  → Fix: changed selector to /^trackItem-freeform-video-/

Attempt 3: PASS ✅
```

### Rule 3: Fix the Code, Not the Test

The test plan is the spec. If the test fails:

- **Code is wrong** → Fix the source code
- **Test plan was wrong** → User would have caught this during review; fix the plan only if it's clearly a planning error
- **Test mechanics are wrong** (focus, timing, selector) → Fix the script

### Rule 4: Same Measurement = Escalate

If the assertion fails with the **exact same measurement** across 3 attempts:

```
Attempt 1: expected width < 500, got 750
Attempt 2: expected width < 500, got 750 (SAME)
Attempt 3: expected width < 500, got 750 (SAME — ESCALATE)
```

This means the code change isn't producing the expected visual result. Escalate to user.

### Rule 5: Always Screenshot on Failure

Every failed attempt must capture a screenshot for diagnostics:

```javascript
await page.screenshot({
  path: `/tmp/verify-fail-attempt${attemptNumber}.png`,
  fullPage: true,
});
```

## Failure Categories and Fix Strategies

### Category A: Focus Issues

**Symptoms:**

- Keyboard shortcut had no effect
- Wrong element received the input
- Action was silently ignored

**Fix strategies:**

1. Click the timeline area before pressing shortcut
2. Click the specific clip before clip-specific shortcuts
3. Ensure no modal/dialog is stealing focus

### Category B: Timing Issues

**Symptoms:**

- Element not found (but exists after manual check)
- Assertion checked before DOM updated
- Width/position still shows old value

**Fix strategies:**

1. Increase `toPass` timeout
2. Add `waitForTimeout` before assertion (last resort)
3. Wait for specific element state: `waitFor({ state: "visible" })`
4. Use `elementUpdated()` or `nextFrame()` patterns

### Category C: Selector Issues

**Symptoms:**

- Element not found
- Wrong element matched (gapless vs freeform)
- Multiple elements matched (need `.first()`)

**Fix strategies:**

1. Check data-testid pattern against `testid-selectors.md`
2. Use regex patterns: `getByTestId(/^trackItem-freeform-video-/)`
3. Add `.first()` for multiple matches
4. Use shadow DOM traversal for deep elements

### Category D: Code Logic Issues

**Symptoms:**

- Assertion fails consistently
- Visual output doesn't match expectation
- CSS class is wrong, element is wrong size, etc.

**Fix strategies:**

1. Re-read the code change to understand what's wrong
2. Fix the source code (not the test)
3. Run lint after fixing
4. Re-run the visual test

### Category E: Infrastructure Issues

**Symptoms:**

- Dev server not running
- Auth failed
- Browser crashed
- Network error

**Fix strategies:**

1. Check if `rspack-node` is running
2. Re-source credentials and retry auth
3. Close and re-launch browser context
4. **Do NOT count infrastructure retries toward the 3-attempt limit**

## Escalation Format

When all 3 attempts fail, report to the user with this format:

```
## Visual Test Failed: [Scenario Name]

### Expected
[What the test plan said should happen]

### Actual
[What actually happened, with measurements]

### Screenshots
- Before: /tmp/verify-before.png
- After attempt 1: /tmp/verify-fail-attempt1.png
- After attempt 2: /tmp/verify-fail-attempt2.png
- After attempt 3: /tmp/verify-fail-attempt3.png

### What Was Tried
1. Attempt 1: [root cause] → [fix applied] → [result]
2. Attempt 2: [root cause] → [fix applied] → [result]
3. Attempt 3: [root cause] → [fix applied] → [result]

### Possible Causes
- [hypothesis 1]
- [hypothesis 2]

### Recommended Next Steps
- [suggestion for user]
```

## Decision Tree

```
Test failed
  ├── Is it a focus issue? → Fix focus → Retry
  ├── Is it a timing issue? → Increase timeout → Retry
  ├── Is it a selector issue? → Fix selector → Retry
  ├── Is it a code logic issue? → Fix source code → Lint → Retry
  ├── Is it infrastructure? → Fix infra (doesn't count) → Retry
  └── Same failure 3 times? → Escalate to user
```
