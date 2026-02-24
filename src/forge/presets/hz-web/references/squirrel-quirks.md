# Squirrel Quirks — Known Pitfalls & Workarounds

Read this before generating any Playwright script. These are verified issues that will cause test failures if not handled.

## 1. Media Panel Has 2 File Inputs

**Problem:** `getByTestId("panel-dropzone").locator('input[type="file"]')` resolves to 2 elements.

**Fix:** Always use `.first()`:

```javascript
const panelInput = page
  .getByTestId("panel-dropzone")
  .locator('input[type="file"]')
  .first();
```

## 2. Keyboard Shortcuts Need Focus Context

**Problem:** Keyboard shortcuts like `Q`, `W`, `S` are silently ignored if the timeline area doesn't have focus.

**Fix:** Click the timeline area before pressing shortcuts:

```javascript
// Click timeline ruler area to establish focus
await page.locator("sq-timeline-ruler-view").click();
// or click the footer content area
await page.locator("sq-app-footer-content").click();

// Now shortcuts work
await page.keyboard.press("q");
```

**For clip-specific shortcuts** (Q, W), also select the clip first:

```javascript
await clipElement.click(); // Select the clip
await page.keyboard.press("q"); // Trim to playhead
```

## 3. Canvas Click Interception

**Problem:** `hz-canvas-input` intercepts pointer events on the `<canvas>` element. Clicking the canvas area does not give focus to the timeline.

**Fix:** Click the timeline area, ruler, or footer instead of the canvas for timeline operations.

## 4. Synthetic Keyboard Events Ignored

**Problem:** `document.dispatchEvent(new KeyboardEvent(...))` creates untrusted events that Squirrel's custom keybinding manager ignores.

**Fix:** Use Playwright's `page.keyboard.press()` which sends trusted native events. When using MCP browser extension, use `browser_press_key` after clicking the timeline.

## 5. Shadow DOM Query Limitations

**Problem:** Standard `document.querySelectorAll('[data-testid]')` doesn't find elements inside shadow roots.

**Fix:** Use Playwright's built-in locators (they auto-pierce shadow DOM), or use recursive shadow DOM traversal:

```javascript
function findAll(root, results = []) {
  for (const el of root.querySelectorAll("[data-testid]")) {
    results.push(el);
  }
  for (const c of root.querySelectorAll("*")) {
    if (c.shadowRoot) findAll(c.shadowRoot, results);
  }
  return results;
}
```

## 6. Use `domcontentloaded`, Not `networkidle`

**Problem:** Squirrel has constant network activity (analytics, polling). `waitUntil: "networkidle"` may hang or timeout.

**Fix:** Use `waitUntil: "domcontentloaded"` and then wait for specific app elements:

```javascript
await page.goto(TARGET_URL, {
  waitUntil: "domcontentloaded",
  timeout: 60000,
});
// Wait for app to be ready
await page.waitForTimeout(5000);
```

## 7. Ruler Click for Playhead Positioning

**Problem:** Clicking anywhere on the timeline selects clips, not the playhead. MCP browser clicks on the ruler are unreliable due to element targeting.

**Fix:** Use Playwright's `page.mouse.click(x, y)` on the ruler's bounding box:

```javascript
const clipBox = await clip.boundingBox();
const rulerBox = await page.locator("sq-timeline-ruler-view").boundingBox();
const targetX = clipBox.x + clipBox.width * 0.5; // Midpoint
await page.mouse.click(targetX, rulerBox.y + rulerBox.height / 2);
```

## 8. Timeline Duration Frame Tolerance

**Problem:** Timeline durations may differ by 1-2 frames due to frame-alignment rounding (e.g., `00:07:06` vs `00:07:07`).

**Fix:** Compare durations with frame tolerance:

```javascript
function parseTimeToFrames(timeStr, fps = 30) {
  const [h, m, s] = timeStr.split(":").map(Number);
  return (h * 3600 + m * 60 + s) * fps;
}

const frameDiff = Math.abs(
  parseTimeToFrames(afterDuration) - parseTimeToFrames(beforeDuration)
);
expect(frameDiff).toBeLessThanOrEqual(2); // Allow 2-frame tolerance
```

## 9. Cold Start Timeout

**Problem:** If Squirrel takes >60s to load on cold start, Playwright's `goto` times out.

**Fix:** Use a generous timeout and ensure the dev server is warm:

```javascript
await page.goto(TARGET_URL, {
  waitUntil: "domcontentloaded",
  timeout: 120000, // 2 minutes for cold start
});
```

## 10. Persistent Profile Cleanup

**Problem:** macOS clears `/tmp` on reboot, deleting the persistent Chrome profile at `/tmp/playwright-chrome-profile`. First run after reboot requires re-login.

**Fix:** Accept this tradeoff. The script handles auth redirect automatically with test account credentials.

## 11. `detectDevServers()` False Positives

**Problem:** The Playwright skill's `detectDevServers()` function may detect macOS AirPlay on port 5000 as a dev server.

**Fix:** Check for the `rspack-node` process directly:

```bash
lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | grep rspack-node
```

## 12. Freeform Track Left Trim vs Gapless

**Problem:** Left trim behavior differs between track types. Freeform trim does NOT change timeline total duration, but gapless trim DOES.

**Fix:** When asserting after trim, check the track type:

```javascript
// Freeform: duration unchanged
expect(afterDuration).toBe(beforeDuration); // within frame tolerance

// Gapless: duration decreased
expect(afterDurationFrames).toBeLessThan(beforeDurationFrames);
```

## 13. Media Panel Open — Use E2E Pattern

**Problem:** Opening the Media Panel has multiple pitfalls: a race condition (DVAWV-18567), the button uses a `disabled` attribute (not property), and the button **toggles** the panel — if the persistent profile had it open, the first click closes it.

**Fix:** Use the pattern from `apps/squirrel/e2e-honeydew/fixtures/base/MediaPanelPage.ts`:

```javascript
// 1. Use regex prefix for testid (E2E pattern)
const mediaPanelButton = page.getByTestId(/^your-media-btn/);

// 2. Wait for button to be enabled (custom component uses attribute)
await mediaPanelButton.waitFor({ state: "visible", timeout: 10000 });

// 3. Click to open
await mediaPanelButton.click();
await page.waitForTimeout(1000); // Required race condition wait (DVAWV-18567)

// 4. Handle toggle: check if panel content appeared
const mediaThumbnails = page.locator("sq-media-thumbnail");
try {
  await mediaThumbnails.first().waitFor({ state: "visible", timeout: 5000 });
} catch {
  // Panel was already open — first click closed it. Click again.
  await mediaPanelButton.click();
  await page.waitForTimeout(1000);
  await mediaThumbnails.first().waitFor({ state: "visible", timeout: 10000 });
}
```

**Key details from the E2E fixture:**

- `getByTestId(/^your-media-btn/)` — regex prefix, not exact string
- Check `not.toHaveAttribute("disabled")` before clicking — custom components use the attribute, not the property
- Upload via panel uses `getByTestId("panel-dropzone").locator("input[type='file']").first()` — always `.first()` (Quirk #1)
- The `openMediaPanel()` method is in `MediaPanelPage.ts` — **always read this fixture** before writing media panel interactions

## 14. Dev Server URL Is `localhost.adobe.com`, Not `localhost`

**Problem:** The Squirrel dev server binds to `https://localhost.adobe.com:<PORT>/new`, NOT `https://localhost:<PORT>/`. Checking the wrong hostname returns connection errors even when the server is running.

**Fix:** Always use `localhost.adobe.com` for all dev server checks and navigation. The port comes from the pipeline context (`Dev server port`), defaulting to 8080:

```javascript
const DEV_PORT = "8080"; // ← set from pipeline context: Dev server port
const TARGET_URL = `https://localhost.adobe.com:${DEV_PORT}/new`;
```

To verify the server is running from bash:

```bash
curl -sk -o /dev/null -w "%{http_code}" "https://localhost.adobe.com:8080/new"
```

## 15. Playhead Cannot Move Past the Longest Timeline Item

**Problem:** The playhead cannot be positioned past the end of the longest item on the timeline. If a freeform clip is the longest item, clicking the ruler past its right edge has no effect — the playhead stays at the clip's out time.

**Impact:** Elongation tests fail silently because the playhead never actually reaches past the clip edge, so `desiredDuration ≈ currentDuration` (no-op).

**Fix:** When testing elongation on freeform clips, ensure the **gapless track is longer** than the freeform clip. Upload multiple items to the gapless track so the timeline extends beyond the freeform clip:

```javascript
// Upload 2 videos to gapless track to make it longer than any freeform clip
const dropzoneInput = page
  .getByTestId("empty-timeline-dropzone")
  .locator('input[type="file"]');
await dropzoneInput.waitFor({ state: "attached", timeout: 30000 });
await dropzoneInput.first().setInputFiles([ASSETS.video, ASSETS.video2]);
await page.waitForTimeout(3000);

// Now drop a single video as freeform — it will be shorter than the gapless track
await dropFilesOnTimeline(page, ASSETS.video);
```

## 16. Webpack Dev Server Overlay Intercepts Pointer Events

**Problem:** The webpack dev server injects an `<iframe id="webpack-dev-server-client-overlay">` that can intercept all pointer events, causing click timeouts.

**Fix:** Dismiss the overlay between scenarios:

```javascript
async function dismissOverlay(page) {
  await page.evaluate(() => {
    const overlay = document.getElementById("webpack-dev-server-client-overlay");
    if (overlay) overlay.remove();
  });
}
```

Call `dismissOverlay(page)` after each timeline clear and before each scenario.

## Quick Reference: What to Check Before Running

- [ ] `rspack-node` process is running (dev server)
- [ ] `SQUIRREL_EMAIL` and `SQUIRREL_PASSWORD` are set
- [ ] Using `channel: "chrome"` (not Playwright's Chromium)
- [ ] Using `ignoreHTTPSErrors: true`
- [ ] Using `headless: false`
- [ ] Using `domcontentloaded` (not `networkidle`)
- [ ] Media Panel file input uses `.first()`
- [ ] Timeline clicks before keyboard shortcuts
- [ ] Frame tolerance on duration comparisons
