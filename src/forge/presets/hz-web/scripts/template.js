/**
 * Squirrel Visual Test Template
 *
 * Base template for Playwright visual verification scripts.
 * Generated scripts should follow this structure.
 *
 * Usage:
 *   source ~/.config/squirrel/credentials.env
 *   cd ~/.cursor/skills/playwright-skill && node run.js /tmp/playwright-test-{name}.js
 */

const { chromium } = require("playwright");
const path = require("path");
const os = require("os");
const fs = require("fs");

// ============================================================
// Configuration
// ============================================================

const DEV_PORT = "8080"; // ← set from pipeline context: Dev server port
const TARGET_URL = `https://localhost.adobe.com:${DEV_PORT}/new`;
const EMAIL = process.env.SQUIRREL_EMAIL;
const PASSWORD = process.env.SQUIRREL_PASSWORD;
const PROFILE_DIR = path.join(os.tmpdir(), "playwright-chrome-profile");
const SCREENSHOT_DIR = process.env.PIPELINE_SESSION_DIR;
const FEATURE_NAME = "REPLACE_ME"; // ← set to kebab-case feature name

// ============================================================
// Helpers
// ============================================================

const results = [];
const consoleErrors = [];

function log(msg) {
  console.log(`[TEST] ${msg}`);
}

/**
 * Record a passing scenario.
 * @param {number} number - Scenario number (1-based)
 * @param {string} title  - Short scenario title
 * @param {object} [opts] - { subtitle, details, tags, assertions, screenshots }
 */
function pass(number, title, opts = {}) {
  const entry = { number, title, status: "PASS", ...opts };
  results.push(entry);
  log(`✅ PASS: S${number} — ${title}${opts.details ? " — " + opts.details : ""}`);
}

/**
 * Record a failing scenario.
 * @param {number} number - Scenario number (1-based)
 * @param {string} title  - Short scenario title
 * @param {object} [opts] - { subtitle, details, tags, assertions, screenshots }
 */
function fail(number, title, opts = {}) {
  const entry = { number, title, status: "FAIL", ...opts };
  results.push(entry);
  log(`❌ FAIL: S${number} — ${title}${opts.details ? " — " + opts.details : ""}`);
}

/**
 * Write canonical v1 test results JSON to /tmp/{FEATURE_NAME}-test-results.json.
 * Called automatically in the summary section and on fatal error for partial results.
 */
function writeResults() {
  const passed = results.filter((r) => r.status === "PASS").length;
  const failed = results.filter((r) => r.status === "FAIL").length;
  const output = {
    schemaVersion: 1,
    feature: FEATURE_NAME,
    timestamp: new Date().toISOString(),
    summary: { total: results.length, passed, failed },
    results,
    consoleErrors,
  };
  const filePath = path.join(SCREENSHOT_DIR, `${FEATURE_NAME}-test-results.json`);
  fs.writeFileSync(filePath, JSON.stringify(output, null, 2));
  log(`Results written to ${filePath}`);
}

/**
 * Recursive shadow DOM traversal to find elements by data-testid prefix.
 * Returns array of { testid, rect } objects.
 */
async function findTestIdElements(page, prefix) {
  return page.evaluate((pfx) => {
    function findAll(root, results = []) {
      for (const el of root.querySelectorAll("[data-testid]")) {
        const tid = el.getAttribute("data-testid");
        if (tid.startsWith(pfx)) {
          const rect = el.getBoundingClientRect();
          results.push({
            testid: tid,
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
          });
        }
      }
      for (const c of root.querySelectorAll("*")) {
        if (c.shadowRoot) findAll(c.shadowRoot, results);
      }
      return results;
    }
    return findAll(document);
  }, prefix);
}

/**
 * Parse time string "HH:MM:SS" to total seconds.
 */
function parseTime(timeStr) {
  const parts = timeStr.split(":").map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parts[0];
}

// ============================================================
// E2E-Proven Interaction Helpers
// Patterns sourced from Squirrel E2E fixtures:
//   - MediaPanelPage.ts (open panel, upload, drag, thumbnails)
//   - MainPage.ts (timeline, playback, clip selection)
// ============================================================

/**
 * Dismiss webpack dev server overlay that intercepts pointer events.
 * Call between scenarios and after timeline clear.
 * Ref: squirrel-quirks.md #16
 */
async function dismissOverlay(page) {
  await page.evaluate(() => {
    const overlay = document.getElementById("webpack-dev-server-client-overlay");
    if (overlay) overlay.remove();
  });
}

/**
 * Open the Media Panel (toggle-safe).
 * Checks disabled attribute, clicks, verifies panel opened.
 * If the panel was already open (toggle closed it), clicks again.
 * Pattern from: MediaPanelPage.ts openMediaPanel()
 */
async function openMediaPanel(page) {
  const mediaPanelButton = page.getByTestId(/^your-media-btn/);

  // Wait for button to be enabled (custom components use attribute, not property)
  await mediaPanelButton.waitFor({ state: "visible", timeout: 10000 });
  await page.waitForFunction(
    () => {
      function findInShadow(root) {
        for (const el of root.querySelectorAll("[data-testid]")) {
          if (/^your-media-btn/.test(el.getAttribute("data-testid"))) {
            return !el.hasAttribute("disabled");
          }
        }
        for (const c of root.querySelectorAll("*")) {
          if (c.shadowRoot) {
            const found = findInShadow(c.shadowRoot);
            if (found !== undefined) return found;
          }
        }
      }
      return findInShadow(document);
    },
    null,
    { timeout: 10000 }
  );

  await mediaPanelButton.click();
  // Race condition workaround (DVAWV-18567)
  await page.waitForTimeout(1000);

  // Handle toggle: check if panel content appeared
  const mediaThumbnails = page.locator("sq-media-thumbnail");
  try {
    await mediaThumbnails.first().waitFor({ state: "visible", timeout: 5000 });
  } catch {
    // Panel was already open — first click closed it. Click again.
    await mediaPanelButton.click();
    await page.waitForTimeout(1000);
    await mediaThumbnails.first().waitFor({ state: "visible", timeout: 10000 });
  }
  log("Media panel opened.");
}

/**
 * Upload files to the empty timeline dropzone (gapless track).
 * Accepts a single path or array of paths for multi-file upload.
 * Pattern from: MainPage.ts dropFiles2()
 */
async function uploadToGapless(page, filePaths) {
  const paths = Array.isArray(filePaths) ? filePaths : [filePaths];
  const fileInput = page
    .locator("sp-dropzone[data-testid='empty-timeline-dropzone'] input[type='file']");
  await fileInput.waitFor({ state: "attached", timeout: 30000 });
  await fileInput.first().setInputFiles(paths);
  // Wait for track items to appear
  await page
    .getByTestId(/^trackItem-gapless-/)
    .first()
    .waitFor({ state: "visible", timeout: 25000 });
  log(`Uploaded ${paths.length} file(s) to gapless track.`);
}

/**
 * Drop a file onto the timeline as a freeform track item.
 * Uses sp-dropzone-drop CustomEvent on sq-app-footer-content.
 * Pattern from: MediaPanelPage.ts mediaPaneldropFiles1()
 */
async function dropToFreeform(page, filePath) {
  const footerContent = page.locator("sq-app-footer-content");
  await footerContent.waitFor({ state: "visible", timeout: 10000 });

  const dataTransfer = await page.evaluateHandle(async (path) => {
    const response = await fetch(`file://${path}`);
    const buffer = await response.arrayBuffer();
    const name = path.split("/").pop();
    const dt = new DataTransfer();
    dt.items.add(new File([buffer], name, { type: "video/mp4" }));
    return dt;
  }, filePath);

  await footerContent.dispatchEvent("sp-dropzone-drop", { dataTransfer });
  // Wait for freeform track item to appear
  await page
    .getByTestId(/^trackItem-freeform-/)
    .first()
    .waitFor({ state: "visible", timeout: 15000 });
  log("Dropped file to freeform track.");
}

/**
 * Upload files via the Media Panel's upload dropzone.
 * Always uses .first() to avoid dual-input ambiguity (Quirk #1).
 * Pattern from: MediaPanelPage.ts mediaPanelDrag()
 */
async function uploadViaMediaPanel(page, filePaths) {
  const paths = Array.isArray(filePaths) ? filePaths : [filePaths];
  const panelInput = page
    .getByTestId("panel-dropzone")
    .locator('input[type="file"]')
    .first();
  await panelInput.waitFor({ state: "attached", timeout: 10000 });
  await panelInput.setInputFiles(paths);
  log(`Uploaded ${paths.length} file(s) via media panel.`);
}

/**
 * Select a track item by clicking it.
 * Uses focus-before-click pattern required for custom web components.
 * Filters out trim handle child elements to avoid mis-clicks.
 * Pattern from: MainPage.ts openVideoClipProperties()
 *
 * @param {Page} page - Playwright page
 * @param {string} testIdPattern - Regex pattern or prefix for data-testid (e.g. "trackItem-gapless-video-")
 * @param {number} [index=0] - Which matching element to select (0-based)
 */
async function selectClip(page, testIdPattern, index = 0) {
  const pattern = testIdPattern instanceof RegExp ? testIdPattern : new RegExp(`^${testIdPattern}`);
  const clips = page
    .getByTestId(pattern)
    .filter({ hasNot: page.getByTestId(/-trimHandle-left$/) })
    .filter({ hasNot: page.getByTestId(/-trimHandle-right$/) });

  const clip = clips.nth(index);
  await clip.waitFor({ state: "visible", timeout: 30000 });
  await clip.focus();
  await clip.click();
  log(`Selected clip matching ${pattern} at index ${index}.`);
  return clip;
}

/**
 * Position the playhead by clicking the timeline ruler at a given X coordinate.
 * Use bounding box of a clip or the ruler itself to compute the target X.
 * Ref: squirrel-quirks.md #7
 *
 * @param {Page} page - Playwright page
 * @param {number} targetX - Absolute X coordinate to click on the ruler
 */
async function clickRulerAtX(page, targetX) {
  const rulerBox = await page.locator("sq-timeline-ruler-view").boundingBox();
  if (!rulerBox) throw new Error("Timeline ruler not found");
  await page.mouse.click(targetX, rulerBox.y + rulerBox.height / 2);
  log(`Clicked ruler at X=${Math.round(targetX)}.`);
}

/**
 * Ensure keyboard focus is on the timeline area before pressing shortcuts.
 * Ref: squirrel-quirks.md #2
 */
async function focusTimeline(page) {
  await page.locator("sq-timeline-ruler-view").click();
}

/**
 * Read the playhead time text from the timeline.
 * Pattern from: MainPage.ts playheadTimeText()
 */
async function getPlayheadTime(page) {
  return page.locator(".playheadTime").textContent();
}

/**
 * Read the total timeline duration text.
 * Pattern from: MainPage.ts totalTimeText()
 */
async function getTotalTime(page) {
  return page.locator(".totalTime").textContent();
}

/**
 * Drag a media thumbnail from the Media Panel onto the timeline.
 * Waits for both source and target to be visible before dragging.
 * Pattern from: MediaPanelPage.ts dragVideoToTimelineFromMediaPanel()
 */
async function dragThumbnailToTimeline(page) {
  const thumbnail = page.locator("sq-media-thumbnail").first();
  const dropzone = page.getByTestId("empty-timeline-dropzone");
  await thumbnail.waitFor({ state: "visible", timeout: 15000 });
  await dropzone.waitFor({ state: "visible", timeout: 10000 });
  await thumbnail.dragTo(dropzone);
  log("Dragged thumbnail to timeline.");
}

// ============================================================
// Main Test
// ============================================================

(async () => {
  // Launch persistent context with Chrome
  // IMPORTANT: Runs in headless mode by default (no visible browser)
  // Set HEADED=true environment variable only for debugging
  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: process.env.HEADED !== "true", // Default: headless (no browser UI)
    channel: "chrome",
    ignoreHTTPSErrors: true,
    viewport: { width: 1920, height: 1080 },
    args: ["--ignore-certificate-errors"],
  });

  const page = context.pages()[0] || (await context.newPage());

  // Monitor console for errors
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => {
    consoleErrors.push(err.message);
  });

  try {
    // --------------------------------------------------------
    // Navigate and authenticate
    // --------------------------------------------------------
    log("Navigating to Squirrel...");
    await page.goto(TARGET_URL, {
      waitUntil: "domcontentloaded",
      timeout: 60000,
    });

    // Handle auth redirect if needed
    if (
      page.url().includes("auth-stg1.services.adobe.com") ||
      page.url().includes("adobelogin")
    ) {
      log("Authentication required. Logging in with test account...");

      if (!EMAIL || !PASSWORD) {
        throw new Error(
          "SQUIRREL_EMAIL and SQUIRREL_PASSWORD must be set. " +
            "Source ~/.config/squirrel/credentials.env before running."
        );
      }

      // Fill email
      await page.fill('input[name="username"]', EMAIL);
      await page.click('button:has-text("Continue")');
      await page.waitForSelector('input[type="password"]', { state: "visible", timeout: 10000 });

      // Fill password
      await page.fill('input[type="password"]', PASSWORD);
      await page.click('button:has-text("Continue")');

      // Wait for redirect back to Squirrel
      await page.waitForURL(`**/localhost.adobe.com:${DEV_PORT}/**`, {
        timeout: 30000,
      });
      log("Login complete! Waiting for app to load...");
    }

    // Wait for the app to fully initialize.
    // Check for app-ready indicators in priority order (fastest first).
    // The empty-timeline-dropzone appears when the app is ready with no project loaded.
    // document-title appears when a project is loaded.
    // Avoid waiting sequentially on multiple locators — check in parallel.
    log("Waiting for app to be ready...");
    try {
      await Promise.race([
        page.getByTestId("empty-timeline-dropzone").waitFor({ state: "visible", timeout: 30000 }),
        page.getByTestId("document-title").waitFor({ state: "visible", timeout: 30000 }),
        page.locator("sq-timeline-view").waitFor({ state: "visible", timeout: 30000 }),
      ]);
    } catch {
      log("App load indicators not found within 30s, continuing anyway...");
    }

    // Wait for the Media Panel button to become interactable (no "disabled" attribute).
    // This is the most reliable signal that the app is fully initialized and ready for action.
    // Pattern from E2E fixture: MediaPanelPage.ts openMediaPanel()
    try {
      const mediaPanelBtn = page.getByTestId(/^your-media-btn/);
      await mediaPanelBtn.waitFor({ state: "visible", timeout: 15000 });
      // Custom components use "disabled" attribute, not property — poll until removed
      await page.waitForFunction(
        () => {
          function findInShadow(root) {
            for (const el of root.querySelectorAll("[data-testid]")) {
              if (/^your-media-btn/.test(el.getAttribute("data-testid"))) {
                return !el.hasAttribute("disabled");
              }
            }
            for (const c of root.querySelectorAll("*")) {
              if (c.shadowRoot) {
                const found = findInShadow(c.shadowRoot);
                if (found !== undefined) return found;
              }
            }
          }
          return findInShadow(document);
        },
        null,
        { timeout: 15000 }
      );
      log("Media Panel button is interactable — app is ready.");
    } catch {
      log("Media Panel button not interactable within 15s, continuing anyway...");
    }
    log(`App loaded. URL: ${page.url()}`);

    // --------------------------------------------------------
    // YOUR TEST SCENARIOS GO HERE
    // --------------------------------------------------------

    // Example scenario structure:
    //
    // log("--- Scenario 1: [Description] ---");
    //
    // // Setup ...
    //
    // await page.screenshot({ path: `${SCREENSHOT_DIR}/verify-before-s1.png`, fullPage: true });
    //
    // // Perform action ...
    //
    // // Verify assertions
    // try {
    //     const box = await element.boundingBox();
    //     const widthOk = box.width < prevWidth;
    //     pass(1, "Short title", {
    //         subtitle: "Implementation detail",
    //         details: `Width ${prevWidth} -> ${box.width}`,
    //         tags: ["gapless"],
    //         assertions: [
    //             { label: "Width decreased", value: `${prevWidth} -> ${box.width} px`, passed: widthOk },
    //         ],
    //         screenshots: { before: "verify-before-s1.png", after: "verify-after-s1.png" },
    //     });
    // } catch (e) {
    //     fail(1, "Short title", {
    //         details: e.message,
    //         screenshots: { before: "verify-before-s1.png", after: "verify-fail-s1.png" },
    //     });
    //     await page.screenshot({ path: `${SCREENSHOT_DIR}/verify-fail-s1.png`, fullPage: true });
    // }
    //
    // await page.screenshot({ path: `${SCREENSHOT_DIR}/verify-after-s1.png`, fullPage: true });

    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------
    log("\n========== TEST RESULTS ==========");
    const passed = results.filter((r) => r.status === "PASS").length;
    const failed = results.filter((r) => r.status === "FAIL").length;
    log(`Total: ${results.length} | Passed: ${passed} | Failed: ${failed}`);

    results.forEach((r) => {
      log(
        `  ${r.status === "PASS" ? "✅" : "❌"} ${r.scenario}${
          r.details ? " — " + r.details : ""
        }`
      );
    });

    if (consoleErrors.length > 0) {
      log(`\n⚠️  ${consoleErrors.length} console error(s) detected:`);
      consoleErrors
        .slice(0, 10)
        .forEach((e) => log(`  - ${e.substring(0, 200)}`));
    }

    log("==================================\n");

    writeResults();

    // Final screenshot
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/verify-final.png`,
      fullPage: true,
    });
    log(`Final screenshot: ${SCREENSHOT_DIR}/verify-final.png`);
  } catch (error) {
    log(`💥 FATAL ERROR: ${error.message}`);
    writeResults(); // partial results so the evidence gate can inspect what ran
    await page
      .screenshot({
        path: `${SCREENSHOT_DIR}/verify-error.png`,
        fullPage: true,
      })
      .catch(() => {});
    throw error;
  } finally {
    await context.close();
  }
})();
