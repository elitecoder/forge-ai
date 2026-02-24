/**
 * Squirrel Upload Helpers
 *
 * Utility functions for uploading media to the Squirrel timeline
 * in Playwright visual tests. These implement the 5 upload methods
 * documented in upload-patterns.md.
 *
 * Usage: Copy the needed functions into your generated test script.
 * These are reference implementations, not meant to be imported directly.
 */

// ============================================================
// Method 1: Empty Timeline Upload
// ============================================================
// When the timeline is empty and you need to add the first clip.
// Creates a gapless (main) track.

async function emptyTimelineUpload(page, videoPath) {
  const dropzoneInput = page
    .getByTestId("empty-timeline-dropzone")
    .locator('input[type="file"]');

  await dropzoneInput.setInputFiles(videoPath);

  // Wait for the clip to appear on the timeline
  await page
    .getByTestId(/^trackItem-gapless-video-/)
    .first()
    .waitFor({ state: "visible", timeout: 30000 });
}

// ============================================================
// Method 2: Media Panel Upload (Upload Only, No Timeline)
// ============================================================
// Upload a file to the Media Panel without placing it on the timeline.

async function mediaPanelUpload(page, videoPath) {
  // Open Media Panel if not already open
  await page.getByTestId("your-media-btn").click();
  await page.waitForTimeout(1000); // Known race condition (TODO: DVAWV-18567)

  // IMPORTANT: Media Panel has 2 file inputs. Always use .first()
  const panelInput = page
    .getByTestId("panel-dropzone")
    .locator('input[type="file"]')
    .first();

  await panelInput.setInputFiles(videoPath);

  // Wait for thumbnail to appear
  await page.waitForTimeout(3000);
}

// ============================================================
// Method 3: Drop Files on Timeline (Creates Freeform Track)
// ============================================================
// When the timeline already has content and you want to add an
// overlapping clip. Creates a freeform track.
//
// This dispatches a `sp-dropzone-drop` CustomEvent on sq-app-footer-content
// with a DataTransfer containing the file data.

async function dropFilesOnTimeline(page, videoPath) {
  const fs = require("fs");

  // Read the file and convert to hex
  const fileBuffer = fs.readFileSync(videoPath);
  const hexData = fileBuffer.toString("hex");
  const fileName = require("path").basename(videoPath);

  // Get the footer element handle
  const footerHandle = await page
    .locator("sq-app-footer-content")
    .elementHandle();

  // Dispatch sp-dropzone-drop CustomEvent
  await footerHandle.evaluate(
    (el, fileData) => {
      const dt = new DataTransfer();
      const buffer = Uint8Array.from(
        fileData.hex.match(/.{1,2}/g).map((b) => parseInt(b, 16))
      );
      dt.items.add(new File([buffer], fileData.name, { type: "video/mp4" }));
      el.dispatchEvent(
        new CustomEvent("sp-dropzone-drop", {
          detail: { dataTransfer: dt, clientX: 100 },
          bubbles: true,
          cancelable: true,
        })
      );
    },
    { name: fileName, hex: hexData }
  );

  // Wait for the freeform clip to appear
  await page
    .getByTestId(/^trackItem-freeform-video-/)
    .first()
    .waitFor({ state: "visible", timeout: 30000 });
}

// ============================================================
// Method 4: Drag from Media Panel to Timeline
// ============================================================
// Drag a file from the Media Panel onto the timeline.
// Creates a gapless track if timeline is empty.

async function dragFromMediaPanelToTimeline(page) {
  // This requires the file to already be in the Media Panel
  // (use mediaPanelUpload first)

  const thumbnail = page.locator(".media-thumbnail").first();
  const timeline = page.locator("sq-app-footer-content");

  await thumbnail.dragTo(timeline);

  await page
    .getByTestId(/^trackItem-gapless-video-/)
    .first()
    .waitFor({ state: "visible", timeout: 30000 });
}

// ============================================================
// Method 5: Position Playhead at Clip Position
// ============================================================
// Click the ruler at a specific X position relative to a clip.
// Used to position the playhead before trim operations.

async function positionPlayheadAtClip(page, clip, positionRatio = 0.5) {
  const clipBox = await clip.boundingBox();
  if (!clipBox) throw new Error("Clip not found or not visible");

  // Find the ruler element
  const ruler = page.locator("sq-timeline-ruler-view");
  const rulerBox = await ruler.boundingBox();
  if (!rulerBox) throw new Error("Ruler not found or not visible");

  // Calculate click position
  const targetX = clipBox.x + clipBox.width * positionRatio;
  const targetY = rulerBox.y + rulerBox.height / 2;

  // Click the ruler to position the playhead
  await page.mouse.click(targetX, targetY);

  // Small wait for playhead to move
  await page.waitForTimeout(500);
}

// ============================================================
// Helper: Wait for Clip Width Change
// ============================================================

async function waitForClipWidthChange(
  clip,
  beforeWidth,
  direction,
  timeout = 5000
) {
  const { expect } = require("@playwright/test");

  await expect(async () => {
    const box = await clip.boundingBox();
    expect(box).toBeTruthy();
    if (direction === "decrease") {
      expect(box.width).toBeLessThan(beforeWidth - 5);
    } else {
      expect(box.width).toBeGreaterThan(beforeWidth + 5);
    }
  }).toPass({ timeout });
}

// ============================================================
// Helper: Get Timeline Duration
// ============================================================

async function getTimelineDuration(page) {
  return page.locator(".totalTime").textContent();
}

module.exports = {
  emptyTimelineUpload,
  mediaPanelUpload,
  dropFilesOnTimeline,
  dragFromMediaPanelToTimeline,
  positionPlayheadAtClip,
  waitForClipWidthChange,
  getTimelineDuration,
};
