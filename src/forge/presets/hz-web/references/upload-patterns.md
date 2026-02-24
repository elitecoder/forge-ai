# Upload Patterns for Visual Tests

Five methods for getting media onto the Squirrel timeline. Choose the right one based on the scenario.

## Method 1: Empty Timeline Upload

**When:** Timeline is empty (first clip)
**Creates:** Gapless (main) track

```javascript
const dropzoneInput = page
  .getByTestId("empty-timeline-dropzone")
  .locator('input[type="file"]');

await dropzoneInput.setInputFiles(videoPath);

// Wait for clip to appear
await page
  .getByTestId(/^trackItem-gapless-video-/)
  .first()
  .waitFor({ state: "visible", timeout: 30000 });
```

## Method 2: Media Panel Upload (No Timeline Placement)

**When:** Need file in Media Panel but not on timeline yet
**Creates:** Nothing on timeline (file appears in Media Panel)

```javascript
// Open Media Panel
await page.getByTestId("your-media-btn").click();
await page.waitForTimeout(1000); // Known race condition

// IMPORTANT: Media Panel has 2 file inputs — always use .first()
const panelInput = page
  .getByTestId("panel-dropzone")
  .locator('input[type="file"]')
  .first();

await panelInput.setInputFiles(videoPath);
await page.waitForTimeout(3000); // Wait for processing
```

## Method 3: Drop Files on Timeline (sp-dropzone-drop)

**When:** Timeline already has content; need to add overlapping clip
**Creates:** Freeform track

This is the most reliable method for adding a second clip. It dispatches a `sp-dropzone-drop` CustomEvent on `sq-app-footer-content`.

```javascript
const fs = require("fs");

// Read file and convert to hex
const fileBuffer = fs.readFileSync(videoPath);
const hexData = fileBuffer.toString("hex");
const fileName = require("path").basename(videoPath);

// Get footer element handle
const footerHandle = await page
  .locator("sq-app-footer-content")
  .elementHandle();

// Dispatch drop event
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

// Wait for freeform clip
await page
  .getByTestId(/^trackItem-freeform-video-/)
  .first()
  .waitFor({ state: "visible", timeout: 30000 });
```

## Method 4: Drag from Media Panel to Timeline

**When:** File is already in Media Panel; drag to empty timeline
**Creates:** Gapless track (if timeline is empty)

```javascript
const thumbnail = page.locator(".media-thumbnail").first();
const timeline = page.locator("sq-app-footer-content");
await thumbnail.dragTo(timeline);
```

## Method 5: Direct Upload with Retries (versacDrag)

**When:** Need reliable upload to empty timeline with retry logic
**Creates:** Gapless track

This pattern is from the E2E test fixtures and includes retry logic for flaky uploads. For standalone Playwright scripts, Method 1 (emptyTimelineUpload) is preferred.

## Choosing the Right Method

| I Need To...                               | Use Method                     |
| ------------------------------------------ | ------------------------------ |
| Add first video to empty timeline          | Method 1 (emptyTimelineUpload) |
| Add a second video (creates overlay track) | Method 3 (dropFilesOnTimeline) |
| Upload to Media Panel for later use        | Method 2 (mediaPanelUpload)    |
| Drag from Media Panel to timeline          | Method 4 (dragFromMediaPanel)  |
| Upload with retry logic                    | Method 5 (versacDrag)          |

## Common Gotchas

1. **Media Panel has 2 file inputs** — Always use `.first()` when selecting the file input in the panel dropzone
2. **`sp-dropzone-drop` requires hex-encoded file data** — Read the file as a buffer and convert to hex string
3. **Wait for clip to appear** — Always wait for the `trackItem-*` testid element to be visible before proceeding
4. **Audio goes to freeform** — Audio file imports always create freeform tracks, even on empty timeline
5. **File size limits** — Very large files may cause timeout. Use small test videos when possible.

## Test Video Files

For standalone Playwright scripts (not using E2E framework), use any `.mp4` file available on the system. Common locations:

```bash
# Check for existing test videos
find /tmp -name "*.mp4" 2>/dev/null
find ~/Downloads -name "*.mp4" 2>/dev/null

# Download a small test video if needed
curl -o /tmp/test-video.mp4 "https://www.w3schools.com/html/mov_bbb.mp4"
```

If using the E2E framework, assets are available from `@hz/honeydew-core`:

- `Assets.MP4.sampleMp4Video`
- `Assets.MP4.smallGlobeMp4Video`
- `Assets.MP3.sample7sAudio`
- `Assets.JPG.image24`
