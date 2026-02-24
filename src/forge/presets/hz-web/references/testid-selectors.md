# Squirrel data-testid Selectors

Complete map of data-testid selectors available in the Squirrel UI. Use these when specifying which elements to target in visual test scenarios.

## App Structure

| Selector          | Component              | Notes                           |
| ----------------- | ---------------------- | ------------------------------- |
| `canvas-inputid`  | Canvas input surface   | Main video preview area         |
| `app-left-panel`  | Left expandable panel  | Media, text, transitions panels |
| `app-right-panel` | Right expandable panel | Properties, generation settings |
| `playhead`        | Playhead               | Blue vertical line on timeline  |
| `ruler-tick`      | Timeline ruler ticks   | Time markers                    |

## Top Toolbar (App Header)

| Selector              | Component               |
| --------------------- | ----------------------- |
| `navigation-menu-btn` | Hamburger menu          |
| `project-name`        | Project name (editable) |
| `document-title`      | Timeline name           |
| `undo-btn`            | Undo                    |
| `redo-btn`            | Redo                    |
| `share-btn`           | Share button            |
| `story-builder-btn`   | Quick cut               |

## Left Rail Buttons

| Selector                 | Component                  |
| ------------------------ | -------------------------- |
| `your-media-btn`         | Your media panel toggle    |
| `generated-media-btn`    | Generation history toggle  |
| `text-left-panel-btn`    | Text panel toggle          |
| `video-transitions-btn`  | Transitions panel toggle   |
| `debug-panel-btn`        | Debug panel toggle         |
| `keyboard-shortcuts-btn` | Keyboard shortcuts overlay |

## Left Panel Content

| Selector                  | Component                                                 |
| ------------------------- | --------------------------------------------------------- |
| `add-text-button`         | "Add text" button (text panel)                            |
| `text-left-panel-header`  | Text panel header                                         |
| `panel-dropzone`          | Media Panel dropzone (has 2 file inputs — use `.first()`) |
| `empty-timeline-dropzone` | Empty timeline upload area                                |

## Right Rail Buttons

| Selector         | Component                  |
| ---------------- | -------------------------- |
| `properties-btn` | Properties panel toggle    |
| `gen-ai-btn`     | Generation settings toggle |
| `transcript-btn` | Text-based editing toggle  |
| `comment-btn`    | Comments toggle            |

## Right Panel Content

| Selector                     | Component             |
| ---------------------------- | --------------------- |
| `aspect-ratio-picker-picker` | Aspect ratio dropdown |
| `frame-rate-picker`          | Frame rate dropdown   |

## Transport / Middle Bar

| Selector                         | Component                    |
| -------------------------------- | ---------------------------- |
| `selection-btn`                  | Select tool                  |
| `split-btn`                      | Split tool                   |
| `slip-btn`                       | Slip tool                    |
| `crop-btn`                       | Crop tool                    |
| `play-btn`                       | Play/Pause                   |
| `step-backward-btn`              | Step backward                |
| `step-forward-btn`               | Step forward                 |
| `unmute-btn`                     | Mute/Unmute                  |
| `middle-bar-aspect-ratio-picker` | Aspect ratio (in middle bar) |
| `zoom-in-btn`                    | Zoom in                      |
| `zoom-out-btn`                   | Zoom out                     |
| `timeline-zoom-slider`           | Zoom slider                  |
| `skimmer-btn`                    | Skimmer toggle               |
| `snapping-btn`                   | Snap mode toggle             |
| `timeline-options`               | Timeline options             |

## Menu Items

| Selector                   | Action         | Shortcut |
| -------------------------- | -------------- | -------- |
| `add-blank-clip-menu-item` | Add blank clip | Y        |
| `add-text-menu-item`       | Add text       | T        |
| `split-menu-item`          | Split clip     | S        |
| `export-menu-item`         | Export         | Shift+E  |

## Timeline Elements (Custom Elements, Not data-testid)

These are custom element tag names, not data-testid attributes. Use `page.locator("tag-name")`:

| Element Tag              | Component                      |
| ------------------------ | ------------------------------ |
| `sq-timeline-container`  | Timeline container             |
| `sq-timeline-view`       | Timeline view                  |
| `sq-track-view`          | Individual track (video/audio) |
| `sq-clip-view`           | Clip on a track                |
| `sq-track-item-view`     | Track item content             |
| `sq-playhead-view`       | Playhead component             |
| `sq-skimmer-view`        | Skimmer component              |
| `sq-timeline-ruler-view` | Timeline ruler                 |
| `sq-marquee-select-view` | Marquee selection              |
| `sq-app-footer-content`  | Footer containing timeline     |

## Timeline Track Items (Dynamic data-testid)

Track items have dynamic testids with this pattern:

```
trackItem-{trackType}-{mediaType}-{id}
```

### Track Type Prefixes

| Prefix     | Track Type               |
| ---------- | ------------------------ |
| `gapless`  | Main/gapless track       |
| `freeform` | Freeform (overlay) track |

### Media Type Suffixes

| Suffix     | Media Type                    |
| ---------- | ----------------------------- |
| `video`    | Video clip                    |
| `audio`    | Audio clip                    |
| `bitmap`   | Image clip                    |
| `blank`    | Blank clip (gapless only)     |
| `graphics` | Graphics clip (freeform only) |

### Locator Patterns

```javascript
// Single gapless video clip
page.getByTestId(/^trackItem-gapless-video-/);

// All freeform video clips
page.getByTestId(/^trackItem-freeform-video-/);

// Specific clip by full ID
page.getByTestId("trackItem-gapless-video-abc123");

// Any track item
page.locator("[data-testid^='trackItem-']");
```

### Trim Handles

| Selector Pattern        | Component         |
| ----------------------- | ----------------- |
| `trimHandle-left-{id}`  | Left trim handle  |
| `trimHandle-right-{id}` | Right trim handle |

```javascript
// Left trim handle of first gapless video
page.getByTestId(/^trimHandle-left-/).first();
```

## Component Hierarchy

```
squirrel-app
└── sq-app-shell-container
    └── sq-app-frame
        ├── sq-app-header-view          (top toolbar)
        ├── sq-app-left-rail-view       (left icon buttons)
        ├── sq-app-left-panel           (left expandable panel)
        ├── sq-app-program-view         (canvas area)
        │   └── hz-canvas-surface > hz-canvas-input
        ├── sq-app-right-rail-view      (right icon buttons)
        ├── sq-app-right-panel          (right expandable panel)
        ├── sq-middle-bar-container     (transport controls)
        └── sq-app-footer-content       (timeline area)
            └── sq-timeline-container
                └── sq-timeline-view
                    ├── sq-playhead-view
                    ├── sq-timeline-ruler-view
                    └── sq-track-view
                        └── sq-clip-view
                            └── sq-track-item-view
```

## Shadow DOM Considerations

Most Squirrel components use Shadow DOM. Standard `querySelector` won't find elements inside shadow roots. For Playwright scripts, use recursive shadow DOM traversal:

```javascript
const items = await page.evaluate(() => {
  function findAll(root, results = []) {
    for (const el of root.querySelectorAll("[data-testid]")) {
      if (el.getAttribute("data-testid").startsWith("trackItem-"))
        results.push({
          testid: el.getAttribute("data-testid"),
          rect: el.getBoundingClientRect(),
        });
    }
    for (const c of root.querySelectorAll("*"))
      if (c.shadowRoot) findAll(c.shadowRoot, results);
    return results;
  }
  return findAll(document);
});
```

However, Playwright's `getByTestId()` and `locator()` methods automatically pierce shadow DOM, so prefer those when possible.

## Timeline Duration Selector

The total timeline duration is displayed in an element with class `.totalTime`:

```javascript
const duration = await page.locator(".totalTime").textContent();
// Returns format: "00:MM:SS" (or "HH:MM:SS" for longer timelines)
```
