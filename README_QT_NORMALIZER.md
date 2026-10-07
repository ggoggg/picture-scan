# Super 8 Qt Normalizer

Run the desktop UI from the project folder:

```bash
env/bin/python qt_normalizer.py
```

Workflow:

1. Open one JPEG or a folder of `frame_*.jpg` images.
2. Choose the output folder.
3. Use `Auto Current` for one frame, or `Start` to normalize the whole list.
4. During a whole-list run, use `Pause`, `Resume`, or `Stop`.
5. Frames marked `MANUAL` need a manual crop.
6. Draw a crop rectangle on the source preview, or edit `Crop x/y/width/height`.
7. Use `Save Manual` to write `normalized_<frame>.jpg`.

`Y offset %` moves automatic crop placement vertically relative to the detected perforation. Positive values move the crop down.

`Search x1/x2/y1/y2` limits automatic perforation detection to the yellow rectangle in the source preview. The default is the right-side middle band where the Super 8 perforation usually appears. Enable `Edit search` to draw, move, or resize that yellow search rectangle with its edge/corner handles; edit the spinboxes for exact pixel values.

The UI writes `positions.json` in the output folder and uses the same crop, stitch, color, mirror, output-size, and JPEG-quality logic as `detect_white_rectangle.py`.

Use `Save Settings` to write current UI options to `qt_normalizer_settings.json`. The Qt app reloads that file on the next start, and also saves settings when closing.

Alternative line-first normalizer:

```bash
env/bin/python normalize_by_interframe.py /path/to/images --output out-interframe --annotate out-interframe/debug
```

It first tries horizontal inter-frame lines, uses one detected line to guess frame height when needed, and falls back to perforation detection when no line is found. The log `method` is `interframe_lines`, `interframe_line_guess`, or `perforation_fallback`.
