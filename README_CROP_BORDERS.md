# Crop black borders from projector frames

Uses the existing `numpy` and `opencv-python` dependencies in `requirements.txt`.
Run from this project directory:

```bash
env/bin/python crop_frame_borders.py /path/to/frames --output /path/to/cropped
```

For a stationary camera/projector, use a well-exposed reference frame to keep
identical crop coordinates and output dimensions throughout the sequence:

```bash
env/bin/python crop_frame_borders.py \
  /home/user/proj/film2/projector-scan/frames-20260927-001428 \
  --output ./cropped-frames \
  --reference /home/user/proj/film2/projector-scan/frames-20260927-001428/frame_000784.jpeg
```

Every output crop is exactly **3302 x 2425 pixels**, without resizing.
Without `--reference`, the crop is centered on the frame detected in each image.
With `--reference`, the crop position is fixed too. Crops are shifted inward if
needed to stay within the source image; sources smaller than the crop are rejected.
A fixed-size crop may include some black border if the illuminated frame is smaller. Detection assumes one large, roughly rectangular illuminated
frame surrounded by black. Dark scenes touching the gate edge can confuse it;
use a reference for such sequences. This crops an axis-aligned rectangle and
does not correct perspective or rotation.

Options:

- `--inset 8`: detection inset; the final crop remains 3302 x 2425 pixels.
- `--threshold 18`: manually set the 8-bit brightness cutoff; default is automatic.
- `--dry-run`: print coordinates without saving images.
- `--recursive`: include subfolders, preserving their relative paths.
- `--quality 95`: JPEG output quality.
- `--mirror-horizontal`: flip each cropped image left to right.
- `--overwrite`: replace existing output files; by default they are skipped.

The default output is a `cropped` subfolder. Source files are retained. Supported
extensions: JPEG, PNG, TIFF, BMP, WebP. Images are decoded as 8-bit color and
re-encoded; metadata, transparency, and higher bit depth are not preserved.
Unreadable images and failed detections are reported and skipped, with a nonzero
exit status when any image fails. Reference images must match input dimensions.
