#!/usr/bin/env python3
"""Detect the film gate and batch crop to a fixed 3302 x 2425 pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def detect_crop(image: np.ndarray, threshold: float | None = None,
                inset: int = 8) -> tuple[int, int, int, int]:
    """Return exclusive (left, top, right, bottom) coordinates at original size.

    Scan a closed foreground component from all four sides. Edge percentiles
    reject isolated glow and scratches and move inside the rounded gate corners.
    This assumes a mostly rectangular, illuminated gate on a near-black surround.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    small = cv2.resize(image, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is None:
        # A low threshold preserves dark picture content; Otsu would split scenes.
        threshold = max(8.0, float(np.percentile(gray, 95)) * 0.12)
    mask = (gray > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No illuminated frame found; use --reference for dark frames')
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if cv2.contourArea(contour) < gray.size * 0.15 or w < gray.shape[1] * .3 or h < gray.shape[0] * .3:
        raise ValueError('Detected region is too small; use --reference or lower --threshold')
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    region = filled[y:y+h, x:x+w] > 0
    # Ignore the outermost 3% of rows/columns where rounded corners dominate.
    dy, dx = max(1, round(h * .03)), max(1, round(w * .03))
    rows = region[dy:h-dy]
    cols = region[:, dx:w-dx]
    left = x + np.percentile(rows.argmax(axis=1), 95)
    right = x + w - np.percentile(rows[:, ::-1].argmax(axis=1), 95)
    top = y + np.percentile(cols.argmax(axis=0), 95)
    bottom = y + h - np.percentile(cols[::-1].argmax(axis=0), 95)
    sx, sy = width / gray.shape[1], height / gray.shape[0]
    box = (int(np.ceil(left * sx)) + inset, int(np.ceil(top * sy)) + inset,
           int(np.floor(right * sx)) - inset, int(np.floor(bottom * sy)) - inset)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Inset leaves an empty crop')
    return box


def fixed_crop(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Center a 3302 x 2425 crop on the detected gate, keeping it in bounds."""
    crop_width, crop_height = 3302, 2425
    height, width = shape[:2]
    if width < crop_width or height < crop_height:
        raise ValueError(f'Image {width}x{height} is smaller than required crop 3302x2425')
    left, top, right, bottom = box
    x = max(0, min(width - crop_width, (left + right - crop_width) // 2))
    y = max(0, min(height - crop_height, (top + bottom - crop_height) // 2))
    return x, y, x + crop_width, y + crop_height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Folder containing input images')
    parser.add_argument('--output', type=Path, help='Default: INPUT/cropped')
    parser.add_argument('--reference', type=Path, help='Detect once on this image and reuse the crop; requires matching dimensions')
    parser.add_argument('--threshold', type=float, help='Foreground brightness threshold, 0–254 (default: automatic)')
    parser.add_argument('--inset', type=int, default=8, help='Detection inset in pixels; final crop stays 3302x2425 (default: 8)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality, 1–100 (default: 95)')
    parser.add_argument('--mirror-horizontal', action='store_true', help='Flip the cropped image left to right')
    parser.add_argument('--recursive', action='store_true', help='Include subfolders and preserve their layout')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output files')
    parser.add_argument('--dry-run', action='store_true', help='Report crop coordinates without writing files')
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error('Input folder does not exist')
    if args.inset < 0 or not 1 <= args.quality <= 100:
        parser.error('Inset must be nonnegative and quality must be 1–100')
    if args.threshold is not None and not 0 <= args.threshold <= 254:
        parser.error('Threshold must be 0–254')
    source = args.folder.resolve()
    output = (args.output or source / 'cropped').resolve()
    if output == source or source.is_relative_to(output):
        parser.error('Output must not equal or contain the input folder')
    paths = sorted(p for p in (source.rglob('*') if args.recursive else source.iterdir())
                   if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.resolve().is_relative_to(output))
    if not paths:
        parser.error('No supported images found')
    reference_shape = reference_box = None
    if args.reference:
        try:
            reference = read_image(args.reference)
            reference_shape = reference.shape[:2]
            reference_box = fixed_crop(detect_crop(reference, args.threshold, args.inset), reference.shape)
        except ValueError as error:
            parser.error(str(error))
        print(f'Reference crop: {reference_box}')
    completed = skipped = failed = 0
    for path in paths:
        destination = output / path.relative_to(source)
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            image = read_image(path)
            if reference_shape is not None and image.shape[:2] != reference_shape:
                raise ValueError('Dimensions differ from reference image')
            left, top, right, bottom = reference_box or fixed_crop(
                detect_crop(image, args.threshold, args.inset), image.shape)
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if path.suffix.lower() in {'.jpg', '.jpeg'} else []
                cropped = image[top:bottom, left:right]
                if args.mirror_horizontal:
                    cropped = cv2.flip(cropped, 1)
                if not cv2.imwrite(str(destination), cropped, params):
                    raise ValueError(f'Cannot write {destination}')
            completed += 1
            print(json.dumps({'file': str(path.relative_to(source)), 'crop': [left, top, right, bottom],
                              'size': [right-left, bottom-top]}))
        except (ValueError, OSError, cv2.error) as error:
            failed += 1
            print(f'ERROR {path.name}: {error}')
    print(f'{"Detected" if args.dry_run else "Cropped"}: {completed}; skipped existing: {skipped}; failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
