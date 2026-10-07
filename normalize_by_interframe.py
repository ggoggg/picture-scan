#!/usr/bin/env python3
"""Normalize Super 8 frames by trying inter-frame lines before perforation detection."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detect_white_rectangle import (
    CropBox,
    annotation_path_for_image,
    crop_and_normalize,
    crop_box_from_perforation,
    detect_perforation,
    fallback_output_path_for_image,
    fallback_normalize,
    image_paths_from_folder,
    imwrite_jpeg,
    load_image,
    output_path_for_image,
    parse_output_size,
    parse_search_area,
    process_image as process_by_perforation,
    repair_perforation_box,
    repair_visible_perforation_left_edge,
    validate_jpeg_quality,
)


@dataclass
class InterFrameLine:
    y: int
    y2: int
    score: float
    dark_fraction: float
    mean_brightness: float
    source: str = "row_scan"

    @property
    def center_y(self) -> float:
        return self.y + (self.y2 - self.y) / 2.0

    @property
    def thickness(self) -> int:
        return self.y2 - self.y + 1

    def to_log_dict(self) -> dict[str, int | float]:
        return {
            "y": self.y,
            "y2": self.y2,
            "center_y": round(self.center_y, 2),
            "thickness": self.thickness,
            "score": round(self.score, 4),
            "dark_fraction": round(self.dark_fraction, 4),
            "mean_brightness": round(self.mean_brightness, 2),
            "source": self.source,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try inter-frame horizontal lines first, then fall back to perforation detection."
    )
    parser.add_argument("input", type=Path, help="Input JPEG image path or folder of JPEG images.")
    parser.add_argument("--output", type=Path, help="Output file or folder.")
    parser.add_argument("--log", type=Path, default=Path("interframe_positions.json"))
    parser.add_argument("--annotate", type=Path, help="Optional debug image or folder.")
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--name-regex", default=r"^frame_\d+\.jpe?g$")
    parser.add_argument("--output-size", default="1440x1080")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--color-mode", choices=("auto", "color", "grayscale"), default="auto")
    parser.add_argument("--artifact-threshold", type=float, default=0.12)
    parser.add_argument("--mirror-horizontal", action="store_true")
    parser.add_argument("--frame-aspect", type=float, default=4 / 3)
    parser.add_argument("--frame-height-perf-ratio", type=float, default=3.5)
    parser.add_argument("--crop-scale", type=float, default=1.08)
    parser.add_argument("--line-crop-scale", type=float, default=1.0)
    parser.add_argument("--line-frame-height-ratio", type=float, default=0.875)
    parser.add_argument("--inter-frame-gap-pixels", type=int, default=8)
    parser.add_argument("--line-search-x", default="5%:92%")
    parser.add_argument("--line-dark-threshold", type=int, default=45)
    parser.add_argument("--line-min-dark-fraction", type=float, default=0.45)
    parser.add_argument("--line-max-mean-brightness", type=float, default=95.0)
    parser.add_argument("--line-min-score", type=float, default=0.01)
    parser.add_argument("--line-min-thickness", type=int, default=5)
    parser.add_argument("--line-max-thickness", type=int, default=180)
    parser.add_argument("--line-band-height", type=int, default=32)
    parser.add_argument("--line-band-step", type=int, default=2)
    parser.add_argument("--line-band-min-score", type=float, default=0.35)
    parser.add_argument("--line-band-bw-threshold", type=int, default=0, help="0 uses Otsu threshold for black/white band scoring.")
    parser.add_argument("--line-band-bw-weight", type=float, default=0.55)
    parser.add_argument("--line-upper-search-percent", type=float, default=13.0)
    parser.add_argument("--line-lower-search-percent", type=float, default=13.0)
    parser.add_argument("--line-smooth-rows", type=int, default=15)
    parser.add_argument("--line-edge-smooth-rows", type=int, default=9)
    parser.add_argument("--line-edge-percentile", type=float, default=96.0)
    parser.add_argument("--line-edge-min-strength", type=float, default=1.5)
    parser.add_argument("--line-anomaly-min-score", type=float, default=0.60)
    parser.add_argument("--line-uniformity-percentile", type=float, default=65.0)
    parser.add_argument("--line-lower-band-min-dark-fraction", type=float, default=0.80)
    parser.add_argument("--line-lower-band-search-start-ratio", type=float, default=0.85)
    parser.add_argument("--line-split-ratio", type=float, default=0.50)
    parser.add_argument("--line-min-distance-ratio", type=float, default=0.35)
    parser.add_argument("--line-horizontal-source", choices=("perforation", "center"), default="perforation")
    parser.add_argument("--threshold", type=int, default=250)
    parser.add_argument("--max-chroma", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--perforation-search-x", default="85%:")
    parser.add_argument("--perforation-search-y", default="20%:85%")
    parser.add_argument("--right-gap-perf-ratio", type=float, default=0.0)
    parser.add_argument("--center-y-offset-perf-ratio", type=float, default=0.0)
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--max-deskew-degrees", type=float, default=8.0)
    parser.add_argument("--perforation-height-width-ratio", type=float, default=2.25)
    parser.add_argument("--disable-perforation-repair", action="store_true")
    return parser.parse_args()


def smooth_rows(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window)
    if window == 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def line_search_bounds(image_width: int, image_height: int, line_search_x: str) -> tuple[int, int]:
    area = parse_search_area(line_search_x, None)
    if area is None:
        return 0, image_width
    x1, x2, _, _ = area.bounds(image_width, image_height)
    return x1, x2


def line_from_group(
    start: int,
    end: int,
    score: np.ndarray,
    dark_smooth: np.ndarray,
    mean_smooth: np.ndarray,
) -> InterFrameLine:
    group_slice = slice(start, end + 1)
    return InterFrameLine(
        y=start,
        y2=end,
        score=float(score[group_slice].max()),
        dark_fraction=float(dark_smooth[group_slice].max()),
        mean_brightness=float(mean_smooth[group_slice].min()),
    )


def collect_line_groups(
    active_rows: np.ndarray,
    score: np.ndarray,
    dark_smooth: np.ndarray,
    mean_smooth: np.ndarray,
    min_thickness: int,
    max_thickness: int,
) -> list[InterFrameLine]:
    lines: list[InterFrameLine] = []
    start: int | None = None
    image_height = len(active_rows)
    for row, active in enumerate(active_rows):
        if active and start is None:
            start = row
        if (not active or row == image_height - 1) and start is not None:
            end = row if active and row == image_height - 1 else row - 1
            thickness = end - start + 1
            if min_thickness <= thickness <= max_thickness:
                lines.append(line_from_group(start, end, score, dark_smooth, mean_smooth))
            start = None
    return lines


def pick_split_line(lines: list[InterFrameLine], upper_half: bool, split_y: float, image_height: int) -> InterFrameLine | None:
    if upper_half:
        candidates = [line for line in lines if line.center_y < split_y]
        if not candidates:
            return None
        return min(candidates, key=lambda line: (line.y, -line.score))

    candidates = [line for line in lines if line.center_y >= split_y]
    if not candidates:
        return None
    return max(candidates, key=lambda line: (line.y2, line.score))


def line_vertical_windows(image_height: int, args: argparse.Namespace) -> tuple[int, int]:
    upper_height = round(image_height * max(0.0, args.line_upper_search_percent) / 100.0)
    lower_height = round(image_height * max(0.0, args.line_lower_search_percent) / 100.0)
    upper_limit = min(image_height, upper_height)
    lower_start = max(0, image_height - lower_height)
    return upper_limit, lower_start


def pick_windowed_lines(lines: list[InterFrameLine], image_height: int, args: argparse.Namespace) -> list[InterFrameLine]:
    upper_limit, lower_start = line_vertical_windows(image_height, args)
    upper_candidates = [line for line in lines if line.center_y <= upper_limit]
    lower_candidates = [line for line in lines if line.center_y >= lower_start]
    upper_line = max(upper_candidates, key=lambda line: line.score) if upper_candidates else None
    lower_line = max(lower_candidates, key=lambda line: line.score) if lower_candidates else None
    selected = [line for line in (upper_line, lower_line) if line is not None]
    if len(selected) == 2:
        min_distance = round(image_height * args.line_min_distance_ratio)
        if abs(selected[1].center_y - selected[0].center_y) < min_distance:
            return []
    return sorted(selected, key=lambda line: line.center_y)


def detect_interframe_lines_by_bands(image: np.ndarray, args: argparse.Namespace) -> list[InterFrameLine]:
    image_height, image_width = image.shape[:2]
    band_height = max(args.line_min_thickness, min(args.line_band_height, image_height))
    step = max(1, args.line_band_step)
    x1, x2 = line_search_bounds(image_width, image_height, args.line_search_x)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[:, x1:x2]
    luminance = lab[:, :, 0]
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, x1:x2]
    if args.line_band_bw_threshold > 0:
        _, binary = cv2.threshold(gray, args.line_band_bw_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    black_rows = (binary == 0).mean(axis=1)

    row_luminance = luminance.mean(axis=1)
    row_chroma = chroma.mean(axis=1)
    csum_luminance = np.concatenate(([0.0], np.cumsum(row_luminance, dtype=np.float64)))
    csum_chroma = np.concatenate(([0.0], np.cumsum(row_chroma, dtype=np.float64)))
    csum_black = np.concatenate(([0.0], np.cumsum(black_rows, dtype=np.float64)))
    global_luminance = float(row_luminance.mean())
    global_chroma = float(row_chroma.mean())
    luminance_scale = max(float(row_luminance.std()), 1.0)
    chroma_scale = max(float(row_chroma.std()), 1.0)
    bw_weight = min(max(float(args.line_band_bw_weight), 0.0), 1.0)

    candidates: list[InterFrameLine] = []
    for y in range(0, image_height - band_height + 1, step):
        y2 = y + band_height - 1
        band_luminance = float((csum_luminance[y + band_height] - csum_luminance[y]) / band_height)
        band_chroma = float((csum_chroma[y + band_height] - csum_chroma[y]) / band_height)
        band_black = float((csum_black[y + band_height] - csum_black[y]) / band_height)
        darkness = max(0.0, (global_luminance - band_luminance) / luminance_scale)
        low_chroma = max(0.0, (global_chroma - band_chroma) / chroma_scale)
        tone_score = 0.75 * darkness + 0.25 * low_chroma
        score = bw_weight * band_black + (1.0 - bw_weight) * tone_score
        if score >= args.line_band_min_score:
            candidates.append(
                InterFrameLine(
                    y=y,
                    y2=y2,
                    score=score,
                    dark_fraction=band_black,
                    mean_brightness=band_luminance,
                    source="band_scan",
                )
            )

    selected = pick_windowed_lines(candidates, image_height, args)
    if len(selected) < 2:
        return []
    return selected


def row_anomaly_score(image: np.ndarray, x1: int, x2: int, args: argparse.Namespace) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[:, x1:x2]
    luminance = lab[:, :, 0]
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)

    row_luminance = smooth_rows(luminance.mean(axis=1).astype(np.float32), args.line_smooth_rows)
    row_chroma = smooth_rows(chroma.mean(axis=1).astype(np.float32), args.line_smooth_rows)
    row_texture = smooth_rows(luminance.std(axis=1).astype(np.float32), args.line_smooth_rows)

    luminance_delta = np.clip(
        np.abs(row_luminance - float(row_luminance.mean())) / max(float(row_luminance.std()), 1.0),
        0.0,
        3.0,
    ) / 3.0
    chroma_delta = np.clip(
        np.abs(row_chroma - float(row_chroma.mean())) / max(float(row_chroma.std()), 1.0),
        0.0,
        3.0,
    ) / 3.0
    texture_reference = max(float(np.percentile(row_texture, args.line_uniformity_percentile)), 1.0)
    uniformity = np.clip((texture_reference - row_texture) / texture_reference, 0.0, 1.0)

    return 0.45 * luminance_delta + 0.35 * chroma_delta + 0.20 * uniformity


def detect_lower_anomaly_band(
    anomaly_score: np.ndarray,
    dark_smooth: np.ndarray,
    mean_smooth: np.ndarray,
    image_height: int,
    args: argparse.Namespace,
) -> InterFrameLine | None:
    search_start = round(image_height * args.line_lower_band_search_start_ratio)
    active_rows = (anomaly_score >= args.line_anomaly_min_score) & (
        dark_smooth >= args.line_lower_band_min_dark_fraction
    )
    active_rows[:search_start] = False
    bands = collect_line_groups(
        active_rows,
        anomaly_score,
        dark_smooth,
        mean_smooth,
        args.line_min_thickness,
        args.line_max_thickness,
    )
    if not bands:
        return None
    return max(bands, key=lambda line: (line.score, line.y2))


def detect_interframe_lines_by_rows(image: np.ndarray, args: argparse.Namespace) -> list[InterFrameLine]:
    image_height, image_width = image.shape[:2]
    x1, x2 = line_search_bounds(image_width, image_height, args.line_search_x)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    roi = gray[:, x1:x2]

    dark_fraction = (roi <= args.line_dark_threshold).mean(axis=1)
    mean_brightness = roi.mean(axis=1)
    dark_smooth = smooth_rows(dark_fraction.astype(np.float32), args.line_smooth_rows)
    mean_smooth = smooth_rows(mean_brightness.astype(np.float32), args.line_smooth_rows)
    score = dark_smooth * np.clip((args.line_max_mean_brightness - mean_smooth) / args.line_max_mean_brightness, 0, 1)
    is_line_row = (dark_smooth >= args.line_min_dark_fraction) & (mean_smooth <= args.line_max_mean_brightness)
    anomaly_score = row_anomaly_score(image, x1, x2, args)

    lines = collect_line_groups(
        is_line_row,
        score,
        dark_smooth,
        mean_smooth,
        args.line_min_thickness,
        args.line_max_thickness,
    )

    row_gradient = np.abs(np.diff(roi.astype(np.float32), axis=0)).mean(axis=1)
    edge_smooth = smooth_rows(row_gradient.astype(np.float32), args.line_edge_smooth_rows)
    edge_smooth = np.pad(edge_smooth, (0, 1), mode="edge")
    edge_threshold = max(args.line_edge_min_strength, float(np.percentile(edge_smooth, args.line_edge_percentile)))
    edge_score = (edge_smooth / max(edge_threshold, 1.0)) * 0.1
    is_edge_row = edge_smooth >= edge_threshold
    lines.extend(
        collect_line_groups(
            is_edge_row,
            edge_score,
            dark_smooth,
            mean_smooth,
            args.line_min_thickness,
            args.line_max_thickness,
        )
    )
    lines = [line for line in lines if line.score >= args.line_min_score]

    selected = pick_windowed_lines(lines, image_height, args)
    lower_anomaly_band = detect_lower_anomaly_band(anomaly_score, dark_smooth, mean_smooth, image_height, args)
    if lower_anomaly_band is not None:
        upper_line = selected[0] if selected and selected[0].center_y < image_height / 2 else None
        selected = [line for line in (upper_line, lower_anomaly_band) if line is not None]
    return sorted(selected, key=lambda item: item.center_y)


def detect_interframe_lines(image: np.ndarray, args: argparse.Namespace) -> list[InterFrameLine]:
    band_lines = detect_interframe_lines_by_bands(image, args)
    if len(band_lines) >= 2:
        return band_lines
    return detect_interframe_lines_by_rows(image, args)


def expected_line_crop_height(image_height: int, args: argparse.Namespace) -> int:
    return max(1, round(image_height * args.line_frame_height_ratio * args.line_crop_scale))


def line_crop_y(lines: list[InterFrameLine], image_height: int, args: argparse.Namespace) -> tuple[int, int, str]:
    gap = args.inter_frame_gap_pixels
    if len(lines) >= 2:
        top_line, bottom_line = lines[0], lines[-1]
        y = top_line.y2 + 1 + gap
        bottom = bottom_line.y - 1 - gap
        height = max(1, bottom - y + 1)
        if args.line_crop_scale != 1.0:
            center_y = y + (height - 1) / 2.0
            height = max(1, round(height * args.line_crop_scale))
            y = round(center_y - height / 2.0)
        method = "interframe_band_scan" if all(line.source == "band_scan" for line in lines) else "interframe_lines"
        return y, height, method

    line = lines[0]
    height = expected_line_crop_height(image_height, args)
    if line.center_y < image_height / 2:
        y = line.y2 + 1 + gap
    else:
        y = line.y - gap - height
    return y, height, "interframe_line_guess"


def detect_perforation_for_horizontal_crop(image: np.ndarray, args: argparse.Namespace):
    search_area = parse_search_area(args.perforation_search_x, args.perforation_search_y)
    perforation, candidate_count = detect_perforation(
        image,
        args.threshold,
        args.max_chroma,
        args.min_area,
        search_area=search_area,
    )
    if perforation is None:
        return None, candidate_count
    repair_enabled = not args.disable_perforation_repair
    perforation = repair_perforation_box(
        perforation,
        image.shape[0],
        args.perforation_height_width_ratio,
        repair_enabled,
    )
    perforation = repair_visible_perforation_left_edge(image, perforation)
    return perforation, candidate_count


def crop_from_lines(
    image: np.ndarray,
    lines: list[InterFrameLine],
    args: argparse.Namespace,
) -> tuple[CropBox, str, object | None, int]:
    image_height, image_width = image.shape[:2]
    y, height, method = line_crop_y(lines, image_height, args)
    width = min(image_width, max(1, round(height * args.frame_aspect)))
    perforation = None
    candidate_count = 0
    if args.line_horizontal_source == "perforation":
        perforation, candidate_count = detect_perforation_for_horizontal_crop(image, args)
    if perforation is not None:
        x = perforation.x - width
        if x < 0:
            width = max(1, perforation.x)
            x = 0
    else:
        x = round((image_width - width) / 2.0)
    x = min(max(0, x), image_width - width)
    return CropBox(x=x, y=y, width=width, height=height), method, perforation, candidate_count


def draw_dashed_horizontal_line(
    image: np.ndarray,
    y: int,
    color: tuple[int, int, int],
    thickness: int = 5,
    dash_length: int = 54,
    gap_length: int = 22,
) -> None:
    y = min(max(0, y), image.shape[0] - 1)
    x = 0
    while x < image.shape[1]:
        x2 = min(image.shape[1] - 1, x + dash_length)
        cv2.line(image, (x, y), (x2, y), color, thickness)
        x += dash_length + gap_length


def annotate_lines(
    image: np.ndarray,
    output_path: Path,
    crop_box: CropBox,
    lines: list[InterFrameLine],
    perforation,
    method: str,
    args: argparse.Namespace,
    jpeg_quality: int,
) -> None:
    annotated = image.copy()
    crop_y = max(0, crop_box.y)
    crop_y2 = min(image.shape[0] - 1, crop_box.y2)
    upper_limit, lower_start = line_vertical_windows(image.shape[0], args)
    draw_dashed_horizontal_line(annotated, upper_limit, (255, 0, 255), 5)
    draw_dashed_horizontal_line(annotated, lower_start, (255, 0, 255), 5)
    for line in lines:
        cv2.rectangle(annotated, (0, line.y), (image.shape[1] - 1, line.y2), (255, 0, 0), 7)
    cv2.rectangle(annotated, (crop_box.x, crop_y), (crop_box.x2, crop_y2), (0, 255, 0), 7)
    cv2.line(annotated, (crop_box.x, crop_y), (crop_box.x2, crop_y), (0, 255, 255), 5)
    cv2.line(annotated, (crop_box.x, crop_y2), (crop_box.x2, crop_y2), (0, 255, 255), 5)
    if perforation is not None:
        cv2.rectangle(annotated, (perforation.x, perforation.y), (perforation.x2, perforation.y2), (0, 0, 255), 7)
    cv2.putText(annotated, method, (24, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 4, cv2.LINE_AA)
    imwrite_jpeg(output_path, annotated, jpeg_quality)


def process_image(
    image_path: Path,
    output_path: Path,
    annotation_path: Path | None,
    output_size: tuple[int, int],
    args: argparse.Namespace,
    next_image_path: Path | None = None,
) -> dict[str, object]:
    image = load_image(image_path)
    lines = detect_interframe_lines(image, args)
    if lines:
        crop_box, method, perforation, candidate_count = crop_from_lines(image, lines, args)
        stitch_image = load_image(next_image_path) if next_image_path is not None else None
        color_mode, artifact_fraction, stitch = crop_and_normalize(
            image,
            image_path,
            crop_box,
            output_path,
            output_size,
            args.color_mode,
            args.artifact_threshold,
            args.mirror_horizontal,
            args.jpeg_quality,
            stitch_image,
            next_image_path,
            args.inter_frame_gap_pixels,
        )
        result = {
            "image": str(image_path),
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "method": method,
            "interframe_lines": [line.to_log_dict() for line in lines],
            "perforation_for_horizontal_crop": perforation.to_log_dict() if perforation is not None else None,
            "crop": crop_box.to_log_dict(),
            "vertical_stitch": stitch.to_log_dict(),
            "vertical_stitch_pixels": stitch.pixels,
            "deskew_degrees": 0.0,
            "normalized_output": str(output_path),
            "normalized_width": output_size[0],
            "normalized_height": output_size[1],
            "jpeg_quality": args.jpeg_quality,
            "color_mode": color_mode,
            "mirror_horizontal": args.mirror_horizontal,
            "cyan_artifact_fraction": round(artifact_fraction, 4),
            "candidate_count": candidate_count,
            "status": "ok",
        }
        if annotation_path is not None:
            annotate_lines(image, annotation_path, crop_box, lines, perforation, method, args, args.jpeg_quality)
            result["annotation_output"] = str(annotation_path)
        return result

    result = process_by_perforation(
        image_path,
        output_path,
        annotation_path,
        output_size,
        args,
        next_image_path=next_image_path,
    )
    result["method"] = "perforation_fallback"
    result["status"] = "ok"
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output_size = parse_output_size(args.output_size)
        args.jpeg_quality = validate_jpeg_quality(args.jpeg_quality)
        parse_search_area(args.perforation_search_x, args.perforation_search_y)
        parse_search_area(args.line_search_x, None)
    except ValueError as error:
        logging.error("%s", error)
        return 1

    input_is_folder = args.input.is_dir()
    if input_is_folder:
        image_paths = image_paths_from_folder(args.input, args.glob, args.name_regex)
        output = args.output or Path("out-interframe")
    else:
        image_paths = [args.input]
        output = args.output or Path("normalized_frame.jpg")

    if not image_paths:
        logging.error("No JPEG images found in %s", args.input)
        return 1

    results: list[dict[str, object]] = []
    failures = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_path_for_image(image_path, output, input_is_folder)
        annotation_path = annotation_path_for_image(image_path, args.annotate, input_is_folder)
        next_image_path = image_paths[index] if input_is_folder and index < len(image_paths) else None
        try:
            result = process_image(image_path, output_path, annotation_path, output_size, args, next_image_path)
        except ValueError as error:
            fallback_output_path = fallback_output_path_for_image(image_path, output, input_is_folder)
            result = fallback_normalize(image_path, fallback_output_path, output_size, args, str(error))
            result["method"] = "fallback_full_image"
            failures += 1
            logging.warning("[%d/%d] %s -> %s, fallback full-image resize: %s", index, len(image_paths), image_path, fallback_output_path, error)
        else:
            logging.info("[%d/%d] %s -> %s, %s", index, len(image_paths), image_path, output_path, result["method"])
        results.append(result)

    log_data: dict[str, object]
    if input_is_folder:
        log_data = {
            "input": str(args.input),
            "output": str(output),
            "count": len(image_paths),
            "succeeded": len(image_paths) - failures,
            "failed": failures,
            "frames": results,
        }
    else:
        log_data = results[0]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log_data, indent=2) + "\n", encoding="utf-8")
    logging.info("wrote log: %s", args.log)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
