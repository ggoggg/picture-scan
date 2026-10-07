#!/usr/bin/env python3
"""Normalize frames with a fixed reference crop size and ordered fallbacks."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from detect_white_rectangle import (
    CropBox,
    annotation_path_for_image,
    annotate_crop_and_perforation,
    crop_and_normalize,
    crop_box_from_perforation,
    fallback_output_path_for_image,
    fallback_normalize,
    image_paths_from_folder,
    load_image,
    output_path_for_image,
    parse_output_size,
    parse_search_area,
    validate_jpeg_quality,
)
from normalize_by_interframe import (
    annotate_lines,
    crop_from_lines,
    detect_interframe_lines_by_bands,
    detect_interframe_lines_by_rows,
    detect_perforation_for_horizontal_crop,
)


@dataclass
class Candidate:
    method: str
    crop: CropBox
    lines: list[Any]
    perforation: Any | None
    candidate_count: int


def parse_reference_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("reference crop size must look like WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("reference crop width and height must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try band-scan, row interframe, then perforation, accepting only crops near a reference size."
    )
    parser.add_argument("input", type=Path, help="Input JPEG image path or folder of JPEG images.")
    parser.add_argument("--reference-crop-size", type=parse_reference_size, required=True, help="Reference crop size, for example 3592x2694.")
    parser.add_argument("--reference-tolerance-pixels", type=int, default=40)
    parser.add_argument("--output", type=Path, help="Output file or folder.")
    parser.add_argument("--log", type=Path, default=Path("reference_crop_positions.json"))
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


def crop_matches_reference(crop: CropBox, reference_size: tuple[int, int], tolerance: int) -> tuple[bool, dict[str, Any]]:
    reference_width, reference_height = reference_size
    width_delta = crop.width - reference_width
    height_delta = crop.height - reference_height
    matches = abs(width_delta) <= tolerance and abs(height_delta) <= tolerance
    return matches, {
        "reference_width": reference_width,
        "reference_height": reference_height,
        "width_delta": width_delta,
        "height_delta": height_delta,
        "tolerance_pixels": tolerance,
        "matches": matches,
    }


def force_reference_size(crop: CropBox, image_width: int, reference_size: tuple[int, int]) -> CropBox:
    reference_width, reference_height = reference_size
    if reference_width > image_width:
        raise ValueError(f"Reference crop width {reference_width} is wider than image width {image_width}")
    x = min(max(0, crop.x), image_width - reference_width)
    return CropBox(x=x, y=crop.y, width=reference_width, height=reference_height)


def line_candidate(image: Any, args: argparse.Namespace, method: str) -> Candidate | None:
    if method == "interframe_band_scan":
        lines = detect_interframe_lines_by_bands(image, args)
    elif method == "interframe_lines":
        lines = detect_interframe_lines_by_rows(image, args)
    else:
        raise ValueError(f"Unknown line method {method}")
    if len(lines) < 2:
        return None
    crop, detected_method, perforation, candidate_count = crop_from_lines(image, lines, args)
    return Candidate(method=method if method == "interframe_band_scan" else detected_method, crop=crop, lines=lines, perforation=perforation, candidate_count=candidate_count)


def perforation_candidate(image: Any, args: argparse.Namespace) -> Candidate | None:
    perforation, candidate_count = detect_perforation_for_horizontal_crop(image, args)
    if perforation is None:
        return None
    crop = crop_box_from_perforation(
        perforation,
        image_width=image.shape[1],
        image_height=image.shape[0],
        frame_aspect=args.frame_aspect,
        frame_height_perf_ratio=args.frame_height_perf_ratio,
        crop_scale=args.crop_scale,
        right_gap_perf_ratio=args.right_gap_perf_ratio,
        center_y_offset_perf_ratio=args.center_y_offset_perf_ratio,
    )
    return Candidate("perforation_fallback", crop, [], perforation, candidate_count)


def candidate_log(candidate: Candidate | None, validation: dict[str, Any] | None, reason: str | None = None) -> dict[str, Any]:
    if candidate is None:
        return {"status": "not_found", "reason": reason}
    data = {
        "method": candidate.method,
        "crop": candidate.crop.to_log_dict(),
        "validation": validation,
        "candidate_count": candidate.candidate_count,
    }
    if candidate.lines:
        data["interframe_lines"] = [line.to_log_dict() for line in candidate.lines]
    if candidate.perforation is not None:
        data["perforation"] = candidate.perforation.to_log_dict()
    if reason is not None:
        data["reason"] = reason
    return data


def choose_candidate(image: Any, args: argparse.Namespace) -> tuple[Candidate | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for method in ("interframe_band_scan", "interframe_lines"):
        candidate = line_candidate(image, args, method)
        if candidate is None:
            attempts.append(candidate_log(None, None, f"{method} did not find two lines"))
            continue
        matches, validation = crop_matches_reference(candidate.crop, args.reference_crop_size, args.reference_tolerance_pixels)
        attempts.append(candidate_log(candidate, validation, None if matches else "crop size differs from reference"))
        if matches:
            return candidate, attempts

    candidate = perforation_candidate(image, args)
    if candidate is None:
        attempts.append(candidate_log(None, None, "perforation_fallback did not find perforation"))
        return None, attempts
    matches, validation = crop_matches_reference(candidate.crop, args.reference_crop_size, args.reference_tolerance_pixels)
    attempts.append(candidate_log(candidate, validation, None if matches else "crop size differs from reference"))
    if matches:
        return candidate, attempts
    return None, attempts


def process_image(
    image_path: Path,
    output_path: Path,
    fallback_output_path: Path,
    annotation_path: Path | None,
    output_size: tuple[int, int],
    args: argparse.Namespace,
    next_image_path: Path | None = None,
) -> dict[str, Any]:
    image = load_image(image_path)
    candidate, attempts = choose_candidate(image, args)
    if candidate is None:
        result = fallback_normalize(image_path, fallback_output_path, output_size, args, "No candidate matched reference crop size")
        result["method"] = "fallback_full_image"
        result["status"] = "needs_manual"
        result["attempts"] = attempts
        return result

    crop = force_reference_size(candidate.crop, image.shape[1], args.reference_crop_size)
    stitch_image = load_image(next_image_path) if next_image_path is not None else None
    color_mode, artifact_fraction, stitch = crop_and_normalize(
        image,
        image_path,
        crop,
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
        "method": candidate.method,
        "reference_crop_width": args.reference_crop_size[0],
        "reference_crop_height": args.reference_crop_size[1],
        "detected_crop": candidate.crop.to_log_dict(),
        "crop": crop.to_log_dict(),
        "interframe_lines": [line.to_log_dict() for line in candidate.lines],
        "perforation": candidate.perforation.to_log_dict() if candidate.perforation is not None else None,
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
        "candidate_count": candidate.candidate_count,
        "status": "ok",
        "attempts": attempts,
    }
    if annotation_path is not None:
        if candidate.lines:
            annotate_lines(image, annotation_path, crop, candidate.lines, candidate.perforation, candidate.method, args, args.jpeg_quality)
        elif candidate.perforation is not None:
            annotate_crop_and_perforation(image, annotation_path, candidate.perforation, crop, 0.0, args.jpeg_quality)
        result["annotation_output"] = str(annotation_path)
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
        output = args.output or Path("out-reference-crop")
    else:
        image_paths = [args.input]
        output = args.output or Path("normalized_frame.jpg")

    if not image_paths:
        logging.error("No JPEG images found in %s", args.input)
        return 1

    results: list[dict[str, Any]] = []
    failures = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_path_for_image(image_path, output, input_is_folder)
        annotation_path = annotation_path_for_image(image_path, args.annotate, input_is_folder)
        next_image_path = image_paths[index] if input_is_folder and index < len(image_paths) else None
        fallback_output_path = fallback_output_path_for_image(image_path, output, input_is_folder)
        result = process_image(image_path, output_path, fallback_output_path, annotation_path, output_size, args, next_image_path)
        logged_output_path = result.get("normalized_output", str(output_path))
        if result["status"] != "ok":
            failures += 1
            logging.warning("[%d/%d] %s -> %s, %s", index, len(image_paths), image_path, logged_output_path, result["method"])
        else:
            logging.info("[%d/%d] %s -> %s, %s", index, len(image_paths), image_path, logged_output_path, result["method"])
        results.append(result)

    log_data: dict[str, Any]
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
