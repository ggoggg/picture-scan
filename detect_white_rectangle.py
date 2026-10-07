#!/usr/bin/env python3
"""Detect a Super 8 perforation, deskew the image, and crop the frame."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Component:
    x: int
    y: int
    width: int
    height: int
    area: int
    fill_ratio: float
    angle_degrees: float = 0.0
    repaired: bool = False

    @property
    def x2(self) -> int:
        return self.x + self.width - 1

    @property
    def y2(self) -> int:
        return self.y + self.height - 1

    @property
    def center_x(self) -> float:
        return self.x + (self.width - 1) / 2

    @property
    def center_y(self) -> float:
        return self.y + (self.height - 1) / 2

    def to_log_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        data.update(
            {
                "x2": self.x2,
                "y2": self.y2,
                "center_x": round(self.center_x, 2),
                "center_y": round(self.center_y, 2),
                "fill_ratio": round(self.fill_ratio, 4),
                "angle_degrees": round(self.angle_degrees, 4),
                "repaired": self.repaired,
            }
        )
        return data


@dataclass
class CropBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width - 1

    @property
    def y2(self) -> int:
        return self.y + self.height - 1

    def to_log_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass
class CropStitch:
    pixels: int = 0
    source_image: str | None = None
    source_y: int | None = None
    target_edge: str | None = None

    def to_log_dict(self) -> dict[str, int | str] | None:
        if self.pixels == 0:
            return None
        return {
            "pixels": self.pixels,
            "source_image": self.source_image or "",
            "source_y": self.source_y or 0,
            "target_edge": self.target_edge or "",
        }


@dataclass
class SearchArea:
    x1: int | float = 0
    x2: int | float | None = None
    y1: int | float = 0
    y2: int | float | None = None

    @staticmethod
    def resolve_bound(value: int | float | None, upper: int, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, float):
            return round(upper * value)
        return value

    def bounds(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x1 = self.resolve_bound(self.x1, image_width, 0)
        x2 = self.resolve_bound(self.x2, image_width, image_width)
        y1 = self.resolve_bound(self.y1, image_height, 0)
        y2 = self.resolve_bound(self.y2, image_height, image_height)
        x1 = min(max(0, x1), image_width - 1)
        y1 = min(max(0, y1), image_height - 1)
        x2 = min(max(x1 + 1, x2), image_width)
        y2 = min(max(y1 + 1, y2), image_height)
        return x1, x2, y1, y2

    def to_log_dict(self, image_width: int, image_height: int) -> dict[str, int]:
        x1, x2, y1, y2 = self.bounds(image_width, image_height)
        return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a Super 8 perforation, deskew the image, and crop the frame."
    )
    parser.add_argument("input", type=Path, help="Input JPEG image path or folder of JPEG images.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Minimum RGB channel value for a pixel to count as white. Default: 250.",
    )
    parser.add_argument(
        "--max-chroma",
        type=int,
        default=10,
        help="Maximum RGB channel spread for a pixel to count as neutral white. Default: 10.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Ignore white components smaller than this many pixels. Default: 1000.",
    )
    parser.add_argument(
        "--perforation-search-x",
        default="85%:",
        help="Limit perforation detection to this horizontal range, as START:END pixels or percentages. Default: 85%%:.",
    )
    parser.add_argument(
        "--perforation-search-y",
        default="20%:85%",
        help="Limit perforation detection to this vertical range, as START:END pixels or percentages. Default: 20%%:85%%.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("rectangle_position.json"),
        help="Where to write detection/crop positions as JSON.",
    )
    parser.add_argument(
        "--annotate",
        type=Path,
        help="Optional debug image path. In folder mode this is treated as a debug output folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save output. Use a file path for one image, or a folder path for folder mode. Default: normalized_frame.jpg for one image, out/ for a folder.",
    )
    parser.add_argument(
        "--glob",
        default="*.jpg",
        help="Filename glob for folder mode. Default: *.jpg.",
    )
    parser.add_argument(
        "--name-regex",
        default=r"^frame_\d+\.jpe?g$",
        help=r"Filename regex for folder mode after --glob filtering. Default: ^frame_\d+\.jpe?g$.",
    )
    parser.add_argument(
        "--output-size",
        default="1440x1080",
        help="Normalized output size as WIDTHxHEIGHT. Default: 1440x1080.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG output quality from 1 to 100. Default: 100.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("auto", "color", "grayscale"),
        default="auto",
        help="Output color handling. auto removes strong cyan/blue artifacts by converting affected frames to grayscale. Default: auto.",
    )
    parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=0.12,
        help="In auto color mode, grayscale the frame when this fraction is cyan/blue contaminated. Default: 0.12.",
    )
    parser.add_argument(
        "--mirror-horizontal",
        action="store_true",
        help="Mirror the normalized output image horizontally.",
    )
    parser.add_argument(
        "--frame-aspect",
        type=float,
        default=4 / 3,
        help="Picture frame aspect ratio, width divided by height. Default: 1.3333.",
    )
    parser.add_argument(
        "--frame-height-perf-ratio",
        type=float,
        default=3.5,
        help="Crop height relative to perforation height. Default: 3.5.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.08,
        help="Expand the computed crop around its center while preserving aspect ratio. Default: 1.08.",
    )
    parser.add_argument(
        "--inter-frame-gap-pixels",
        type=int,
        default=8,
        help="Rows to skip past the inter-frame line before taking a stitched vertical piece. Default: 8.",
    )
    parser.add_argument(
        "--right-gap-perf-ratio",
        type=float,
        default=0.0,
        help="Deprecated. Horizontal crop now ends exactly at the perforation left edge.",
    )
    parser.add_argument(
        "--center-y-offset-perf-ratio",
        type=float,
        default=0.0,
        help="Vertical crop center offset from perforation center, relative to perf height. Positive moves crop down. Default: 0.0.",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable rotation alignment before cropping.",
    )
    parser.add_argument(
        "--max-deskew-degrees",
        type=float,
        default=8.0,
        help="Do not rotate if detected skew is larger than this. Default: 8.",
    )
    parser.add_argument(
        "--perforation-height-width-ratio",
        type=float,
        default=2.25,
        help="Expected perforation height divided by width, used to repair partly obscured perforations. Default: 2.25.",
    )
    parser.add_argument(
        "--disable-perforation-repair",
        action="store_true",
        help="Do not extend short perforation detections caused by color artifacts.",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def white_mask(image: np.ndarray, threshold: int, max_chroma: int) -> np.ndarray:
    channel_min = image.min(axis=2)
    channel_max = image.max(axis=2)
    mask = (channel_min >= threshold) & ((channel_max - channel_min) <= max_chroma)
    return mask.astype(np.uint8) * 255


def contour_long_edge_angle(contour: np.ndarray) -> float:
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    longest_angle = 0.0
    longest_length = 0.0

    for index in range(4):
        start = box[index]
        end = box[(index + 1) % 4]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = float(np.hypot(dx, dy))
        if length > longest_length:
            longest_length = length
            longest_angle = float(np.degrees(np.arctan2(dy, dx)))

    while longest_angle <= -90.0:
        longest_angle += 180.0
    while longest_angle > 90.0:
        longest_angle -= 180.0

    return longest_angle


def deskew_degrees_from_vertical(angle_degrees: float) -> float:
    correction = angle_degrees - 90.0
    while correction <= -90.0:
        correction += 180.0
    while correction > 90.0:
        correction -= 180.0
    return correction


def rotate_bound(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def find_white_components(mask: np.ndarray, min_area: int) -> list[Component]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[Component] = []

    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width == 0 or height == 0:
            continue

        fill_ratio = area / float(width * height)
        angle_degrees = contour_long_edge_angle(contour)
        components.append(
            Component(
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                fill_ratio=fill_ratio,
                angle_degrees=angle_degrees,
            )
        )

    return components


def choose_perforation(
    components: list[Component],
    image_width: int,
    image_height: int,
    size_image_width: int | None = None,
    size_image_height: int | None = None,
) -> Component | None:
    if not components:
        return None

    size_width = image_width if size_image_width is None else size_image_width
    size_height = image_height if size_image_height is None else size_image_height
    likely_perforations: list[Component] = []
    for component in components:
        aspect = component.height / float(component.width)
        touches_right_edge = component.x2 >= image_width * 0.97
        reasonable_size = (
            size_width * 0.025 <= component.width <= size_width * 0.20
            and size_height * 0.05 <= component.height <= size_height * 0.50
        )
        perforation_shape = 0.65 <= aspect <= 6.0
        if touches_right_edge and reasonable_size and perforation_shape:
            likely_perforations.append(component)

    if not likely_perforations:
        return None

    return max(
        likely_perforations,
        key=lambda component: (
            component.x2,
            component.area,
            component.fill_ratio,
        ),
    )


def merge_fragment_group(group: list[Component]) -> Component:
    x1 = min(component.x for component in group)
    y1 = min(component.y for component in group)
    x2 = max(component.x2 for component in group)
    y2 = max(component.y2 for component in group)
    area = sum(component.area for component in group)
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    fill_ratio = area / float(width * height)
    angle_degrees = max(group, key=lambda component: component.area).angle_degrees
    return Component(
        x=x1,
        y=y1,
        width=width,
        height=height,
        area=area,
        fill_ratio=fill_ratio,
        angle_degrees=angle_degrees,
        repaired=True,
    )


def add_merged_perforation_fragments(
    components: list[Component],
    image_width: int,
    image_height: int,
    size_image_width: int,
    size_image_height: int,
) -> list[Component]:
    right_edge_fragments = [
        component
        for component in components
        if component.x2 >= image_width * 0.94
        and size_image_width * 0.02 <= component.width <= size_image_width * 0.20
        and size_image_height * 0.03 <= component.height <= size_image_height * 0.30
    ]
    if len(right_edge_fragments) < 2:
        return components

    merged_components: list[Component] = []
    used_indexes: set[int] = set()
    max_vertical_gap = round(size_image_height * 0.04)

    for index, component in enumerate(right_edge_fragments):
        if index in used_indexes:
            continue
        group = [component]
        used_indexes.add(index)
        changed = True
        while changed:
            changed = False
            group_x1 = min(item.x for item in group)
            group_x2 = max(item.x2 for item in group)
            group_y1 = min(item.y for item in group)
            group_y2 = max(item.y2 for item in group)
            for other_index, other in enumerate(right_edge_fragments):
                if other_index in used_indexes:
                    continue
                horizontal_overlap = min(group_x2, other.x2) - max(group_x1, other.x) + 1
                min_width = min(group_x2 - group_x1 + 1, other.width)
                vertical_gap = max(other.y - group_y2 - 1, group_y1 - other.y2 - 1, 0)
                if horizontal_overlap >= min_width * 0.45 and vertical_gap <= max_vertical_gap:
                    group.append(other)
                    used_indexes.add(other_index)
                    changed = True
        if len(group) >= 2:
            merged_components.append(merge_fragment_group(group))

    if not merged_components:
        return components
    return components + merged_components


def repair_perforation_box(
    perforation: Component,
    image_height: int,
    expected_height_width_ratio: float,
    enabled: bool,
) -> Component:
    if not enabled:
        return perforation

    expected_height = round(perforation.width * expected_height_width_ratio)
    detected_ratio = perforation.height / float(perforation.width)
    requested_deskew = abs(deskew_degrees_from_vertical(perforation.angle_degrees))
    if perforation.repaired and perforation.height < image_height * 0.22 and detected_ratio >= 2.0:
        expected_height = max(expected_height, round(image_height * 0.24))
        if expected_height > perforation.height:
            repaired_y = max(0, perforation.y - (expected_height - perforation.height))
            repaired_height = min(expected_height, image_height - repaired_y)
            return Component(
                x=perforation.x,
                y=repaired_y,
                width=perforation.width,
                height=repaired_height,
                area=perforation.area,
                fill_ratio=perforation.fill_ratio,
                angle_degrees=perforation.angle_degrees,
                repaired=True,
            )

    if detected_ratio >= 3.0 and perforation.height < image_height * 0.22 and requested_deskew > 2.0:
        expected_height = max(expected_height, round(image_height * 0.24))
        if expected_height > perforation.height:
            repaired_y = max(0, perforation.y - (expected_height - perforation.height))
            repaired_height = min(expected_height, image_height - repaired_y)
            return Component(
                x=perforation.x,
                y=repaired_y,
                width=perforation.width,
                height=repaired_height,
                area=perforation.area,
                fill_ratio=perforation.fill_ratio,
                angle_degrees=perforation.angle_degrees,
                repaired=True,
            )

    if expected_height <= perforation.height:
        return perforation

    if detected_ratio < 1.2:
        return perforation

    if detected_ratio >= 1.75:
        return perforation

    repaired_height = min(expected_height, image_height - perforation.y)
    return Component(
        x=perforation.x,
        y=perforation.y,
        width=perforation.width,
        height=repaired_height,
        area=perforation.area,
        fill_ratio=perforation.fill_ratio,
        angle_degrees=perforation.angle_degrees,
        repaired=True,
    )


def repair_visible_perforation_left_edge(image: np.ndarray, perforation: Component) -> Component:
    if perforation.repaired:
        return perforation

    image_height, image_width = image.shape[:2]
    scan_margin = round(perforation.width * 0.8)
    x1 = max(0, perforation.x - scan_margin)
    x2 = min(image_width, perforation.x2 + 1)
    y1 = max(0, perforation.y)
    y2 = min(image_height, perforation.y2 + 1)
    if x2 <= x1 or y2 <= y1:
        return perforation

    roi = image[y1:y2, x1:x2]
    bright_mask = roi.max(axis=2) >= 140
    column_fraction = bright_mask.mean(axis=0)
    bright_columns = np.where(column_fraction >= 0.05)[0]
    if len(bright_columns) == 0:
        return perforation

    visible_left = x1 + int(bright_columns[0])
    shift = perforation.x - visible_left
    if shift <= max(24, round(perforation.width * 0.30)):
        return perforation
    if shift > round(perforation.width * 0.55):
        return perforation

    return Component(
        x=visible_left,
        y=perforation.y,
        width=perforation.x2 - visible_left + 1,
        height=perforation.height,
        area=perforation.area,
        fill_ratio=perforation.fill_ratio,
        angle_degrees=perforation.angle_degrees,
        repaired=True,
    )


def parse_output_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("--output-size must use WIDTHxHEIGHT, for example 1440x1080")

    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("--output-size width and height must be positive")
    return width, height


def validate_jpeg_quality(value: int) -> int:
    if not 1 <= value <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    return value


def parse_search_bound(text: str, label: str) -> int | float:
    if text.endswith("%"):
        percent = float(text[:-1])
        if not 0 <= percent <= 100:
            raise ValueError(f"{label} percentage must be between 0% and 100%")
        return percent / 100.0
    value = int(text)
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def parse_search_range(value: str | None, label: str) -> tuple[int | float | None, int | float | None]:
    if value is None or value == "":
        return None, None
    if ":" not in value:
        raise ValueError(f"{label} must use START:END, for example 3600:4056")

    start_text, end_text = value.split(":", maxsplit=1)
    start = parse_search_bound(start_text, f"{label} start") if start_text else None
    end = parse_search_bound(end_text, f"{label} end") if end_text else None
    if start is not None and end is not None and end <= start:
        raise ValueError(f"{label} end must be greater than start")
    return start, end


def parse_search_area(x_range: str | None, y_range: str | None) -> SearchArea | None:
    x1, x2 = parse_search_range(x_range, "--perforation-search-x")
    y1, y2 = parse_search_range(y_range, "--perforation-search-y")
    if x1 is None and x2 is None and y1 is None and y2 is None:
        return None
    return SearchArea(x1=x1 or 0, x2=x2, y1=y1 or 0, y2=y2)


def imwrite_jpeg(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(str(path), image, params):
        raise ValueError(f"Could not write image: {path}")


def clamp_crop_box(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> CropBox:
    width = min(width, image_width)
    height = min(height, image_height)
    x = min(max(0, x), image_width - width)
    return CropBox(x=x, y=y, width=width, height=height)


def crop_box_from_perforation(
    perforation: Component,
    image_width: int,
    image_height: int,
    frame_aspect: float,
    frame_height_perf_ratio: float,
    crop_scale: float,
    right_gap_perf_ratio: float,
    center_y_offset_perf_ratio: float,
) -> CropBox:
    crop_height = round(perforation.height * frame_height_perf_ratio * crop_scale)
    crop_width = round(crop_height * frame_aspect)
    crop_right = perforation.x
    crop_x = crop_right - crop_width
    if crop_x < 0:
        crop_width = max(1, crop_right)
        crop_x = 0
    crop_center_y = perforation.center_y + perforation.height * center_y_offset_perf_ratio
    crop_y = round(crop_center_y - crop_height / 2)
    bottom_overflow = crop_y + crop_height - image_height
    if 0 < bottom_overflow <= 64:
        crop_y -= bottom_overflow
    return clamp_crop_box(crop_x, crop_y, crop_width, crop_height, image_width, image_height)


def cyan_artifact_fraction(image: np.ndarray) -> float:
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    artifact_mask = (green > red + 45) & (blue > red + 35) & (green > 100)
    return float(artifact_mask.mean())


def normalize_color(image: np.ndarray, color_mode: str, artifact_threshold: float) -> tuple[np.ndarray, str, float]:
    artifact_fraction = cyan_artifact_fraction(image)
    if color_mode == "grayscale" or (color_mode == "auto" and artifact_fraction >= artifact_threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "grayscale", artifact_fraction
    return image, "color", artifact_fraction


def extract_crop(
    image: np.ndarray,
    image_path: Path,
    crop_box: CropBox,
    stitch_image: np.ndarray | None,
    stitch_image_path: Path | None,
    inter_frame_gap_pixels: int,
) -> tuple[np.ndarray, CropStitch]:
    image_height = image.shape[0]
    x_slice = slice(crop_box.x, crop_box.x + crop_box.width)

    if crop_box.y < 0:
        stitch_pixels = min(-crop_box.y, crop_box.height)
        current_pixels = crop_box.height - stitch_pixels
        current_part = image[0:current_pixels, x_slice]

        source_y = crop_box.y2 + 1 + inter_frame_gap_pixels
        if source_y + stitch_pixels <= image_height:
            stitch_part = image[source_y : source_y + stitch_pixels, x_slice]
            stitch = CropStitch(stitch_pixels, str(image_path), source_y, "top")
        elif stitch_image is not None:
            source_y = max(0, min(source_y, stitch_image.shape[0] - stitch_pixels))
            stitch_part = stitch_image[source_y : source_y + stitch_pixels, x_slice]
            stitch = CropStitch(stitch_pixels, str(stitch_image_path), source_y, "top")
        else:
            source_y = max(0, image_height - stitch_pixels)
            stitch_part = image[source_y : source_y + stitch_pixels, x_slice]
            stitch = CropStitch(stitch_pixels, str(image_path), source_y, "top")

        return np.vstack((stitch_part, current_part)), stitch

    if crop_box.y2 >= image_height:
        current_pixels = max(0, image_height - crop_box.y)
        stitch_pixels = crop_box.height - current_pixels
        current_part = image[crop_box.y : image_height, x_slice]

        source_y = inter_frame_gap_pixels
        if source_y + stitch_pixels <= image_height:
            stitch_part = image[source_y : source_y + stitch_pixels, x_slice]
            stitch = CropStitch(stitch_pixels, str(image_path), source_y, "bottom")
        elif stitch_image is not None:
            source_y = max(0, min(source_y, stitch_image.shape[0] - stitch_pixels))
            stitch_part = stitch_image[source_y : source_y + stitch_pixels, x_slice]
            stitch = CropStitch(stitch_pixels, str(stitch_image_path), source_y, "bottom")
        else:
            source_y = 0
            stitch_part = image[source_y : source_y + stitch_pixels, x_slice]
            stitch = CropStitch(stitch_pixels, str(image_path), source_y, "bottom")

        return np.vstack((current_part, stitch_part)), stitch

    return image[crop_box.y : crop_box.y + crop_box.height, x_slice], CropStitch()


def crop_and_normalize(
    image: np.ndarray,
    image_path: Path,
    crop_box: CropBox,
    output_path: Path,
    output_size: tuple[int, int],
    color_mode: str,
    artifact_threshold: float,
    mirror_horizontal: bool,
    jpeg_quality: int,
    stitch_image: np.ndarray | None,
    stitch_image_path: Path | None,
    inter_frame_gap_pixels: int,
) -> tuple[str, float, CropStitch]:
    cropped, stitch = extract_crop(
        image,
        image_path,
        crop_box,
        stitch_image,
        stitch_image_path,
        inter_frame_gap_pixels,
    )
    cropped, applied_color_mode, artifact_fraction = normalize_color(cropped, color_mode, artifact_threshold)
    if mirror_horizontal:
        cropped = cv2.flip(cropped, 1)
    normalized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    imwrite_jpeg(output_path, normalized, jpeg_quality)
    return applied_color_mode, artifact_fraction, stitch


def fallback_normalize(
    image_path: Path,
    output_path: Path,
    output_size: tuple[int, int],
    args: argparse.Namespace,
    reason: str,
) -> dict[str, object]:
    image = load_image(image_path)
    image, applied_color_mode, artifact_fraction = normalize_color(
        image,
        args.color_mode,
        args.artifact_threshold,
    )
    if args.mirror_horizontal:
        image = cv2.flip(image, 1)
    normalized = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    imwrite_jpeg(output_path, normalized, args.jpeg_quality)

    return {
        "image": str(image_path),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "perforation": None,
        "crop": None,
        "deskew_degrees": 0.0,
        "normalized_output": str(output_path),
        "normalized_width": output_size[0],
        "normalized_height": output_size[1],
        "jpeg_quality": args.jpeg_quality,
        "color_mode": applied_color_mode,
        "mirror_horizontal": args.mirror_horizontal,
        "cyan_artifact_fraction": round(artifact_fraction, 4),
        "candidate_count": 0,
        "status": "fallback",
        "fallback_reason": reason,
    }


def crop_box_from_log(data: dict[str, object]) -> CropBox:
    return CropBox(
        x=int(data["x"]),
        y=int(data["y"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def is_reliable_folder_geometry(result: dict[str, object]) -> bool:
    return result.get("status") == "ok" and isinstance(result.get("crop"), dict)


def nearest_reliable_result_index(
    results: list[dict[str, object]],
    reliable_indexes: list[int],
    target_index: int,
) -> int | None:
    if not reliable_indexes:
        return None
    return min(
        reliable_indexes,
        key=lambda reliable_index: (
            abs(reliable_index - target_index),
            reliable_index > target_index,
        ),
    )


def normalize_with_borrowed_geometry(
    image_path: Path,
    output_path: Path,
    output_size: tuple[int, int],
    args: argparse.Namespace,
    source_result: dict[str, object],
    next_image_path: Path | None,
) -> dict[str, object]:
    image = load_image(image_path)
    deskew_degrees = float(source_result.get("deskew_degrees", 0.0))
    if abs(deskew_degrees) > 0.01:
        image = rotate_bound(image, deskew_degrees)

    stitch_image = None
    if next_image_path is not None:
        stitch_image = load_image(next_image_path)
        if abs(deskew_degrees) > 0.01:
            stitch_image = rotate_bound(stitch_image, deskew_degrees)

    crop_box = crop_box_from_log(source_result["crop"])
    applied_color_mode, artifact_fraction, stitch = crop_and_normalize(
        image,
        image_path,
        crop_box,
        output_path,
        output_size,
        color_mode=args.color_mode,
        artifact_threshold=args.artifact_threshold,
        mirror_horizontal=args.mirror_horizontal,
        jpeg_quality=args.jpeg_quality,
        stitch_image=stitch_image,
        stitch_image_path=next_image_path,
        inter_frame_gap_pixels=args.inter_frame_gap_pixels,
    )

    return {
        "image": str(image_path),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "perforation": None,
        "crop": crop_box.to_log_dict(),
        "vertical_stitch": stitch.to_log_dict(),
        "vertical_stitch_pixels": stitch.pixels,
        "deskew_degrees": round(deskew_degrees, 4),
        "normalized_output": str(output_path),
        "normalized_width": output_size[0],
        "normalized_height": output_size[1],
        "jpeg_quality": args.jpeg_quality,
        "color_mode": applied_color_mode,
        "mirror_horizontal": args.mirror_horizontal,
        "cyan_artifact_fraction": round(artifact_fraction, 4),
        "candidate_count": 0,
        "status": "borrowed_geometry",
        "borrowed_geometry_from": str(source_result["image"]),
    }


def annotate_crop_and_perforation(
    image: np.ndarray,
    output_path: Path,
    perforation: Component,
    crop_box: CropBox,
    deskew_degrees: float,
    jpeg_quality: int,
) -> None:
    annotated = image.copy()
    if crop_box.y < 0:
        wrapped_height = -crop_box.y
        cv2.rectangle(
            annotated,
            (crop_box.x, image.shape[0] - wrapped_height),
            (crop_box.x2, image.shape[0] - 1),
            (0, 255, 0),
            12,
        )
        cv2.rectangle(
            annotated,
            (crop_box.x, 0),
            (crop_box.x2, crop_box.y2),
            (0, 255, 0),
            12,
        )
    elif crop_box.y2 >= image.shape[0]:
        visible_bottom = image.shape[0] - crop_box.y
        wrapped_height = crop_box.height - visible_bottom
        cv2.rectangle(
            annotated,
            (crop_box.x, crop_box.y),
            (crop_box.x2, image.shape[0] - 1),
            (0, 255, 0),
            12,
        )
        cv2.rectangle(
            annotated,
            (crop_box.x, 0),
            (crop_box.x2, wrapped_height - 1),
            (0, 255, 0),
            12,
        )
    else:
        cv2.rectangle(
            annotated,
            (crop_box.x, crop_box.y),
            (crop_box.x2, crop_box.y2),
            (0, 255, 0),
            12,
        )
    cv2.rectangle(
        annotated,
        (perforation.x, perforation.y),
        (perforation.x2, perforation.y2),
        (0, 0, 255),
        12,
    )
    cv2.putText(
        annotated,
        "crop",
        (crop_box.x, max(24, crop_box.y + 36)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "perforation",
        (perforation.x, max(24, perforation.y - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"deskew {deskew_degrees:.2f} deg",
        (24, image.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        4,
        cv2.LINE_AA,
    )
    imwrite_jpeg(output_path, annotated, jpeg_quality)


def relaxed_detection_tiers(threshold: int, max_chroma: int) -> list[tuple[int, int]]:
    tiers = [
        (threshold, max_chroma),
        (min(threshold, 240), max(max_chroma, 20)),
        (min(threshold, 235), max(max_chroma, 30)),
        (min(threshold, 230), max(max_chroma, 35)),
        (min(threshold, 220), max(max_chroma, 60)),
    ]
    unique_tiers: list[tuple[int, int]] = []
    for tier in tiers:
        if tier not in unique_tiers:
            unique_tiers.append(tier)
    return unique_tiers


def offset_component(component: Component, x_offset: int, y_offset: int) -> Component:
    return Component(
        x=component.x + x_offset,
        y=component.y + y_offset,
        width=component.width,
        height=component.height,
        area=component.area,
        fill_ratio=component.fill_ratio,
        angle_degrees=component.angle_degrees,
        repaired=component.repaired,
    )


def detect_perforation(
    image: np.ndarray,
    threshold: int,
    max_chroma: int,
    min_area: int,
    search_area: SearchArea | None = None,
) -> tuple[Component | None, int]:
    x_offset = 0
    y_offset = 0
    search_image = image
    if search_area is not None:
        x1, x2, y1, y2 = search_area.bounds(image.shape[1], image.shape[0])
        x_offset = x1
        y_offset = y1
        search_image = image[y1:y2, x1:x2]

    best_candidates: list[tuple[Component, int]] = []
    strict_candidate: Component | None = None
    last_candidate_count = 0
    for tier_index, (relaxed_threshold, relaxed_chroma) in enumerate(relaxed_detection_tiers(threshold, max_chroma)):
        mask = white_mask(search_image, relaxed_threshold, relaxed_chroma)
        components = find_white_components(mask, min_area)
        last_candidate_count = len(components)
        candidate_components = add_merged_perforation_fragments(
            components,
            search_image.shape[1],
            search_image.shape[0],
            image.shape[1],
            image.shape[0],
        )
        perforation = choose_perforation(
            candidate_components,
            search_image.shape[1],
            search_image.shape[0],
            size_image_width=image.shape[1],
            size_image_height=image.shape[0],
        )
        if perforation is not None:
            full_component = offset_component(perforation, x_offset, y_offset)
            if tier_index == 0:
                strict_candidate = full_component
            best_candidates.append((full_component, len(components)))

    if not best_candidates:
        return None, last_candidate_count

    if strict_candidate is not None:
        max_replacement_width = max(round(strict_candidate.width * 1.8), strict_candidate.width + 80)
        filtered_candidates = [
            candidate
            for candidate in best_candidates
            if candidate[0] is strict_candidate or candidate[0].width <= max_replacement_width
        ]
        if filtered_candidates:
            best_candidates = filtered_candidates

    return max(
        best_candidates,
        key=lambda candidate: (
            candidate[0].area,
            candidate[0].fill_ratio,
            candidate[0].width,
        ),
    )


def image_paths_from_folder(folder: Path, pattern: str, name_regex: str) -> list[Path]:
    regex = re.compile(name_regex, re.IGNORECASE)
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and regex.fullmatch(path.name)
    )


def output_path_for_image(image_path: Path, output: Path, input_is_folder: bool) -> Path:
    if input_is_folder or output.suffix == "":
        return output / f"normalized_{image_path.stem}.jpg"
    return output


def fallback_output_path_for_image(image_path: Path, output: Path, input_is_folder: bool) -> Path:
    output_path = output_path_for_image(image_path, output, input_is_folder)
    return output_path.with_name(f"fb_{output_path.name}")


def annotation_path_for_image(image_path: Path, annotate: Path | None, input_is_folder: bool) -> Path | None:
    if annotate is None:
        return None
    if input_is_folder or annotate.suffix == "":
        return annotate / f"{image_path.stem}_debug.jpg"
    return annotate


def process_image(
    image_path: Path,
    output_path: Path,
    annotation_path: Path | None,
    output_size: tuple[int, int],
    args: argparse.Namespace,
    next_image_path: Path | None = None,
) -> dict[str, object]:
    image = load_image(image_path)
    search_area = parse_search_area(
        getattr(args, "perforation_search_x", None),
        getattr(args, "perforation_search_y", None),
    )

    rectangle, candidate_count = detect_perforation(
        image,
        args.threshold,
        args.max_chroma,
        args.min_area,
        search_area=search_area,
    )

    if rectangle is None:
        raise ValueError(f"No perforation found in {image_path}")

    repair_enabled = not args.disable_perforation_repair
    rectangle = repair_perforation_box(
        rectangle,
        image_height=image.shape[0],
        expected_height_width_ratio=args.perforation_height_width_ratio,
        enabled=repair_enabled,
    )
    rectangle = repair_visible_perforation_left_edge(image, rectangle)

    deskew_degrees = 0.0
    if not args.no_deskew:
        requested_deskew = deskew_degrees_from_vertical(rectangle.angle_degrees)
        if abs(requested_deskew) <= args.max_deskew_degrees:
            deskew_degrees = requested_deskew
            if abs(deskew_degrees) > 0.01:
                image = rotate_bound(image, deskew_degrees)
                rectangle, candidate_count = detect_perforation(
                    image,
                    args.threshold,
                    args.max_chroma,
                    args.min_area,
                )
                if rectangle is None:
                    raise ValueError(f"No perforation found after deskewing {image_path}")
                rectangle = repair_perforation_box(
                    rectangle,
                    image_height=image.shape[0],
                    expected_height_width_ratio=args.perforation_height_width_ratio,
                    enabled=repair_enabled,
                )
                rectangle = repair_visible_perforation_left_edge(image, rectangle)
        else:
            logging.warning(
                "%s: skipping deskew, detected %.2f degrees, which is above --max-deskew-degrees %.2f",
                image_path,
                requested_deskew,
                args.max_deskew_degrees,
                )

    stitch_image = None
    if next_image_path is not None:
        stitch_image = load_image(next_image_path)
        if abs(deskew_degrees) > 0.01:
            stitch_image = rotate_bound(stitch_image, deskew_degrees)

    crop_box = crop_box_from_perforation(
        rectangle,
        image_width=image.shape[1],
        image_height=image.shape[0],
        frame_aspect=args.frame_aspect,
        frame_height_perf_ratio=args.frame_height_perf_ratio,
        crop_scale=args.crop_scale,
        right_gap_perf_ratio=args.right_gap_perf_ratio,
        center_y_offset_perf_ratio=args.center_y_offset_perf_ratio,
    )
    applied_color_mode, artifact_fraction, stitch = crop_and_normalize(
        image,
        image_path,
        crop_box,
        output_path,
        output_size,
        color_mode=args.color_mode,
        artifact_threshold=args.artifact_threshold,
        mirror_horizontal=args.mirror_horizontal,
        jpeg_quality=args.jpeg_quality,
        stitch_image=stitch_image,
        stitch_image_path=next_image_path,
        inter_frame_gap_pixels=args.inter_frame_gap_pixels,
    )
    stitch_log = stitch.to_log_dict()

    result = {
        "image": str(image_path),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "perforation": rectangle.to_log_dict(),
        "crop": crop_box.to_log_dict(),
        "vertical_stitch": stitch_log,
        "vertical_stitch_pixels": stitch.pixels,
        "deskew_degrees": round(deskew_degrees, 4),
        "normalized_output": str(output_path),
        "normalized_width": output_size[0],
        "normalized_height": output_size[1],
        "jpeg_quality": args.jpeg_quality,
        "color_mode": applied_color_mode,
        "mirror_horizontal": args.mirror_horizontal,
        "cyan_artifact_fraction": round(artifact_fraction, 4),
        "candidate_count": candidate_count,
    }
    if search_area is not None:
        result["perforation_search_area"] = search_area.to_log_dict(image.shape[1], image.shape[0])

    if annotation_path:
        annotate_crop_and_perforation(image, annotation_path, rectangle, crop_box, deskew_degrees, args.jpeg_quality)
        result["annotation_output"] = str(annotation_path)

    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        output_size = parse_output_size(args.output_size)
        args.jpeg_quality = validate_jpeg_quality(args.jpeg_quality)
        parse_search_area(args.perforation_search_x, args.perforation_search_y)
    except ValueError as error:
        logging.error("%s", error)
        return 1

    input_is_folder = args.input.is_dir()
    if input_is_folder:
        image_paths = image_paths_from_folder(args.input, args.glob, args.name_regex)
        output = args.output or Path("out")
        log_path = args.log
    else:
        image_paths = [args.input]
        output = args.output or Path("normalized_frame.jpg")
        log_path = args.log

    if not image_paths:
        logging.error(
            "No JPEG images matching glob %r and regex %r found in %s",
            args.glob,
            args.name_regex,
            args.input,
        )
        return 1

    results: list[dict[str, object]] = []
    failures = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_path_for_image(image_path, output, input_is_folder)
        annotation_path = annotation_path_for_image(image_path, args.annotate, input_is_folder)

        next_image_path = image_paths[index] if input_is_folder and index < len(image_paths) else None

        try:
            result = process_image(
                image_path,
                output_path,
                annotation_path,
                output_size,
                args,
                next_image_path=next_image_path,
            )
        except ValueError as error:
            fallback_output_path = fallback_output_path_for_image(image_path, output, input_is_folder)
            result = fallback_normalize(image_path, fallback_output_path, output_size, args, str(error))
            failures += 1
            logging.warning(
                "[%d/%d] %s -> %s, fallback full-image resize: %s",
                index,
                len(image_paths),
                image_path,
                fallback_output_path,
                error,
            )
        else:
            result["status"] = "ok"
            logging.info(
                "[%d/%d] %s -> %s, deskew %.4f degrees, %s",
                index,
                len(image_paths),
                image_path,
                output_path,
                result["deskew_degrees"],
                result["color_mode"],
            )

        results.append(result)

    borrowed_geometry = 0
    if input_is_folder:
        reliable_indexes = [
            result_index
            for result_index, result in enumerate(results)
            if is_reliable_folder_geometry(result)
        ]
        for result_index, result in enumerate(results):
            if result.get("status") != "fallback":
                continue

            source_index = nearest_reliable_result_index(results, reliable_indexes, result_index)
            if source_index is None:
                continue

            image_path = image_paths[result_index]
            output_path = output_path_for_image(image_path, output, input_is_folder)
            next_image_path = image_paths[result_index + 1] if result_index + 1 < len(image_paths) else None
            source_result = results[source_index]

            results[result_index] = normalize_with_borrowed_geometry(
                image_path,
                output_path,
                output_size,
                args,
                source_result,
                next_image_path,
            )
            borrowed_geometry += 1
            logging.info(
                "[%d/%d] %s -> %s, borrowed crop geometry from %s",
                result_index + 1,
                len(image_paths),
                image_path,
                output_path,
                source_result["image"],
            )

        failures = sum(1 for result in results if result.get("status") == "fallback")

    log_data: dict[str, object]
    if input_is_folder:
        log_data = {
            "input": str(args.input),
            "output": str(output),
            "count": len(image_paths),
            "succeeded": len(image_paths) - failures,
            "failed": failures,
            "borrowed_geometry": borrowed_geometry,
            "frames": results,
        }
    else:
        log_data = results[0]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, indent=2) + "\n", encoding="utf-8")
    logging.info("wrote log: %s", log_path)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
