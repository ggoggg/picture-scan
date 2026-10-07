#!/usr/bin/env python3
"""Rename numbered frame files so their sequence has no gaps."""

from __future__ import annotations

import argparse
import re
import shutil
import uuid
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renumber files like frame_000001.jpg, frame_000004.jpg to frame_000001.jpg, frame_000002.jpg."
    )
    parser.add_argument("folder", type=Path, help="Folder containing sequenced files.")
    parser.add_argument(
        "--prefix",
        help="Filename prefix before the six-digit number. Default: infer from matching files.",
    )
    parser.add_argument(
        "--suffix",
        default=".jpg",
        help="Filename suffix after the number. Default: .jpg.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First output number. Default: 1.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Output number width. Default: 6.",
    )
    parser.add_argument(
        "--mirror-horizontal",
        action="store_true",
        help="Mirror every matching image left-to-right in place, including unchanged filenames. Requires Pillow; JPEGs are re-encoded.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename and optionally mirror files. Without this, only print the planned changes.",
    )
    return parser.parse_args()


def matching_files(folder: Path, prefix: str | None, suffix: str, digits: int) -> tuple[str, list[tuple[int, Path]]]:
    if prefix is None:
        pattern = re.compile(rf"^(.+_)(\d{{{digits}}}){re.escape(suffix)}$")
    else:
        pattern = re.compile(rf"^({re.escape(prefix)})(\d{{{digits}}}){re.escape(suffix)}$")

    matches_by_prefix: dict[str, list[tuple[int, Path]]] = {}

    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            matched_prefix = match.group(1)
            number = int(match.group(2))
            matches_by_prefix.setdefault(matched_prefix, []).append((number, path))

    if not matches_by_prefix:
        return prefix or "", []

    if prefix is None and len(matches_by_prefix) > 1:
        names = ", ".join(sorted(matches_by_prefix))
        raise ValueError(f"Multiple prefixes match *_%0{digits}d{suffix}: {names}. Use --prefix.")

    matched_prefix, matches = next(iter(matches_by_prefix.items()))
    return matched_prefix, sorted(matches, key=lambda item: (item[0], item[1].name))


def planned_renames(
    files: list[tuple[int, Path]],
    prefix: str,
    suffix: str,
    start: int,
    digits: int,
) -> list[tuple[Path, Path]]:
    renames: list[tuple[Path, Path]] = []

    for offset, (_, source) in enumerate(files):
        number = start + offset
        target = source.with_name(f"{prefix}{number:0{digits}d}{suffix}")
        if source != target:
            renames.append((source, target))

    return renames


def validate_targets(renames: list[tuple[Path, Path]], sources: set[Path]) -> None:
    targets = [target for _, target in renames]
    duplicate_targets = {target for target in targets if targets.count(target) > 1}
    if duplicate_targets:
        names = ", ".join(sorted(path.name for path in duplicate_targets))
        raise ValueError(f"Duplicate output names would be created: {names}")

    conflicts = [target for target in targets if target.exists() and target not in sources]
    if conflicts:
        names = ", ".join(sorted(path.name for path in conflicts))
        raise ValueError(f"Output names already exist and are not part of this sequence: {names}")


def apply_renames(renames: list[tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    temporary_paths: list[tuple[Path, Path]] = []

    for source, _ in renames:
        temporary = source.with_name(f".{source.name}.renumber-{token}.tmp")
        source.rename(temporary)
        temporary_paths.append((temporary, source))

    temporary_by_source_name = {original: temporary for temporary, original in temporary_paths}
    for source, target in renames:
        temporary_by_source_name[source].rename(target)


def mirror_horizontal(path: Path) -> None:
    from PIL import Image, ImageOps

    temporary = path.with_name(f".{path.name}.mirror-{uuid.uuid4().hex}.tmp")
    try:
        with Image.open(path) as original:
            image = ImageOps.mirror(ImageOps.exif_transpose(original))
            options = {}
            for key in ("exif", "icc_profile", "dpi"):
                if key in image.info:
                    options[key] = image.info[key]
            if original.format == "JPEG":
                options.update(quality=95, subsampling=0)
            image.save(temporary, format=original.format, **options)
        shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"Not a folder: {args.folder}")

    try:
        prefix, files = matching_files(args.folder, args.prefix, args.suffix, args.digits)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if not files:
        raise SystemExit("No matching files found.")

    renames = planned_renames(files, prefix, args.suffix, args.start, args.digits)
    if not renames and not args.mirror_horizontal:
        print("Sequence is already continuous.")
        return 0

    validate_targets(renames, {path for _, path in files})

    for source, target in renames:
        print(f"{source.name} -> {target.name}")

    if args.mirror_horizontal:
        print(f"Mirror horizontally: all {len(files)} matching image(s), including unchanged filenames.")

    if args.apply:
        if args.mirror_horizontal:
            try:
                from PIL import Image  # noqa: F401
            except ImportError as error:
                raise SystemExit("Horizontal mirroring requires Pillow. Install with: python -m pip install Pillow") from error
            for index, (_, source) in enumerate(files, start=1):
                try:
                    mirror_horizontal(source)
                except Exception as error:
                    raise SystemExit(
                        f"Mirroring failed for {source.name}: {error}. "
                        f"{index - 1} image(s) already mirrored; renumbering has not started."
                    ) from error
                print(f"Mirrored {index}/{len(files)}: {source.name}")
        apply_renames(renames)
        print(f"Renamed {len(renames)} file(s).")
    else:
        print(f"Dry run: {len(renames)} file(s) would be renamed. Add --apply to perform the planned changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
