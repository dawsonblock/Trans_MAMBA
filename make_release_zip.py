#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Iterable, Iterator
from zipfile import ZIP_DEFLATED, ZipInfo, ZipFile


DEFAULT_TOP_DIR = "Trans_MAMBA-lean-infinity-dual-hybrid-v1"


def _iter_files(root_dir: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        p = Path(dirpath)

        dirnames.sort()
        filenames.sort()

        for name in filenames:
            yield p / name


def _should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    if not parts:
        return True

    excluded_top = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "legacy",
    }

    if parts[0] in excluded_top:
        return True

    if rel_path.suffix in {".pyc", ".pyo"}:
        return True

    if any(part == "__pycache__" for part in parts):
        return True

    return False


def build_release_zip(
    *,
    repo_root: Path,
    output_zip: Path,
    top_dir: str = DEFAULT_TOP_DIR,
) -> Path:
    repo_root = repo_root.resolve()
    output_zip = output_zip.resolve()

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    fixed_time = (1980, 1, 1, 0, 0, 0)

    with ZipFile(output_zip, mode="w", compression=ZIP_DEFLATED) as zf:
        for abs_path in _iter_files(repo_root):
            rel_path = abs_path.relative_to(repo_root)
            if _should_exclude(rel_path):
                continue

            arcname = Path(top_dir) / rel_path

            info = ZipInfo(str(arcname).replace(os.sep, "/"))
            info.date_time = fixed_time
            info.compress_type = ZIP_DEFLATED

            with open(abs_path, "rb") as f:
                zf.writestr(info, f.read())

    return output_zip


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build release zip for Lean Infinity Dual Hybrid v1"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{DEFAULT_TOP_DIR}.zip",
        help="Output zip file path",
    )
    parser.add_argument(
        "--top-dir",
        type=str,
        default=DEFAULT_TOP_DIR,
        help="Top-level directory name inside the zip",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    build_release_zip(
        repo_root=Path(__file__).parent,
        output_zip=Path(args.output),
        top_dir=args.top_dir,
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
