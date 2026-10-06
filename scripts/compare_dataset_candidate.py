#!/usr/bin/env python3
"""Compare an external candidate CSV to the repository reference CSV (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_detection.dataset_compare import (  # noqa: E402
    DEFAULT_DETAIL_LIMIT,
    DEFAULT_PREFIX_ROWS,
    compare_files,
)
from stress_detection.io_guard import OutputDirError  # noqa: E402

DEFAULT_REFERENCE = ROOT / "StressLevelDataset_original.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Layered comparison of a candidate CSV against the repo reference. "
            "Output directory must be new or empty. Does not modify inputs."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate CSV path")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference CSV (default: StressLevelDataset_original.csv)",
    )
    parser.add_argument(
        "--prefix-rows",
        type=int,
        default=DEFAULT_PREFIX_ROWS,
        help=f"Prefix length for the 0-based data-row prefix compare (default {DEFAULT_PREFIX_ROWS})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (required). Must not already contain files.",
    )
    parser.add_argument(
        "--detail-limit",
        type=int,
        default=DEFAULT_DETAIL_LIMIT,
        help="Max difference rows written to CSV (counts remain complete)",
    )
    args = parser.parse_args()
    if args.prefix_rows < 0:
        parser.error("--prefix-rows must be >= 0")
    if not args.candidate.is_file():
        print(f"error: candidate not found: {args.candidate}", file=sys.stderr)
        return 2
    if not args.reference.is_file():
        print(f"error: reference not found: {args.reference}", file=sys.stderr)
        return 2
    try:
        report = compare_files(
            args.candidate.resolve(),
            args.reference.resolve(),
            args.out_dir.resolve(),
            prefix_rows=args.prefix_rows,
            detail_limit=args.detail_limit,
        )
    except OutputDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"statuses": report["statuses"], "out_dir": str(args.out_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
