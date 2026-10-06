"""Refuse to write into a non-empty output directory.

A failed run may leave partial files. Those files are not a completed audit;
rerun into a new directory instead of overwriting.
"""

from __future__ import annotations

from pathlib import Path


class OutputDirError(ValueError):
    """Output path cannot be used without destroying existing files."""


def list_output_entries(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir())


def ensure_empty_output_dir(path: Path | str) -> Path:
    """Accept a missing path (create it) or an existing empty directory.

    Reject a non-empty directory or a non-directory path. Never deletes.
    """
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise OutputDirError(
            f"output path exists and is not a directory: {path}. "
            "Choose a new directory."
        )
    if path.exists():
        existing = list_output_entries(path)
        if existing:
            preview = ", ".join(existing[:8])
            extra = "" if len(existing) <= 8 else f" (+{len(existing) - 8} more)"
            raise OutputDirError(
                f"output directory is not empty ({len(existing)} entries: {preview}{extra}): {path}. "
                "Refusing to overwrite. Use a new directory. "
                "A previous failed run may have left partial files; "
                "do not treat those as a completed audit."
            )
        return path
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_run_status(out_dir: Path, status: str, **extra: object) -> None:
    import json

    payload = {"status": status, **extra}
    (Path(out_dir) / "run_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
