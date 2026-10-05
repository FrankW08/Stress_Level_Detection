"""Environment snapshots, file hashes, and deterministic code manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def normalize_newlines_to_lf(data: bytes) -> bytes:
    """Normalize CRLF and lone CR to LF for stable cross-platform hashing."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_file_normalized_lf(path: Path) -> str:
    return sha256_bytes(normalize_newlines_to_lf(Path(path).read_bytes()))


def build_environment_lock_text() -> str:
    """
    Sorted name==version snapshot via importlib.metadata.
    Raises RuntimeError if the snapshot would be empty.
    """
    rows: list[str] = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        version = dist.version
        rows.append(f"{name}=={version}")
    rows = sorted(set(rows), key=str.lower)
    if not rows:
        raise RuntimeError(
            "Environment lock snapshot is empty (no installed distributions found). "
            "Refusing to write an empty requirements-lock.txt."
        )
    return "\n".join(rows) + "\n"


def write_environment_lock(out_path: Path) -> str:
    text = build_environment_lock_text()
    out_path = Path(out_path)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    if not text.strip():
        raise RuntimeError("Refusing empty requirements-lock.txt")
    return text


def code_manifest_entries(project_root: Path) -> list[tuple[str, str]]:
    """
    Deterministic (relpath, sha256) pairs for:
    pyproject.toml, scripts/run_audit.py, src/stress_detection/**/*.py
    Text files are hashed after CRLF→LF normalization.
    """
    root = Path(project_root).resolve()
    paths: list[Path] = [
        root / "pyproject.toml",
        root / "scripts" / "run_audit.py",
    ]
    pkg = root / "src" / "stress_detection"
    if pkg.is_dir():
        paths.extend(sorted(pkg.rglob("*.py")))
    entries: list[tuple[str, str]] = []
    for p in paths:
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        digest = sha256_bytes(normalize_newlines_to_lf(p.read_bytes()))
        entries.append((rel, digest))
    entries.sort(key=lambda t: t[0])
    return entries


def code_manifest_sha256(project_root: Path) -> tuple[str, list[dict[str, str]]]:
    entries = code_manifest_entries(project_root)
    # Hash of "path\\nsha\\n" lines in sorted path order
    payload = "".join(f"{rel}\n{digest}\n" for rel, digest in entries).encode("utf-8")
    return sha256_bytes(payload), [{"path": rel, "sha256": digest} for rel, digest in entries]
