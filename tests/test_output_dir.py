"""Output directories must be explicit, new or empty, and never overwritten."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stress_detection.audit import run_full_audit
from stress_detection.io_guard import OutputDirError, ensure_empty_output_dir

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"
SCRIPT = ROOT / "scripts" / "run_audit.py"
BENCH = ROOT / "scripts" / "run_benchmark.py"


def _snapshot(path: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in path.iterdir() if p.is_file()}


def test_ensure_empty_accepts_missing_and_empty(tmp_path):
    missing = tmp_path / "a" / "b"
    got = ensure_empty_output_dir(missing)
    assert got.is_dir()
    assert list(got.iterdir()) == []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ensure_empty_output_dir(empty) == empty


def test_ensure_empty_rejects_nonempty(tmp_path):
    d = tmp_path / "used"
    d.mkdir()
    (d / "sentinel.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OutputDirError, match="not empty"):
        ensure_empty_output_dir(d)
    assert (d / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert list(d.iterdir()) == [d / "sentinel.txt"]


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_api_rejects_nonempty_and_existing_results_json(tmp_path):
    d = tmp_path / "occupied"
    d.mkdir()
    (d / "results.json").write_text('{"status": "old"}', encoding="utf-8")
    before = _snapshot(d)
    with pytest.raises(OutputDirError, match="not empty"):
        run_full_audit(CSV, d, smoke=True, seed=0, project_root=ROOT)
    assert _snapshot(d) == before


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_cli_rejects_nonempty_sentinel(tmp_path):
    d = tmp_path / "occupied"
    d.mkdir()
    (d / "sentinel.txt").write_text("keep-me", encoding="utf-8")
    names_before = {p.name for p in d.iterdir()}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(CSV), str(d), "--smoke"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "not empty" in proc.stderr
    assert {p.name for p in d.iterdir()} == names_before
    assert (d / "sentinel.txt").read_text(encoding="utf-8") == "keep-me"


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_cli_smoke_does_not_write_historical_audit_full(tmp_path):
    fake_hist = tmp_path / "results" / "audit_full"
    fake_hist.mkdir(parents=True)
    (fake_hist / "historical.txt").write_text("untouched", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(CSV), str(fake_hist), "--smoke"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert (fake_hist / "historical.txt").read_text(encoding="utf-8") == "untouched"
    assert {p.name for p in fake_hist.iterdir()} == {"historical.txt"}


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_cli_accepts_empty_dir(tmp_path):
    d = tmp_path / "empty_ok"
    d.mkdir()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(CSV), str(d), "--smoke", "--seed", "0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    status = json.loads((d / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert (d / "results.json").is_file()


def test_benchmark_rejects_nonempty(tmp_path):
    d = tmp_path / "bench"
    d.mkdir()
    (d / "old.json").write_text("{}", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(BENCH), str(CSV), str(d)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "not empty" in proc.stderr
    assert (d / "old.json").read_text(encoding="utf-8") == "{}"
