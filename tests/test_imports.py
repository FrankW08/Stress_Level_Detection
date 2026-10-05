import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_import_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for mod in (
        "stress_detection",
        "stress_detection.data",
        "stress_detection.audit",
        "stress_detection.evaluation",
        "stress_detection.models",
    ):
        importlib.import_module(mod)


def test_run_audit_smoke():
    csv = ROOT / "StressLevelDataset_original.csv"
    if not csv.is_file():
        return
    out = ROOT / "results" / "pytest_smoke_audit"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_audit.py"),
        str(csv),
        str(out),
        "--smoke",
        "--n-perm",
        "3",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
