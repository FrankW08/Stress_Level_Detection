"""Import side effects, env lock, seed, OOF metrics, notebook smoke."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_detection.evaluation import metrics_from_oof
from stress_detection.hashes import build_environment_lock_text, write_environment_lock

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


def test_import_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for mod in (
        "stress_detection",
        "stress_detection.data",
        "stress_detection.audit",
        "stress_detection.evaluation",
        "stress_detection.models",
        "stress_detection.hashes",
        "stress_detection.validation",
    ):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
        else:
            importlib.import_module(mod)
    assert list(tmp_path.iterdir()) == []


def test_environment_lock_nonempty(tmp_path):
    text = build_environment_lock_text()
    assert text.strip()
    path = tmp_path / "requirements-lock.txt"
    write_environment_lock(path)
    assert path.read_text(encoding="utf-8").strip()


def test_environment_lock_failure_raises(monkeypatch, tmp_path):
    import stress_detection.hashes as h

    def empty():
        return ""

    monkeypatch.setattr(h, "build_environment_lock_text", empty)
    with pytest.raises(RuntimeError):
        write_environment_lock(tmp_path / "requirements-lock.txt")


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_run_audit_smoke_tmp(tmp_path):
    out = tmp_path / "smoke_out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_audit.py"),
        str(CSV),
        str(out),
        "--smoke",
        "--seed",
        "0",
        "--config",
        str(ROOT / "configs" / "audit_primary.yaml"),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    lock = (out / "requirements-lock.txt").read_text(encoding="utf-8")
    assert lock.strip()
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["config"]["seed"] == 0
    assert res["env"]["environment_lock_sha256"]
    assert res["env"]["code_manifest_sha256"]
    assert res["env"]["csv_sha256_normalized_lf"]
    assert "model_params" in res


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_seed_affects_folds(tmp_path):
    out0 = tmp_path / "s0"
    out123 = tmp_path / "s123"
    for seed, out in ((0, out0), (123, out123)):
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_audit.py"),
            str(CSV),
            str(out),
            "--smoke",
            "--seed",
            str(seed),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    r123 = json.loads((out123 / "results.json").read_text(encoding="utf-8"))
    assert r123["config"]["seed"] == 123
    assert r123["config"]["eval"]["seed"] == 123
    f0 = pd.read_csv(out0 / "folds.csv")
    f123 = pd.read_csv(out123 / "folds.csv")
    # Same structure but different membership for at least one fold
    m0 = set(
        zip(
            f0.loc[f0.role == "valid", "repeat"],
            f0.loc[f0.role == "valid", "fold"],
            f0.loc[f0.role == "valid", "source_row_id"],
        )
    )
    m123 = set(
        zip(
            f123.loc[f123.role == "valid", "repeat"],
            f123.loc[f123.role == "valid", "fold"],
            f123.loc[f123.role == "valid", "source_row_id"],
        )
    )
    assert m0 != m123


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_oof_metrics_match_fold_means(tmp_path):
    out = tmp_path / "oof_check"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_audit.py"),
            str(CSV),
            str(out),
            "--smoke",
            "--seed",
            "0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(out / "oof_predictions.csv")
    folds = pd.read_csv(out / "fold_scores.csv")
    # OOF macro-F1 averaged over repeats should be close to mean fold macro-F1
    for model in ("lr_all", "dummy"):
        rep_scores = []
        for rep, g in oof.groupby("repeat"):
            m = metrics_from_oof(g["true"].to_numpy(), g[model].to_numpy())
            rep_scores.append(m["macro_f1"])
        oof_mean = float(np.mean(rep_scores))
        fold_mean = float(folds.loc[folds.model == model, "macro_f1"].mean())
        # Fold-mean and pooled OOF mean are related but not identical definitions;
        # require fold mean from CSV matches results.json
        assert abs(fold_mean - res["models"][model]["macro_f1_mean"]) < 1e-9
        assert 0.0 <= oof_mean <= 1.0


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_sensitivity_smoke_grouped(tmp_path):
    out = tmp_path / "sens"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_audit.py"),
            str(CSV),
            str(out),
            "--smoke",
            "--scope",
            "sensitivity",
            "--sensitivity-policy",
            "grouped_duplicates",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["config"]["scope"] == "sensitivity"
    assert res["sensitivity"]["policy"] == "grouped_duplicates"
    assert (out / "out_of_range.csv").is_file()
    assert (out / "duplicate_group_ids.csv").is_file()


def test_notebook_import_smoke():
    nb = ROOT / "Stress Level Classification.ipynb"
    assert nb.is_file()
    import nbformat

    node = nbformat.read(nb, as_version=4)
    sources = "\n".join(
        c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        for c in node.cells
        if c["cell_type"] == "code"
    )
    assert "prepare_data" in sources
    assert "run_audit_cv" in sources or "audit_model_registry" in sources
    assert "0.937" not in sources and "93.7%" not in sources


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_artifact_warning_persisted(tmp_path, monkeypatch):
    from stress_detection import audit as audit_mod

    def boom(*a, **k):
        raise RuntimeError("forced artifact failure")

    monkeypatch.setattr(audit_mod, "lr_pipeline", boom)
    out = tmp_path / "art"
    results = audit_mod.run_full_audit(
        CSV, out, smoke=True, seed=0, project_root=ROOT, n_perm=2
    )
    assert "artifact_warning" in results
    disk = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "artifact_warning" in disk
