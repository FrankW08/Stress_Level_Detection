"""Config resolution for outer CV settings: precedence, validation, and actual effect."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stress_detection.audit import PERSISTENT_ERROR_THRESHOLD  # noqa: E402
from stress_detection.run_config import (  # noqa: E402
    ConfigError,
    resolve_eval_settings,
    validate_file_config,
)

CSV = ROOT / "StressLevelDataset_original.csv"
SCRIPT = ROOT / "scripts" / "run_audit.py"


def _validate(cfg: dict) -> dict:
    return validate_file_config(cfg, persistent_error_threshold=PERSISTENT_ERROR_THRESHOLD)


def _resolve(smoke=False, n_splits=None, n_repeats=None, file_eval=None) -> dict:
    return resolve_eval_settings(
        smoke=smoke,
        cli_n_splits=n_splits,
        cli_n_repeats=n_repeats,
        file_eval=file_eval or {},
    )


def test_full_defaults_without_config():
    r = _resolve()
    assert (r["n_splits"], r["n_repeats"]) == (5, 10)
    assert r["sources"] == {"n_splits": "full_default", "n_repeats": "full_default"}
    assert r["notes"] == []


def test_config_values_used_in_full_mode():
    r = _resolve(file_eval={"n_splits": 4, "n_repeats": 3})
    assert (r["n_splits"], r["n_repeats"]) == (4, 3)
    assert r["sources"] == {"n_splits": "config:eval.n_splits", "n_repeats": "config:eval.n_repeats"}


def test_smoke_defaults_take_precedence_over_config_with_note():
    r = _resolve(smoke=True, file_eval={"n_splits": 5, "n_repeats": 10})
    assert (r["n_splits"], r["n_repeats"]) == (3, 1)
    assert r["sources"] == {"n_splits": "smoke_default", "n_repeats": "smoke_default"}
    assert len(r["notes"]) == 2
    assert all("not applied" in n for n in r["notes"])


@pytest.mark.parametrize("smoke", [False, True])
def test_cli_overrides_config_and_smoke(smoke):
    r = _resolve(smoke=smoke, n_splits=2, n_repeats=4, file_eval={"n_splits": 5, "n_repeats": 10})
    assert (r["n_splits"], r["n_repeats"]) == (2, 4)
    assert r["sources"] == {"n_splits": "cli", "n_repeats": "cli"}


def test_primary_config_file_is_valid():
    from run_audit import _load_yaml_config

    cfg = _load_yaml_config(ROOT / "configs" / "audit_primary.yaml")
    assert _validate(cfg) == {"n_splits": 5, "n_repeats": 10, "seed": 0}


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"eval": {"n_splits": 1}}, "eval.n_splits must be >= 2"),
        ({"eval": {"n_repeats": 0}}, "eval.n_repeats must be >= 1"),
        ({"eval": {"n_splits": "5"}}, "must be an integer"),
        ({"eval": {"n_splits": 5.0}}, "must be an integer"),
        ({"eval": {"n_repeats": True}}, "must be an integer"),
        ({"eval": {"n_split": 5}}, "unknown config keys under 'eval'"),
        ({"evaluation": {"n_splits": 5}}, "unknown config keys"),
        ({"eval": [5, 10]}, "must be a mapping"),
        ({"seed": 0, "eval": {"seed": 1}}, "disagree"),
        ({"persistent_error_threshold": 0.9}, "not configurable"),
    ],
)
def test_invalid_config_rejected(cfg, match):
    with pytest.raises(ConfigError, match=match):
        _validate(cfg)


@pytest.mark.parametrize("flag, value", [("--n-splits", "1"), ("--n-repeats", "0")])
def test_invalid_cli_values_rejected(flag, value):
    with pytest.raises(ConfigError):
        _resolve(**{flag.lstrip("-").replace("-", "_"): int(value)})


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_cli_invalid_config_fails_before_creating_output(tmp_path):
    cfg = _write_config(tmp_path / "bad.yaml", "smoke: false\neval:\n  n_splits: 1\n  n_repeats: 2\n")
    out = tmp_path / "never_created"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(CSV), str(out), "--config", str(cfg)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "eval.n_splits must be >= 2" in proc.stderr
    assert not out.exists()


def _run_cli(tmp_path: Path, out: Path, *extra: str) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), str(CSV), str(out), *extra],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def _assert_effective_cv(out: Path, res: dict, n_splits: int, n_repeats: int) -> None:
    assert res["config"]["eval"]["n_splits"] == n_splits
    assert res["config"]["eval"]["n_repeats"] == n_repeats
    folds = pd.read_csv(out / "folds.csv")
    valid = folds[folds.role == "valid"]
    assert sorted(valid.repeat.unique()) == list(range(n_repeats))
    assert sorted(valid.fold.unique()) == list(range(n_splits))
    fold_scores = pd.read_csv(out / "fold_scores.csv")
    assert fold_scores.groupby("model").size().eq(n_splits * n_repeats).all()
    repeat_scores = pd.read_csv(out / "repeat_scores.csv")
    assert repeat_scores.groupby("model").size().eq(n_repeats).all()
    assert res["models"]["lr_all"]["n_folds"] == n_splits * n_repeats
    assert res["models_repeat"]["lr_all"]["n_repeats"] == n_repeats


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_cli_config_eval_values_take_effect(tmp_path):
    cfg = _write_config(
        tmp_path / "cfg.yaml",
        "smoke: false\nn_perm: 2\nseed: 0\neval:\n  n_splits: 2\n  n_repeats: 2\n  seed: 0\n",
    )
    out = tmp_path / "cfg_run"
    res = _run_cli(tmp_path, out, "--config", str(cfg))
    assert res["run_mode"] == "full"
    _assert_effective_cv(out, res, n_splits=2, n_repeats=2)
    resolution = res["config"]["settings_resolution"]
    assert resolution["eval_sources"] == {
        "n_splits": "config:eval.n_splits",
        "n_repeats": "config:eval.n_repeats",
    }
    assert resolution["eval_notes"] == []
    assert resolution["config_file"]["sha256_normalized_lf"]


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_cli_smoke_with_config_uses_smoke_defaults_and_records_why(tmp_path):
    out = tmp_path / "smoke_cfg"
    res = _run_cli(tmp_path, out, "--smoke", "--config", str(ROOT / "configs" / "audit_primary.yaml"))
    _assert_effective_cv(out, res, n_splits=3, n_repeats=1)
    resolution = res["config"]["settings_resolution"]
    assert resolution["eval_sources"] == {"n_splits": "smoke_default", "n_repeats": "smoke_default"}
    assert len(resolution["eval_notes"]) == 2


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_cli_flags_override_smoke_and_config(tmp_path):
    out = tmp_path / "smoke_cli"
    res = _run_cli(
        tmp_path,
        out,
        "--smoke",
        "--config",
        str(ROOT / "configs" / "audit_primary.yaml"),
        "--n-splits",
        "2",
        "--n-repeats",
        "2",
    )
    _assert_effective_cv(out, res, n_splits=2, n_repeats=2)
    assert res["config"]["settings_resolution"]["eval_sources"] == {"n_splits": "cli", "n_repeats": "cli"}
