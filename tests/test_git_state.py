"""Git state is captured at run start, before any artifact is written."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from stress_detection import audit as audit_mod

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"

pytestmark = [
    pytest.mark.skipif(not CSV.is_file(), reason="dataset missing"),
    pytest.mark.skipif(shutil.which("git") is None, reason="git not installed"),
]


@pytest.fixture
def isolated_git_env(tmp_path, monkeypatch):
    """Ignore the developer's global/system git config (identity, autocrlf, hooks)."""
    empty_cfg = tmp_path / "empty_gitconfig"
    empty_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c", "user.name=Audit Test",
            "-c", "user.email=audit-test@example.invalid",
            "-c", "commit.gpgsign=false",
            "-c", "core.autocrlf=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _make_clean_repo(base: Path) -> Path:
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "init")
    assert _git(repo, "status", "--porcelain").strip() == ""
    return repo


def _run_audit_into(repo: Path) -> dict:
    out = repo / "results" / "run_new"
    audit_mod.run_full_audit(
        CSV, out, smoke=True, seed=0, n_perm=2, project_root=repo
    )
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_clean_repo_recorded_clean_despite_new_artifacts(isolated_git_env):
    repo = _make_clean_repo(isolated_git_env)
    head = _git(repo, "rev-parse", "HEAD").strip()
    res = _run_audit_into(repo)
    git = res["git"]
    assert git["git_state_capture"] == "run_start"
    assert git["available"] is True
    assert git["commit"] == head
    assert git["dirty"] is False
    # The artifacts themselves now make the tree dirty; the recorded state must not.
    assert _git(repo, "status", "--porcelain").strip() != ""


def test_modified_tracked_file_recorded_dirty(isolated_git_env):
    repo = _make_clean_repo(isolated_git_env)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    res = _run_audit_into(repo)
    assert res["git"]["dirty"] is True
    assert res["git"]["git_state_capture"] == "run_start"


def test_untracked_file_recorded_dirty(isolated_git_env):
    repo = _make_clean_repo(isolated_git_env)
    (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")
    res = _run_audit_into(repo)
    assert res["git"]["dirty"] is True


def test_git_unavailable_is_not_reported_clean(isolated_git_env, monkeypatch):
    not_a_repo = isolated_git_env / "plain_dir"
    not_a_repo.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(isolated_git_env))
    info = audit_mod.git_info(not_a_repo)
    assert info["available"] is False
    assert info["dirty"] == "unknown"
    assert info["commit"] == "unknown"
