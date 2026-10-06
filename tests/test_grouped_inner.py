"""Grouped sensitivity must isolate groups in outer and inner CV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier

from stress_detection.evaluation import EvalConfig, run_audit_cv
from stress_detection.models import NestedBestSingleFeature, grouped_inner_splits

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


def _synthetic_grouped(n_groups: int = 12) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    y = []
    groups = []
    ids = []
    rid = 0
    for g in range(n_groups):
        label = g % 3
        feat = float(g)
        for _ in range(2):
            rows.append({"f1": feat, "f2": feat + 0.1, "f3": 1.0})
            y.append(label)
            groups.append(g)
            ids.append(rid)
            rid += 1
    X = pd.DataFrame(rows)
    return X, np.array(y), np.array(ids), np.array(groups)


def test_grouped_outer_and_inner_groups_disjoint():
    X, y, ids, groups = _synthetic_grouped()
    models = {
        "nested_single": (
            lambda: NestedBestSingleFeature(random_state=0, inner_splits=2),
            list(X.columns),
        ),
        "dummy": (lambda: DummyClassifier(strategy="most_frequent"), list(X.columns)),
    }
    cfg = EvalConfig(n_splits=2, n_repeats=1, seed=0)
    out = run_audit_cv(X, y, ids, models, cfg, groups=groups)
    folds = out["folds"]
    inner = out["inner_folds"]
    assert "group_id" in folds.columns
    for (rep, fold), g in folds.groupby(["repeat", "fold"]):
        tr = set(g.loc[g.role == "train", "group_id"])
        te = set(g.loc[g.role == "valid", "group_id"])
        assert tr.isdisjoint(te)
    assert not inner.empty
    outer_train_ids = {}
    for (rep, fold), g in folds.groupby(["repeat", "fold"]):
        outer_train_ids[(rep, fold)] = set(g.loc[g.role == "train", "source_row_id"])
    for (rep, fold, inner_fold), g in inner.groupby(["repeat", "fold", "inner_fold"]):
        assert set(g.source_row_id).issubset(outer_train_ids[(rep, fold)])
        tr = set(g.loc[g.role == "train", "group_id"])
        te = set(g.loc[g.role == "valid", "group_id"])
        assert tr.isdisjoint(te)


def test_grouped_inner_refuses_ungrouped_fallback():
    X = pd.DataFrame({"f": [0.0, 0.0, 1.0, 1.0]})
    y = np.array([0, 0, 1, 1])
    groups = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="refusing to fall back"):
        grouped_inner_splits(X, y, groups, n_splits=3, random_state=0)


def test_groups_none_nested_matches_previous_algorithm():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f1": rng.integers(0, 5, size=30).astype(float),
            "f2": rng.integers(0, 5, size=30).astype(float),
            "f3": rng.integers(0, 5, size=30).astype(float),
        }
    )
    y = np.array([0, 1, 2] * 10)
    a = NestedBestSingleFeature(random_state=0, inner_splits=3).fit(X, y)
    b = NestedBestSingleFeature(random_state=0, inner_splits=3).fit(X, y, groups=None)
    assert a.feature_ == b.feature_
    assert a.candidates_ == b.candidates_


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_grouped_sensitivity_smoke_skips_permutation_and_writes_inner_folds(tmp_path):
    from stress_detection.audit import run_full_audit

    out = tmp_path / "grouped"
    res = run_full_audit(
        CSV,
        out,
        smoke=True,
        seed=0,
        scope="sensitivity",
        sensitivity_policy="grouped_duplicates",
        project_root=ROOT,
    )
    assert res["permutation"]["status"] == "not_available"
    assert "exchangeability" in res["permutation"]["reason"]
    assert not (out / "permutations.csv").exists()
    inner = pd.read_csv(out / "inner_folds.csv")
    folds = pd.read_csv(out / "folds.csv")
    for (rep, fold), g in folds.groupby(["repeat", "fold"]):
        assert set(g.loc[g.role == "train", "group_id"]).isdisjoint(
            set(g.loc[g.role == "valid", "group_id"])
        )
    for (rep, fold, inner_fold), g in inner.groupby(["repeat", "fold", "inner_fold"]):
        tr = set(g.loc[g.role == "train", "group_id"])
        te = set(g.loc[g.role == "valid", "group_id"])
        assert tr.isdisjoint(te)
    pe = res["persistent_errors"]["feature_description_same_data"]
    assert pe["status"] == "not_available"
    json.dumps(res["permutation"])
