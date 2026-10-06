"""3-class protocol: every outer/inner train and valid fold must cover labels 0,1,2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_detection.audit import run_full_audit
from stress_detection.cv_protocol import assert_split_pair, validate_generated_splits
from stress_detection.evaluation import CVPlan, EvalConfig, generate_inner_splits, plan_cv, run_audit_cv
from stress_detection.models import NestedBestSingleFeature, grouped_inner_splits
from stress_detection.run_config import ConfigError

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"
FEATURES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]


def test_handmade_fold_missing_class_rejected():
    y = np.array([0, 0, 1, 1, 2, 2])
    tr = np.array([0, 1, 2, 3])
    te = np.array([4, 5])
    with pytest.raises(ConfigError, match="missing required class"):
        assert_split_pair(tr, te, y, name="outer CV", fold_i=0)


def test_overall_counts_ok_but_a_fold_missing_class_rejected():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    splits = [
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8])),
        (np.array([3, 4, 5, 6, 7, 8]), np.array([0, 1, 2])),
        (np.array([0, 1, 2, 6, 7, 8]), np.array([3, 4, 5])),
    ]
    with pytest.raises(ConfigError, match=r"fold 0 .*missing required class"):
        validate_generated_splits(splits, y, name="outer CV")


def test_pure_class_groups_rejected_by_plan_cv():
    y = np.array([0] * 9 + [1] * 9 + [2] * 9)
    groups = np.array([0] * 9 + [1] * 9 + [2] * 9)
    cfg = EvalConfig(n_splits=3, n_repeats=1, seed=0)
    with pytest.raises(ConfigError, match="missing required class"):
        plan_cv(y, cfg, groups=groups, inner_splits=2)


def test_grouped_inner_pure_classes_rejected():
    n = 9
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    with pytest.raises(ConfigError, match="missing required class|refusing"):
        grouped_inner_splits(X, y, groups, n_splits=3, random_state=0)


def test_legal_grouped_covers_classes_and_isolates_groups():
    rows = []
    y = []
    groups = []
    ids = []
    rid = 0
    for g in range(12):
        label = g % 3
        for _ in range(4):
            rows.append({"f1": float(g), "f2": 1.0, "f3": 2.0})
            y.append(label)
            groups.append(g)
            ids.append(rid)
            rid += 1
    X = pd.DataFrame(rows)
    y = np.array(y)
    groups = np.array(groups)
    ids = np.array(ids)
    cfg = EvalConfig(n_splits=2, n_repeats=1, seed=0)
    plan = plan_cv(y, cfg, groups=groups, inner_splits=2)
    assert isinstance(plan, CVPlan)
    validate_generated_splits(plan.outer, y, groups=groups, name="outer")
    models = {
        "nested_single": (
            lambda: NestedBestSingleFeature(random_state=0, inner_splits=2),
            list(X.columns),
        )
    }
    out = run_audit_cv(
        X,
        y,
        ids,
        models,
        cfg,
        groups=groups,
        outer=plan.outer,
        inner_by_outer=plan.inner,
    )
    folds = out["folds"]
    inner = out["inner_folds"]
    outer_train = {}
    for (rep, fold), g in folds.groupby(["repeat", "fold"]):
        tr = g.loc[g.role == "train"]
        te = g.loc[g.role == "valid"]
        assert set(tr.group_id).isdisjoint(set(te.group_id))
        for part in (tr, te):
            assert set(y[np.isin(ids, part.source_row_id.tolist())]) >= {0, 1, 2}
        outer_train[(rep, fold)] = set(tr.source_row_id)
    for (rep, fold, _i), g in inner.groupby(["repeat", "fold", "inner_fold"]):
        assert set(g.source_row_id).issubset(outer_train[(rep, fold)])
        assert set(g.loc[g.role == "train", "group_id"]).isdisjoint(
            set(g.loc[g.role == "valid", "group_id"])
        )


def test_ungrouped_inner_generation_matches_estimator():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f1": rng.integers(0, 5, size=30).astype(float),
            "f2": rng.integers(0, 5, size=30).astype(float),
            "f3": rng.integers(0, 5, size=30).astype(float),
        }
    )
    y = np.array([0, 1, 2] * 10)
    planned = generate_inner_splits(len(y), y, n_splits=3, random_state=0)
    est = NestedBestSingleFeature(random_state=0, inner_splits=3).fit(X, y)
    for (a, b), (c, d) in zip(planned, est.inner_splits_used_):
        assert np.array_equal(a, c) and np.array_equal(b, d)
    reused = NestedBestSingleFeature(random_state=0, inner_splits=3).fit(X, y, inner_cv=planned)
    assert reused.feature_ == est.feature_
    assert reused.candidates_ == est.candidates_


def _tiny_pure_class_csv(path: Path) -> Path:
    rows = []
    for label in (0, 1, 2):
        for _ in range(4):
            rec = {c: int(label) for c in FEATURES}
            rec["stress_level"] = label
            rows.append(rec)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_failed_grouped_plan_does_not_create_output_dir(tmp_path):
    csv = _tiny_pure_class_csv(tmp_path / "tiny.csv")
    out = tmp_path / "must_not_exist"
    with pytest.raises(ConfigError, match="missing required class"):
        run_full_audit(
            csv,
            out,
            scope="sensitivity",
            sensitivity_policy="grouped_duplicates",
            smoke=True,
            seed=0,
            n_perm=1,
            project_root=ROOT,
        )
    assert not out.exists()


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_primary_plan_covers_three_classes():
    from stress_detection.data import prepare_data

    prep = prepare_data(CSV, scope="primary")
    cfg = EvalConfig(n_splits=5, n_repeats=10, seed=0)
    plan = plan_cv(prep.y, cfg, groups=None, inner_splits=3)
    assert len(plan.outer) == 50
    validate_generated_splits(plan.outer, prep.y, groups=None, name="primary outer")
    for inner, (tr, _) in zip(plan.inner, plan.outer):
        validate_generated_splits(inner, prep.y[tr], groups=None, name="primary inner")
