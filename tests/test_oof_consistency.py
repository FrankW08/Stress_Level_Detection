"""Saved OOF predictions must reproduce repeat- and fold-level summaries independently."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from stress_detection import audit as audit_mod
from stress_detection.data import prepare_data
from stress_detection.evaluation import repeat_oof_scores, summarize_fold_scores

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"
LABELS = [0, 1, 2]
MODELS = ["dummy", "lr_all", "rf_all", "svm_all", "lr_no_psych", "nested_single"]
N_SPLITS, N_REPEATS = 3, 2


def _sk_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Independent reference built only from sklearn."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "cm": cm,
        "cross_num": int(cm[0, 2] + cm[2, 0]),
        "cross_den": int(np.sum(y_true == 0) + np.sum(y_true == 2)),
    }
    for i, lab in enumerate(LABELS):
        out[f"class_{lab}_precision"] = p[i]
        out[f"class_{lab}_recall"] = r[i]
        out[f"class_{lab}_f1"] = f[i]
        out[f"class_{lab}_support"] = s[i]
    out["cross_rate"] = out["cross_num"] / out["cross_den"]
    return out


@pytest.fixture(scope="module")
def audit_run(tmp_path_factory):
    if not CSV.is_file():
        pytest.skip("dataset missing")
    out = tmp_path_factory.mktemp("oof_audit") / "run"
    audit_mod.run_full_audit(
        CSV,
        out,
        smoke=True,
        seed=0,
        n_perm=2,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        project_root=ROOT,
    )
    prep = prepare_data(CSV, scope="primary")
    return {
        "out": out,
        "res": json.loads((out / "results.json").read_text(encoding="utf-8")),
        "oof": pd.read_csv(out / "oof_predictions.csv"),
        "repeat_scores": pd.read_csv(out / "repeat_scores.csv"),
        "fold_scores": pd.read_csv(out / "fold_scores.csv"),
        "folds": pd.read_csv(out / "folds.csv"),
        "labels": pd.Series(prep.y, index=prep.source_row_ids),
    }


def test_oof_coverage_per_repeat(audit_run):
    oof, labels = audit_run["oof"], audit_run["labels"]
    assert sorted(oof["repeat"].unique()) == list(range(N_REPEATS))
    for rep, g in oof.groupby("repeat"):
        ids = g["source_row_id"]
        assert ids.is_unique, f"repeat {rep}: duplicated source_row_id"
        assert set(ids) == set(labels.index), f"repeat {rep}: missing/extra rows"
        assert len(g) == len(labels)
        assert (g["true"].to_numpy() == labels.loc[ids].to_numpy()).all()
        for m in MODELS:
            assert g[m].notna().all()
            assert set(g[m].unique()) <= set(LABELS), f"{m} repeat {rep}: illegal class"


def test_repeat_scores_match_independent_recompute(audit_run):
    oof, rs = audit_run["oof"], audit_run["repeat_scores"]
    n_total = len(audit_run["labels"])
    assert len(rs) == len(MODELS) * N_REPEATS
    for m in MODELS:
        for rep in range(N_REPEATS):
            g = oof[oof["repeat"] == rep]
            ref = _sk_metrics(g["true"].to_numpy(), g[m].to_numpy())
            row = rs[(rs["model"] == m) & (rs["repeat"] == rep)]
            assert len(row) == 1
            row = row.iloc[0]
            assert row["n_samples"] == n_total
            for key in ("accuracy", "macro_f1"):
                assert row[key] == pytest.approx(ref[key], abs=1e-12)
            for lab in LABELS:
                for k in ("precision", "recall", "f1", "support"):
                    col = f"class_{lab}_{k}"
                    assert row[col] == pytest.approx(ref[col], abs=1e-12), (m, rep, col)
            cm = np.array(json.loads(row["confusion_matrix"]))
            assert cm.shape == (3, 3)
            assert cm.sum() == n_total
            np.testing.assert_array_equal(cm, ref["cm"])
            np.testing.assert_array_equal(
                cm.sum(axis=1), np.bincount(g["true"], minlength=3)
            )
            assert row["cross_level_0_2_numerator"] == ref["cross_num"]
            assert row["cross_level_0_2_denominator"] == ref["cross_den"]
            assert row["cross_level_0_2_rate"] == pytest.approx(ref["cross_rate"], abs=1e-12)


def test_results_repeat_summary_matches_repeat_scores(audit_run):
    res, oof = audit_run["res"], audit_run["oof"]
    assert res["models_repeat_aggregation"]["unit"] == "repeat"
    for m in MODELS:
        entry = res["models_repeat"][m]
        assert entry["n_repeats"] == N_REPEATS
        assert entry["aggregation_unit"] == "repeat"
        per_rep = [
            _sk_metrics(g["true"].to_numpy(), g[m].to_numpy())
            for _, g in oof.groupby("repeat")
        ]
        for key in ("accuracy", "macro_f1"):
            vals = np.array([d[key] for d in per_rep])
            assert entry["metrics"][key]["mean"] == pytest.approx(vals.mean(), abs=1e-12)
            assert entry["metrics"][key]["std"] == pytest.approx(vals.std(ddof=1), abs=1e-12)
        cms = np.array([d["cm"] for d in per_rep])
        np.testing.assert_array_equal(np.array(entry["confusion_matrix_per_repeat"]), cms)
        np.testing.assert_array_equal(
            np.array(entry["confusion_matrix_sum_over_repeats"]), cms.sum(axis=0)
        )


def test_fold_scores_match_oof_and_fold_summary(audit_run):
    res, oof, fs, folds = (
        audit_run["res"], audit_run["oof"], audit_run["fold_scores"], audit_run["folds"]
    )
    assert res["models_aggregation"]["unit"] == "outer_fold"
    valid = folds[folds["role"] == "valid"]
    for (rep, fold), memb in valid.groupby(["repeat", "fold"]):
        g = oof[oof["repeat"] == rep].set_index("source_row_id").loc[memb["source_row_id"]]
        for m in MODELS:
            row = fs[(fs["model"] == m) & (fs["repeat"] == rep) & (fs["fold"] == fold)].iloc[0]
            y_t, y_p = g["true"].to_numpy(), g[m].to_numpy()
            assert row["macro_f1"] == pytest.approx(f1_score(y_t, y_p, average="macro"), abs=1e-12)
            assert row["accuracy"] == pytest.approx(accuracy_score(y_t, y_p), abs=1e-12)
    for m in MODELS:
        g = fs[fs["model"] == m]
        entry = res["models"][m]
        assert entry["aggregation_unit"] == "outer_fold"
        assert entry["n_folds"] == N_SPLITS * N_REPEATS
        assert entry["macro_f1_mean"] == pytest.approx(g["macro_f1"].mean(), abs=1e-12)
        assert entry["macro_f1_std"] == pytest.approx(g["macro_f1"].std(ddof=1), abs=1e-12)
        # Within each repeat, fold confusion matrices add up to the pooled OOF matrix.
        rep_cms = np.array(res["models_repeat"][m]["confusion_matrix_per_repeat"])
        for rep in range(N_REPEATS):
            fold_cm = sum(
                np.array(json.loads(c))
                for c in g.loc[g["repeat"] == rep, "confusion_matrix"]
            )
            np.testing.assert_array_equal(fold_cm, rep_cms[rep])


def test_single_repeat_std_is_null(tmp_path):
    oof = {"m": np.array([[0, 1, 2, 2]])}
    rs = repeat_oof_scores(oof, np.array([0, 1, 2, 2]))
    from stress_detection.evaluation import summarize_repeat_scores

    entry = summarize_repeat_scores(rs)["m"]
    assert entry["n_repeats"] == 1
    assert entry["metrics"]["macro_f1"]["std"] is None
    assert "cannot be estimated" in entry["std_note"]


def test_pooled_oof_f1_differs_from_fold_mean():
    """Hand-built case where averaging fold macro-F1 gives a different number."""
    y = np.array([0, 1, 2, 2])
    pred = np.array([0, 1, 2, 0])
    folds = [np.array([0, 1]), np.array([2, 3])]
    fold_f1 = [f1_score(y[idx], pred[idx], average="macro") for idx in folds]
    fold_mean = float(np.mean(fold_f1))  # (1.0 + 1/3) / 2 = 0.6667
    pooled = f1_score(y, pred, average="macro")  # (2/3 + 1 + 2/3) / 3 = 0.7778
    assert fold_mean == pytest.approx(2 / 3)
    assert pooled == pytest.approx(7 / 9)

    rs = repeat_oof_scores({"m": pred.reshape(1, -1)}, y)
    assert rs.loc[0, "macro_f1"] == pytest.approx(pooled)
    assert rs.loc[0, "macro_f1"] != pytest.approx(fold_mean)

    # The fold-level summary keeps the fold-mean definition.
    fs = pd.DataFrame(
        [
            {"model": "m", "macro_f1": f, "accuracy": 0.0, "confusion_matrix": [[0] * 3] * 3}
            for f in fold_f1
        ]
    )
    assert summarize_fold_scores(fs)["m"]["macro_f1_mean"] == pytest.approx(fold_mean)


def test_incomplete_oof_rejected():
    with pytest.raises(ValueError, match="no OOF prediction"):
        repeat_oof_scores({"m": np.array([[0, -1, 2]])}, np.array([0, 1, 2]))
