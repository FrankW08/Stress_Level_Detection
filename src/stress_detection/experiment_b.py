"""Experiment B nested-search *design* only — this module does not run the search."""

from __future__ import annotations

from typing import Any

from stress_detection.cv_protocol import PROTOCOL_LABELS

PRIMARY_METRIC = "macro_f1"
INNER_SPLITS = 3
OUTER_SPLITS = 5
OUTER_REPEATS = 10
N_OUTER_FOLDS = OUTER_SPLITS * OUTER_REPEATS
SEED = 0

# Shared with Experiment A outer protocol. Do not change these when implementing B.
SHARED_OUTER = {
    "splitter": "RepeatedStratifiedKFold",
    "n_splits": OUTER_SPLITS,
    "n_repeats": OUTER_REPEATS,
    "seed": SEED,
    "note": "All Experiment B candidates must use the same outer splits as Experiment A primary.",
}

# Limited grids. XGBoost / deep models are not in the default B design.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "lr": {
        "logisticregression__C": [0.1, 1.0, 10.0],
    },
    "svm": {
        "svc__C": [0.05, 0.2, 0.8],
    },
    "rf": {
        "randomforestclassifier__n_estimators": [100, 300],
        "randomforestclassifier__max_depth": [None, 8],
        "randomforestclassifier__min_samples_leaf": [1, 4],
    },
}

PARAM_RATIONALE = {
    "logisticregression__C": (
        "Inverse L2 regularization strength. Smaller C shrinks coefficients more; "
        "the Experiment A default is sklearn's C=1.0 after scaling."
    ),
    "svc__C": (
        "Soft-margin penalty. Experiment A used C=0.2; the grid brackets that value."
    ),
    "randomforestclassifier__n_estimators": "Number of trees; 100 is the Experiment A default.",
    "randomforestclassifier__max_depth": (
        "None grows until leaves are pure (Experiment A default); 8 is a shallower regularizer."
    ),
    "randomforestclassifier__min_samples_leaf": "Larger values reduce variance on small leaves.",
}

FIXED_BASELINES = ["dummy", "nested_single", "lr_all", "svm_all", "rf_all"]

PRE_SPECIFIED_COMPARISONS = [
    ("b_lr", "dummy"),
    ("b_lr", "nested_single"),
    ("b_lr", "lr_all"),
    ("b_best", "lr_all"),
]

TIE_BREAK = (
    "Highest inner-CV mean macro-F1 with labels [0,1,2] and zero_division=0; "
    "ties broken by smaller search-index (grid iteration order), then smaller C / "
    "smaller n_estimators / shallower max_depth (None last) / larger min_samples_leaf."
)


def n_candidates(model: str) -> int:
    grid = PARAM_GRIDS[model]
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


def inner_fit_count(model: str) -> int:
    """Inner GridSearch fits (not counting the outer-train refit)."""
    return n_candidates(model) * INNER_SPLITS * N_OUTER_FOLDS


def refit_count() -> int:
    return N_OUTER_FOLDS


def budget_summary() -> dict[str, Any]:
    inner = {m: inner_fit_count(m) for m in PARAM_GRIDS}
    refits = {m: refit_count() for m in PARAM_GRIDS}
    total_inner = sum(inner.values())
    total_refit = sum(refits.values())
    return {
        "n_outer_folds": N_OUTER_FOLDS,
        "inner_splits": INNER_SPLITS,
        "n_candidates": {m: n_candidates(m) for m in PARAM_GRIDS},
        "inner_cv_fits": inner,
        "outer_train_refits": refits,
        "total_search_fits": total_inner + total_refit,
        "final_full_data_fits_not_for_oof": 3,
        "note": (
            "Counts are pipeline.fit calls inside nested CV. "
            "Final full-data fits are artifacts only and must not be reported as OOF performance. "
            "This design is not executed by the current scripts."
        ),
    }


def experiment_b_status() -> dict[str, Any]:
    return {
        "status": "design_only_not_executed",
        "inner_splits": INNER_SPLITS,
        "shared_outer": SHARED_OUTER,
        "param_grids": PARAM_GRIDS,
        "param_rationale": PARAM_RATIONALE,
        "fixed_baselines_from_experiment_a": FIXED_BASELINES,
        "pre_specified_comparisons": PRE_SPECIFIED_COMPARISONS,
        "selection_metric": PRIMARY_METRIC,
        "labels": list(PROTOCOL_LABELS),
        "zero_division": 0,
        "tie_break": TIE_BREAK,
        "grouped_mode": (
            "If run under grouped_duplicates, outer and inner splitters must be "
            "StratifiedGroupKFold with group isolation and 3-class coverage; "
            "do not fall back to ungrouped CV."
        ),
        "budget": budget_summary(),
        "artifacts_required": [
            "outer splits (repeat, fold, source_row_id, role)",
            "inner splits per outer fold",
            "inner candidate scores for every grid point",
            "selected params per outer fold",
            "OOF predictions",
            "environment lock and code manifest",
        ],
    }
