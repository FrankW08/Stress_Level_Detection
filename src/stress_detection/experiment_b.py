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

SHARED_OUTER = {
    "splitter": "RepeatedStratifiedKFold",
    "n_splits": OUTER_SPLITS,
    "n_repeats": OUTER_REPEATS,
    "seed": SEED,
    "note": (
        "All Experiment B procedures and Experiment A baselines must use the same "
        "outer splits and the same included rows as Experiment A primary. "
        "Do not retune the outer seed after seeing fold scores."
    ),
}

# Family order is the cross-family tie-break. XGBoost / deep models are not included.
FAMILY_ORDER: tuple[str, ...] = ("lr", "svm", "rf")

# Explicit candidate lists. Position in this list is the only tie-break after inner macro-F1.
# Do not add a second "stronger regularization" rule; it would never apply after list order.
CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "lr": [
        {"logisticregression__C": 0.1},
        {"logisticregression__C": 1.0},
        {"logisticregression__C": 10.0},
    ],
    "svm": [
        {"svc__C": 0.05},
        {"svc__C": 0.2},
        {"svc__C": 0.8},
    ],
    "rf": [
        {
            "randomforestclassifier__n_estimators": 100,
            "randomforestclassifier__max_depth": 8,
            "randomforestclassifier__min_samples_leaf": 4,
        },
        {
            "randomforestclassifier__n_estimators": 100,
            "randomforestclassifier__max_depth": 8,
            "randomforestclassifier__min_samples_leaf": 1,
        },
        {
            "randomforestclassifier__n_estimators": 100,
            "randomforestclassifier__max_depth": None,
            "randomforestclassifier__min_samples_leaf": 4,
        },
        {
            "randomforestclassifier__n_estimators": 100,
            "randomforestclassifier__max_depth": None,
            "randomforestclassifier__min_samples_leaf": 1,
        },
        {
            "randomforestclassifier__n_estimators": 300,
            "randomforestclassifier__max_depth": 8,
            "randomforestclassifier__min_samples_leaf": 4,
        },
        {
            "randomforestclassifier__n_estimators": 300,
            "randomforestclassifier__max_depth": 8,
            "randomforestclassifier__min_samples_leaf": 1,
        },
        {
            "randomforestclassifier__n_estimators": 300,
            "randomforestclassifier__max_depth": None,
            "randomforestclassifier__min_samples_leaf": 4,
        },
        {
            "randomforestclassifier__n_estimators": 300,
            "randomforestclassifier__max_depth": None,
            "randomforestclassifier__min_samples_leaf": 1,
        },
    ],
}

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "lr": {"logisticregression__C": [0.1, 1.0, 10.0]},
    "svm": {"svc__C": [0.05, 0.2, 0.8]},
    "rf": {
        "randomforestclassifier__n_estimators": [100, 300],
        "randomforestclassifier__max_depth": [None, 8],
        "randomforestclassifier__min_samples_leaf": [1, 4],
    },
}

PARAM_RATIONALE = {
    "logisticregression__C": (
        "Inverse L2 strength. Smaller C shrinks coefficients more. "
        "Experiment A uses sklearn's C=1.0 after scaling. "
        "Tie-break lists C=0.1 before 1.0 before 10.0 (stronger L2 first)."
    ),
    "svc__C": (
        "Soft-margin penalty. Experiment A used C=0.2. "
        "Tie-break lists C=0.05 before 0.2 before 0.8."
    ),
    "randomforestclassifier__n_estimators": (
        "Number of trees; 100 is the Experiment A default. "
        "Fewer trees is a compute-cost preference in the tie-break list, "
        "not a claim of stronger regularization."
    ),
    "randomforestclassifier__max_depth": (
        "8 is a shallower tree than unrestricted None (Experiment A default)."
    ),
    "randomforestclassifier__min_samples_leaf": (
        "Larger leaves (4 before 1 in the tie-break list) reduce variance on small leaves."
    ),
}

FIXED_BASELINES = ["dummy", "nested_single", "lr_all", "svm_all", "rf_all"]

# Pre-declared comparison set (four pairs). Not a winner-picked-from-outer-scores comparison.
PRE_SPECIFIED_COMPARISONS = [
    ("b_lr", "dummy"),
    ("b_lr", "nested_single"),
    ("b_lr", "lr_all"),
    ("b_select", "lr_all"),
]

SELECTION_RULES = {
    "metric": PRIMARY_METRIC,
    "labels": list(PROTOCOL_LABELS),
    "zero_division": 0,
    "where": "inner training folds of the current outer training set only",
    "outer_validation": (
        "Used once after selection for OOF. Must not choose family, hyperparameters, "
        "thresholds, features, or search-space expansions."
    ),
    "b_lr": (
        "Within the LR family only, pick the candidate with the highest inner-CV mean "
        "macro-F1. Ties: earlier entry in CANDIDATES['lr']."
    ),
    "b_select": (
        "On each outer training fold, using the same inner splits as the family searches, "
        "pick among all LR, linear SVM, and RF candidates. Different outer folds may "
        "select different families. Ties: earlier entry in candidate_priority_list()."
    ),
    "not_a_comparison": (
        "Ranking families by pooled outer scores (e.g. 'the family with the highest "
        "mean outer macro-F1') is exploratory only. A paired interval between that "
        "post-hoc family and lr_all does not correct for winner selection."
    ),
}

TIE_BREAK = (
    "1. Higher inner-CV mean macro-F1 (labels [0,1,2], zero_division=0) wins. "
    "2. If still tied, the earlier candidate in the pre-specified list wins. "
    "Family order: lr, then svm, then rf. Within each family the order is CANDIDATES[family]. "
    "There is no second-pass regularization rule after this list."
)


def n_candidates(model: str) -> int:
    return len(CANDIDATES[model])


def candidate_priority_list() -> list[dict[str, Any]]:
    """Single ordered list used for b_select ties (and concatenated family lists)."""
    out: list[dict[str, Any]] = []
    priority = 0
    for family in FAMILY_ORDER:
        for local_i, params in enumerate(CANDIDATES[family]):
            out.append(
                {
                    "priority": priority,
                    "family": family,
                    "family_index": local_i,
                    "params": params,
                }
            )
            priority += 1
    return out


def pick_by_inner_scores(
    scores: list[tuple[int, float]],
) -> int:
    """Return the winning priority index.

    scores: (priority, inner_macro_f1) pairs. Highest F1 wins; ties take smaller priority.
    """
    if not scores:
        raise ValueError("no candidates to select")
    return max(scores, key=lambda t: (t[1], -t[0]))[0]


def family_priority_indices(family: str) -> list[int]:
    return [c["priority"] for c in candidate_priority_list() if c["family"] == family]


def inner_fit_count(model: str) -> int:
    return n_candidates(model) * INNER_SPLITS * N_OUTER_FOLDS


def family_refit_count() -> int:
    return N_OUTER_FOLDS


def budget_summary() -> dict[str, Any]:
    n_lr, n_svm, n_rf = (n_candidates(m) for m in FAMILY_ORDER)
    n_all = n_lr + n_svm + n_rf
    inner = {m: inner_fit_count(m) for m in FAMILY_ORDER}
    total_inner = n_all * INNER_SPLITS * N_OUTER_FOLDS
    family_refits = {m: family_refit_count() for m in FAMILY_ORDER}
    total_family_refit = len(FAMILY_ORDER) * N_OUTER_FOLDS
    return {
        "n_outer_folds": N_OUTER_FOLDS,
        "inner_splits": INNER_SPLITS,
        "n_candidates": {m: n_candidates(m) for m in FAMILY_ORDER},
        "inner_cv_fits": inner,
        "inner_cv_fits_total": total_inner,
        "inner_cv_fits_total_formula": f"({n_lr}+{n_svm}+{n_rf}) x {INNER_SPLITS} x {N_OUTER_FOLDS} = {total_inner}",
        "family_outer_train_refits": family_refits,
        "family_outer_train_refits_total": total_family_refit,
        "family_outer_train_refits_formula": (
            f"{len(FAMILY_ORDER)} families x {N_OUTER_FOLDS} = {total_family_refit}"
        ),
        "search_plus_family_refit_total": total_inner + total_family_refit,
        "search_plus_family_refit_total_note": (
            f"{total_inner} inner fits + {total_family_refit} family outer-train refits "
            f"= {total_inner + total_family_refit}. This is not '{total_inner + total_family_refit} inner fits'."
        ),
        "b_select_reuse": {
            "inner_scores": (
                "Reuse the inner-CV mean for every candidate already scored on that "
                "outer training fold. Do not rescore the same (data, splits, params, seed, fit)."
            ),
            "outer_refit_and_oof": (
                "b_select's pick is always one of the three family-level winners when "
                "the same inner scores and the same tie-break list are used. Reuse that "
                "family's outer-train refit and OOF predictions. Extra b_select refits "
                "are 0 under those conditions."
            ),
            "reuse_allowed_only_if": (
                "Included rows, outer/inner membership, candidate params, seed, "
                "preprocessor, estimator class, and scoring are identical."
            ),
            "b_select_extra_outer_refit_if_reuse_holds": 0,
            "b_select_extra_outer_refit_if_reuse_impossible": N_OUTER_FOLDS,
        },
        "costs_not_in_search_plus_family_refit_total": {
            "uncached_experiment_a_baselines": (
                "If A OOF is not reused from a matching primary run: one outer-train "
                "fit per A model per outer fold (dummy, lr_all, svm_all, rf_all, and "
                "any other A model retained for comparison)."
            ),
            "nested_single_inner_fits": (
                "Experiment A nested-single: inner 3-fold over each feature on each "
                "outer training fold, plus one outer-train refit of the selected tree. "
                "Not part of the LR/SVM/RF 2100+150 budget."
            ),
            "final_full_data_artifact_fits": (
                "Optional fits of frozen selected pipelines on all included rows after "
                "OOF is complete. Must not retune from outer validation scores. "
                "Must not be reported as performance."
            ),
            "b_select_extra_refit_if_reuse_fails": (
                "Up to 50 extra outer-train fits only if family-winner predictions "
                "cannot be reused."
            ),
        },
        "note": (
            "Counts are pipeline.fit calls. This design is not executed by current scripts."
        ),
    }


def experiment_b_status() -> dict[str, Any]:
    return {
        "status": "design_only_not_executed",
        "inner_splits": INNER_SPLITS,
        "shared_outer": SHARED_OUTER,
        "family_order": list(FAMILY_ORDER),
        "candidates": CANDIDATES,
        "candidate_priority_list": candidate_priority_list(),
        "param_grids": PARAM_GRIDS,
        "param_rationale": PARAM_RATIONALE,
        "procedures": {
            "b_lr": SELECTION_RULES["b_lr"],
            "b_select": SELECTION_RULES["b_select"],
        },
        "fixed_baselines_from_experiment_a": FIXED_BASELINES,
        "pre_specified_comparisons": PRE_SPECIFIED_COMPARISONS,
        "pre_specified_comparisons_note": (
            "These four pairs are one declared comparison set. Per-pair Nadeau–Bengio "
            "intervals are not simultaneous confidence intervals. Excluding 0 in any "
            "unadjusted interval does not confirm the whole set. Do not add extra "
            "significance tests in this design. This CSV has already been explored; "
            "the design is not a prospective registration on untouched data."
        ),
        "exploratory_not_pre_specified": SELECTION_RULES["not_a_comparison"],
        "selection_metric": PRIMARY_METRIC,
        "labels": list(PROTOCOL_LABELS),
        "zero_division": 0,
        "selection_rules": SELECTION_RULES,
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
            "inner candidate scores for every grid point (family, params, mean macro-F1)",
            "selected family and params per outer fold for b_lr and for b_select",
            "OOF predictions for b_lr, b_select, and shared A baselines",
            "environment lock and code manifest",
        ],
    }
