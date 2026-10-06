"""Budget, selection, and tie-break for the unimplemented Experiment B design."""

from stress_detection.experiment_b import (
    FAMILY_ORDER,
    PRE_SPECIFIED_COMPARISONS,
    budget_summary,
    candidate_priority_list,
    n_candidates,
    pick_by_inner_scores,
)


def test_experiment_b_budget_matches_design():
    assert n_candidates("lr") == 3
    assert n_candidates("svm") == 3
    assert n_candidates("rf") == 8
    b = budget_summary()
    assert b["inner_cv_fits"]["lr"] == 3 * 3 * 50
    assert b["inner_cv_fits"]["svm"] == 3 * 3 * 50
    assert b["inner_cv_fits"]["rf"] == 8 * 3 * 50
    assert b["inner_cv_fits_total"] == 2100
    assert b["family_outer_train_refits_total"] == 150
    assert b["search_plus_family_refit_total"] == 2250
    assert b["b_select_reuse"]["b_select_extra_outer_refit_if_reuse_holds"] == 0


def test_pre_specified_comparisons_are_fixed_not_outer_winner():
    assert PRE_SPECIFIED_COMPARISONS == [
        ("b_lr", "dummy"),
        ("b_lr", "nested_single"),
        ("b_lr", "lr_all"),
        ("b_select", "lr_all"),
    ]
    assert "b_best" not in {a for a, _ in PRE_SPECIFIED_COMPARISONS}


def test_candidate_priority_is_lr_then_svm_then_rf():
    plist = candidate_priority_list()
    assert [c["family"] for c in plist[:3]] == ["lr", "lr", "lr"]
    assert FAMILY_ORDER == ("lr", "svm", "rf")
    assert plist[0]["params"]["logisticregression__C"] == 0.1
    assert plist[-1]["params"]["randomforestclassifier__n_estimators"] == 300
    assert len(plist) == 14


def test_tie_break_uses_list_order_not_a_second_regularization_pass():
    # Equal inner F1: earlier priority wins (LR C=0.1 over LR C=10).
    assert pick_by_inner_scores([(0, 0.8), (2, 0.8)]) == 0
    # Higher F1 wins even if later in the list.
    assert pick_by_inner_scores([(0, 0.7), (5, 0.9)]) == 5
    # Cross-family tie: LR (priority 0-2) before RF (later).
    assert pick_by_inner_scores([(2, 0.81), (6, 0.81)]) == 2
