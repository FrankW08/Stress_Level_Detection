"""Budget arithmetic for the unimplemented Experiment B design."""

from stress_detection.experiment_b import budget_summary, n_candidates


def test_experiment_b_budget_matches_design():
    assert n_candidates("lr") == 3
    assert n_candidates("svm") == 3
    assert n_candidates("rf") == 8
    b = budget_summary()
    assert b["inner_cv_fits"]["lr"] == 3 * 3 * 50
    assert b["inner_cv_fits"]["svm"] == 3 * 3 * 50
    assert b["inner_cv_fits"]["rf"] == 8 * 3 * 50
    assert b["total_search_fits"] == 450 + 450 + 1200 + 150
