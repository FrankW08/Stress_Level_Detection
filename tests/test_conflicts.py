"""Feature-group label conflicts and near-duplicate missing-value semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stress_detection.audit import duplicate_audit
from stress_detection.data import (
    assign_duplicate_group_ids,
    feature_values_differ,
    label_conflict_table,
)


def _parsed(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_conflict_same_label_not_reported():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": 1.0, "b": 2.0, "stress_level": 0},
            {"source_row_id": 1, "a": 1.0, "b": 2.0, "stress_level": 0},
        ]
    )
    out = label_conflict_table(parsed)
    assert out.empty


def test_conflict_two_labels_reports_both():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": 1.0, "b": 2.0, "stress_level": 0},
            {"source_row_id": 1, "a": 1.0, "b": 2.0, "stress_level": 1},
        ]
    )
    out = label_conflict_table(parsed)
    assert set(out.source_row_id) == {0, 1}
    assert int(out.n_distinct_valid_labels.iloc[0]) == 2


def test_conflict_three_members_all_reported():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": 1.0, "stress_level": 0},
            {"source_row_id": 1, "a": 1.0, "stress_level": 0},
            {"source_row_id": 2, "a": 1.0, "stress_level": 1},
        ]
    )
    out = label_conflict_table(parsed)
    assert set(out.source_row_id) == {0, 1, 2}
    assert int(out.group_size.iloc[0]) == 3


def test_conflict_balanced_two_two_all_reported():
    parsed = _parsed(
        [
            {"source_row_id": i, "a": 1.0, "stress_level": 0 if i < 2 else 1}
            for i in range(4)
        ]
    )
    out = label_conflict_table(parsed)
    assert set(out.source_row_id) == {0, 1, 2, 3}
    assert "0:2" in out.valid_label_counts.iloc[0]
    assert "1:2" in out.valid_label_counts.iloc[0]


def test_conflict_matching_nan_features_grouped():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": np.nan, "b": 1.0, "stress_level": 0},
            {"source_row_id": 1, "a": np.nan, "b": 1.0, "stress_level": 2},
        ]
    )
    gids = assign_duplicate_group_ids(parsed[["a", "b"]])
    assert gids[0] == gids[1]
    out = label_conflict_table(parsed)
    assert set(out.source_row_id) == {0, 1}


def test_conflict_missing_label_not_a_class_but_member_reported():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": 1.0, "stress_level": 0},
            {"source_row_id": 1, "a": 1.0, "stress_level": 1},
            {"source_row_id": 2, "a": 1.0, "stress_level": np.nan},
        ]
    )
    out = label_conflict_table(parsed)
    assert set(out.source_row_id) == {0, 1, 2}
    assert int(out.n_distinct_valid_labels.iloc[0]) == 2
    assert int(out.n_invalid_or_missing_labels.iloc[0]) == 1


def test_missing_label_only_is_not_conflict():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": 1.0, "stress_level": 0},
            {"source_row_id": 1, "a": 1.0, "stress_level": np.nan},
        ]
    )
    assert label_conflict_table(parsed).empty


def test_different_features_not_merged():
    parsed = _parsed(
        [
            {"source_row_id": 0, "a": 1.0, "stress_level": 0},
            {"source_row_id": 1, "a": 2.0, "stress_level": 1},
        ]
    )
    assert label_conflict_table(parsed).empty


def test_near_dup_nan_pattern_is_not_a_difference():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert int(feature_values_differ(a, b).sum()) == 0
    c = np.array([1.0, np.nan, 4.0])
    assert int(feature_values_differ(a, c).sum()) == 1


def test_duplicate_audit_uses_conflict_table():
    raw = pd.DataFrame({"a": ["1", "1", "1", "1"], "stress_level": ["0", "0", "1", "1"]})
    parsed = _parsed(
        [{"source_row_id": i, "a": 1.0, "stress_level": 0 if i < 2 else 1} for i in range(4)]
    )
    out = duplicate_audit(raw, parsed, include_near=False)
    ids = set(out["same_features_different_label"]["source_row_id"])
    assert ids == {0, 1, 2, 3}
