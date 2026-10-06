"""Provisional schema reports extra violations without changing primary inclusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_detection.data import (
    classify_cell_violations,
    domain_violations_table,
    prepare_data,
)
from stress_detection.schema import rule_for

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


def test_mental_health_history_codes():
    rule = rule_for("mental_health_history")
    assert "below_min" in classify_cell_violations("mental_health_history", -1, rule)
    assert "invalid_category" in classify_cell_violations("mental_health_history", -1, rule)
    codes_half = classify_cell_violations("mental_health_history", 0.5, rule)
    assert "non_integer" in codes_half
    assert "invalid_category" in codes_half
    codes_two = classify_cell_violations("mental_health_history", 2, rule)
    assert "above_max" in codes_two
    assert "invalid_category" in codes_two
    assert classify_cell_violations("mental_health_history", 0, rule) == []
    assert classify_cell_violations("mental_health_history", 1, rule) == []


def test_ordinary_field_and_missing():
    rule = rule_for("headache")
    assert "below_min" in classify_cell_violations("headache", -2, rule)
    assert "above_max" in classify_cell_violations("headache", 9, rule)
    assert classify_cell_violations("headache", np.nan, rule) == []
    assert classify_cell_violations("headache", np.inf, rule) == ["non_finite"]


def test_domain_table_counts_cells_and_rules():
    parsed = pd.DataFrame(
        {
            "source_row_id": [0, 1],
            "mental_health_history": [-1.0, 0.5],
            "headache": [0.0, np.nan],
            "stress_level": [0.0, 1.0],
        }
    )
    table = domain_violations_table(parsed)
    assert set(table.violation) >= {"below_min", "invalid_category", "non_integer"}
    # row 0 mental_health_history=-1 triggers below_min + invalid_category
    row0 = table[table.source_row_id == 0]
    assert len(row0) >= 2


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_primary_inclusion_unchanged():
    prep = prepare_data(CSV, scope="primary")
    assert list(prep.source_row_ids[:5]) == [0, 1, 2, 3, 4]
    assert len(prep.y) == 1098
    assert "stress_level" not in prep.X.columns
    assert prep.y.min() == 0 and prep.y.max() == 2
