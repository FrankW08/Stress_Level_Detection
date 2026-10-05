from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_detection.data import (
    apply_label_driven_zero_map,
    parse_numeric_series,
    prepare_data,
    stress_label_exclusion_reason,
)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_curly_quote_parses_to_number():
    s = pd.Series(['"14"', "“16”", "0", " 3 "])
    out = parse_numeric_series(s)
    assert out.iloc[0] == 14
    assert out.iloc[1] == 16
    assert out.iloc[2] == 0
    assert out.iloc[3] == 3


def test_stress_label_rules():
    assert stress_label_exclusion_reason(np.nan) == "missing_stress_level"
    assert stress_label_exclusion_reason(1.5) == "non_integer_stress_level"
    assert stress_label_exclusion_reason(3) == "illegal_stress_level_3"
    assert stress_label_exclusion_reason(1) is None


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_primary_row_count():
    prep = prepare_data(CSV, scope="primary")
    assert len(prep.y) == 1098
    assert "stress_level" not in prep.feature_names
    assert prep.y.min() >= 0 and prep.y.max() <= 2


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_zero_preserved_not_dropped():
    prep = prepare_data(CSV, scope="primary")
    assert (prep.X == 0).any().any()


def test_label_zero_map_changes_zeros():
    X = pd.DataFrame({"headache": [0, 1], "sleep_quality": [1, 0]})
    y = np.array([0, 2])
    Xm = apply_label_driven_zero_map(X, y)
    assert Xm.loc[0, "headache"] == 0
    assert Xm.loc[1, "sleep_quality"] == 4
