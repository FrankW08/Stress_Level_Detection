"""Unit tests for data parsing and prepare_data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_detection.data import (
    apply_label_driven_zero_map,
    normalize_cell,
    parse_numeric_series,
    prepare_data,
    stress_label_exclusion_reason,
)
from stress_detection.hashes import normalize_newlines_to_lf, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


def test_curly_quote_parses_to_number():
    s = pd.Series(['"14"', "“16”", "0", " 3 "])
    out = parse_numeric_series(s)
    assert out.iloc[0] == 14
    assert out.iloc[1] == 16
    assert out.iloc[2] == 0
    assert out.iloc[3] == 3


def test_normalize_cell_actions():
    v, actions = normalize_cell("“16”")
    assert v == "16"
    assert "removed_outer_quotes" in actions
    v2, actions2 = normalize_cell("Null")
    assert v2 is None
    assert "null_token_to_missing" in actions2


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
    assert prep.csv_sha256_normalized_lf == "05145e0a27395e85f6ed062d6f89f99351bdd16f8fa7dc5e60243bd1083dab26"


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


def test_lf_crlf_normalized_hash_equal():
    lf = b"a,b\n1,2\n"
    crlf = b"a,b\r\n1,2\r\n"
    assert sha256_bytes(normalize_newlines_to_lf(lf)) == sha256_bytes(
        normalize_newlines_to_lf(crlf)
    )
