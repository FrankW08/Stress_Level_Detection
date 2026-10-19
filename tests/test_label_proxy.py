"""Label-proxy check: crosstab, purity and LR variants on the primary scope."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stress_detection.data import prepare_data
from stress_detection.label_proxy import crosstab, feature_label_purity, label_proxy_report

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


def test_blood_pressure_codes_1_and_2_map_to_single_labels():
    prep = prepare_data(CSV, "primary")
    table = crosstab(prep.X, prep.y)
    # crosstab drops missing codes, so it covers every row with a recorded value
    assert int(table.to_numpy().sum()) == int(prep.X["blood_pressure"].notna().sum())
    # code 1 only ever appears with label 1; code 2 only with label 0
    assert (table.loc[1.0].to_numpy() > 0).sum() == 1 and table.loc[1.0, 1] > 0
    assert (table.loc[2.0].to_numpy() > 0).sum() == 1 and table.loc[2.0, 0] > 0


def test_purity_is_a_share_per_feature():
    prep = prepare_data(CSV, "primary")
    purity = feature_label_purity(prep.X, prep.y)
    assert set(purity["feature"]) == set(prep.X.columns)
    assert purity["pure_row_share"].between(0, 1).all()


def test_report_variants_use_requested_columns():
    prep = prepare_data(CSV, "primary")
    rep = label_proxy_report(
        prep.X, prep.y, strong_single_features=["blood_pressure", "depression"], n_splits=3, n_repeats=2, seed=0
    )
    v = rep["lr_variants"]
    n = prep.X.shape[1]
    assert v["lr_all"]["n_features"] == n
    assert v["lr_blood_pressure_only"]["n_features"] == 1
    assert v["lr_without_blood_pressure"]["n_features"] == n - 1
    assert v["lr_without_strong_single_features"]["n_features"] == n - 2
    assert all(0 <= x["mean"] <= 1 and np.isfinite(x["std"]) for x in v.values())
