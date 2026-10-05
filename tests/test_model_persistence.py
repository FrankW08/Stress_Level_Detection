import joblib
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from stress_detection.data import prepare_data
from stress_detection.models import lr_pipeline

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "StressLevelDataset_original.csv"


@pytest.mark.skipif(not CSV.is_file(), reason="dataset missing")
def test_saved_lr_predicts_without_labels(tmp_path):
    prep = prepare_data(CSV, scope="primary")
    pipe: Pipeline = lr_pipeline(random_state=0)
    pipe.fit(prep.X, prep.y)
    path = tmp_path / "lr.joblib"
    joblib.dump({"pipeline": pipe, "feature_names": prep.feature_names}, path)
    loaded = joblib.load(path)
    pred = loaded["pipeline"].predict(prep.X.iloc[:5])
    assert len(pred) == 5
