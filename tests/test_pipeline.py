
import json
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "lab7"


def test_preprocessed_files_exist():
    required = [
        DATA_DIR / "lab7_X_train.npy",
        DATA_DIR / "lab7_X_test.npy",
        DATA_DIR / "lab7_y_train.npy",
        DATA_DIR / "lab7_y_test.npy",
        DATA_DIR / "lab7_metadata.json",
    ]

    for path in required:
        assert path.is_file(), f"Missing file: {path}"


def test_train_test_dimensions_match():
    X_train = np.load(DATA_DIR / "lab7_X_train.npy")
    y_train = np.load(DATA_DIR / "lab7_y_train.npy")
    X_test = np.load(DATA_DIR / "lab7_X_test.npy")
    y_test = np.load(DATA_DIR / "lab7_y_test.npy")

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == 10
    assert X_test.shape[1] == 10


def test_metadata_is_valid():
    with open(DATA_DIR / "lab7_metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["dataset"] == "Walmart.csv"
    assert metadata["target"] == "Weekly_Sales"
    assert metadata["training_rows"] == 5148
    assert metadata["testing_rows"] == 1287


def test_trained_model_can_predict():
    model_path = ARTIFACT_DIR / "walmart_random_forest.joblib"
    X_test_path = DATA_DIR / "lab7_X_test.npy"

    assert model_path.is_file(), "Trained model is missing"

    model = joblib.load(model_path)
    X_test = np.load(X_test_path)

    predictions = model.predict(X_test[:5])

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()
