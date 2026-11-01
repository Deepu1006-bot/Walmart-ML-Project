from pathlib import Path
import joblib
import json

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "artifacts" / "lab7" / "walmart_random_forest.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "lab7" / "evaluation_metrics.json"
METADATA_PATH = ROOT_DIR / "data" / "processed" / "lab7_metadata.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_model_info():
    metadata = load_json(METADATA_PATH)
    metrics = load_json(METRICS_PATH)

    return {
        "model_name": "Walmart Random Forest Regressor",
        "model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "target": metadata.get("target", "Weekly_Sales"),
        "features": metadata.get("features", []),
        "metrics": metrics,
    }