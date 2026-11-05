from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "deployment" / "walmart_retrained_model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run Lab 12 retraining first."
        )
    return joblib.load(MODEL_PATH)
