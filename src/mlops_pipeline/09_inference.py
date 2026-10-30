from pathlib import Path
import json
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "artifacts" / "lab7" / "walmart_random_forest.joblib"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "lab7_X_test.npy"
OUTPUT_FILE = PROJECT_ROOT / "artifacts" / "lab7" / "sample_predictions.npy"
REPORT_FILE = PROJECT_ROOT / "artifacts" / "lab7" / "inference_report.json"

def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_FILE}")

    model = joblib.load(MODEL_FILE)
    X_test = np.load(TEST_FILE)

    if X_test.ndim != 2 or X_test.shape[1] != 10:
        raise ValueError(f"Expected 10 input features; got {X_test.shape}")

    predictions = model.predict(X_test)
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain invalid values.")

    np.save(OUTPUT_FILE, predictions)
    report = {
        "model": type(model).__name__,
        "test_samples": int(len(X_test)),
        "predictions_generated": int(len(predictions)),
        "sample_predictions": predictions[:5].tolist(),
        "status": "PASS"
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[SUCCESS] Walmart inference completed.")
    print(f"Predictions generated: {len(predictions)}")
    print(f"Saved predictions: {OUTPUT_FILE}")
    print(f"Saved report: {REPORT_FILE}")

if __name__ == "__main__":
    main()
