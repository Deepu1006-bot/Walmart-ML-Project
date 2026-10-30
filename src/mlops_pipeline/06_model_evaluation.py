from pathlib import Path
import json
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "artifacts" / "lab7" / "evaluation_metrics.json"

def main():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

    with METRICS_FILE.open(encoding="utf-8") as f:
        metrics = json.load(f)

    required = ["MAE", "MSE", "RMSE", "R2"]
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"Missing metrics: {missing}")

    for key in required:
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"Invalid metric {key}: {value}")

    print("WALMART MODEL EVALUATION")
    print("=" * 40)
    for key in required:
        print(f"{key}: {float(metrics[key]):.4f}")
    print("[SUCCESS] Evaluation metrics validated.")

if __name__ == "__main__":
    main()
