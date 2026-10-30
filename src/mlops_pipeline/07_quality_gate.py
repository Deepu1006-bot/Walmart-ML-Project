from pathlib import Path
import json
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "artifacts" / "lab7" / "evaluation_metrics.json"
MIN_R2 = 0.80


def main():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found: {METRICS_FILE}. "
            "Run model evaluation first."
        )

    with open(METRICS_FILE, encoding="utf-8") as file:
        metrics = json.load(file)

    required = ["MAE", "RMSE", "R2"]
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"Missing metrics: {missing}")

    for key in required:
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        if key in ("MAE", "RMSE") and value < 0:
            raise ValueError(f"{key} cannot be negative.")

    r2 = float(metrics["R2"])

    print("WALMART MODEL QUALITY GATE")
    print("=" * 40)
    print(f"MAE:  {float(metrics['MAE']):.4f}")
    print(f"RMSE: {float(metrics['RMSE']):.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Required R2: >= {MIN_R2:.2f}")

    if r2 < MIN_R2:
        raise SystemExit("[FAIL] Model did not pass the quality gate.")

    print("[PASS] Model passed the quality gate.")


if __name__ == "__main__":
    main()
