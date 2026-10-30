
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "artifacts" / "lab7" / "evaluation_metrics.json"


def main():
    print("=" * 40)
    print("WALMART MODEL EVALUATION")
    print("=" * 40)

    # Ensure the output directory exists
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Use existing metrics when available.
    # Otherwise, train/evaluate the model from processed data.
    if not METRICS_FILE.exists():
        print("[INFO] Metrics file not found. Generating metrics...")

        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        data_dir = PROJECT_ROOT / "data" / "processed"

        X_train = np.load(data_dir / "X_train.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_train = np.load(data_dir / "y_train.npy")
        y_test = np.load(data_dir / "y_test.npy")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "MSE": float(mse),
            "RMSE": float(math.sqrt(mse)),
            "R2": float(r2_score(y_test, predictions)),
        }

        METRICS_FILE.write_text(
            json.dumps(metrics, indent=4),
            encoding="utf-8",
        )
        print(f"[INFO] Metrics saved to: {METRICS_FILE}")

    with METRICS_FILE.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    required = ["MAE", "MSE", "RMSE", "R2"]

    for name in required:
        if name not in metrics:
            raise ValueError(f"Required metric missing: {name}")
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"Metric is not finite: {name}")

    for name in ["MAE", "MSE", "RMSE"]:
        if float(metrics[name]) < 0:
            raise ValueError(f"{name} cannot be negative")

    for name in required:
        print(f"{name}: {float(metrics[name]):.4f}")

    print("[SUCCESS] Evaluation metrics validated.")


if __name__ == "__main__":
    main()
