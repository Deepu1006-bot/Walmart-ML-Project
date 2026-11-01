
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# STEP 1: PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "lab7"

MODEL_PATH = MODEL_DIR / "walmart_random_forest.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

MODEL_NAME = "Walmart_Weekly_Sales_Production_Model"
EXPERIMENT_NAME = "Walmart_Weekly_Sales_Prediction"


# ============================================================
# STEP 2: SAVE MODEL SAFELY
# ============================================================

def save_model_safely(model):
    """
    Save a compressed model to a temporary file first.
    Then copy it to the project's artifact directory.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    temporary_directory = Path(tempfile.gettempdir())
    temporary_model_path = (
        temporary_directory / "walmart_random_forest.joblib"
    )

    print("\n[INFO] Saving compressed model to temporary storage...")

    joblib.dump(
        model,
        temporary_model_path,
        compress=3,
    )

    if not temporary_model_path.exists():
        raise OSError("Temporary model file was not created.")

    if temporary_model_path.stat().st_size == 0:
        raise OSError("Temporary model file is empty.")

    print(
        f"[SUCCESS] Temporary model saved: "
        f"{temporary_model_path}"
    )

    print("[INFO] Copying model to project artifacts...")

    # Copy to a temporary destination in the target directory
    # before replacing the final model file.
    staged_model_path = MODEL_DIR / "walmart_random_forest.tmp"

    shutil.copy2(
        temporary_model_path,
        staged_model_path,
    )

    staged_model_path.replace(MODEL_PATH)

    print(f"[SUCCESS] Model saved: {MODEL_PATH}")
    print(
        f"[INFO] Model size: "
        f"{MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )


# ============================================================
# STEP 3: TRAIN AND REGISTER MODEL
# ============================================================

def train_and_register_model():

    print("\n" + "=" * 60)
    print("WALMART ML MODEL TRAINING AND REGISTRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    data_files = {
        "X_train": DATA_DIR / "lab7_X_train.npy",
        "X_test": DATA_DIR / "lab7_X_test.npy",
        "y_train": DATA_DIR / "lab7_y_train.npy",
        "y_test": DATA_DIR / "lab7_y_test.npy",
    }

    missing_files = [
        str(path)
        for path in data_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Processed data files are missing:\n"
            + "\n".join(missing_files)
            + "\nRun 03_preprocess_data.py first."
        )

    print("\n[INFO] Loading processed data...")

    X_train = np.load(data_files["X_train"])
    X_test = np.load(data_files["X_test"])
    y_train = np.load(data_files["y_train"]).ravel()
    y_test = np.load(data_files["y_test"]).ravel()

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape:  {X_test.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] y_test shape:  {y_test.shape}")

    # Validate dimensions
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Training and testing features must be 2D.")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "Training feature and target row counts do not match."
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "Testing feature and target row counts do not match."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and testing feature counts do not match."
        )

    # --------------------------------------------------------
    # Initialize and train Random Forest
    # --------------------------------------------------------

    params = {
        "n_estimators": 100,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": 1,
    }

    print("\n[INFO] Initializing Random Forest model...")
    print(f"[INFO] Parameters: {params}")

    model = RandomForestRegressor(**params)

    print("\n[INFO] Training model...")
    model.fit(X_train, y_train)

    print("[SUCCESS] Model training completed.")

    # --------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------

    print("\n[INFO] Evaluating model...")

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, predictions)),
    }

    print("\n" + "-" * 40)
    print("MODEL EVALUATION RESULTS")
    print("-" * 40)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # --------------------------------------------------------
    # Save model and metrics locally
    # --------------------------------------------------------

    print("\n[INFO] Saving model locally...")

    save_model_safely(model)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"[SUCCESS] Metrics saved: {METRICS_PATH}")

    # --------------------------------------------------------
    # Configure MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    # --------------------------------------------------------
    # Log parameters, metrics and model to MLflow
    # --------------------------------------------------------

    print("\n[INFO] Logging results to MLflow...")

    with mlflow.start_run(
        run_name="RandomForest_Registry_V1"
    ):

        mlflow.log_params(params)
        mlflow.log_param(
            "model_family",
            "RandomForestRegressor",
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(METRICS_PATH))

        metadata_path = DATA_DIR / "lab7_metadata.json"

        if metadata_path.exists():
            mlflow.log_artifact(str(metadata_path))

        print("[INFO] Registering model with MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
            registered_model_name=MODEL_NAME,
        )

        print("[SUCCESS] Model logged to MLflow.")
        print(
            "[INFO] MLflow run ID: "
            f"{mlflow.active_run().info.run_id}"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING AND REGISTRATION COMPLETED")
    print("=" * 60)
    print(f"Local model: {MODEL_PATH}")
    print(f"Metrics file: {METRICS_PATH}")
    print(f"Registered model: {MODEL_NAME}")


# ============================================================
# STEP 4: MAIN
# ============================================================

if __name__ == "__main__":
    try:
        train_and_register_model()
    except Exception as error:
        print(f"\n[ERROR] {type(error).__name__}: {error}")
        raise
