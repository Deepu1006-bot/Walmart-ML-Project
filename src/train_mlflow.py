import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_and_track(
    run_name="RandomForest_Baseline",
    params=None
):

    if params is None:
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # Load processed data
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    # MLflow experiment
    mlflow.set_experiment("Walmart_Weekly_Sales_Prediction")

    with mlflow.start_run(run_name=run_name):

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        # Train model
        model = RandomForestRegressor(**params)

        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # Log metadata
        metadata_path = "data/processed/dataset_metadata.json"

        if os.path.exists(metadata_path):
            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        
        )

        # Local model backup
        os.makedirs("models", exist_ok=True)

        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print(
            f"[SUCCESS] Run '{run_name}' successfully tracked!"
        )


if __name__ == "__main__":

    # Baseline Random Forest
    train_and_track(
        run_name="RandomForest_Baseline"
    )

    # Second experiment
    tuned_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "n_jobs": -1
    }

    train_and_track(
        run_name="RandomForest_Tuned",
        params=tuned_params
    )