import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
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
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        }

    print(
        f"\n--- Starting MLflow Run: {run_name} ---"
    )

    # 1. Load Data
    X_train = np.load(
        "data/processed/X_train.npy"
    )

    X_test = np.load(
        "data/processed/X_test.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    # Set Experiment
    mlflow.set_experiment(
        "Walmart_Weekly_Sales_Prediction"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # 2. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        # 3. Train Model
        model = RandomForestRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        # 4. Evaluate Predictions
        y_pred = model.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        metrics = {
            "MAE": mean_absolute_error(
                y_test,
                y_pred
            ),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(
                y_test,
                y_pred
            )
        }

        # 5. Log Metrics to MLflow
        mlflow.log_metrics(metrics)

        print(
            f"Metrics logged: "
            f"R2 = {metrics['R2']:.4f} | "
            f"RMSE = {metrics['RMSE']:.4f}"
        )

        # 6. Generate Diagnostic Plot
        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.scatter(
            y_test,
            y_pred,
            alpha=0.5
        )

        ax.set_xlabel(
            "Actual Weekly Sales"
        )

        ax.set_ylabel(
            "Predicted Weekly Sales"
        )

        ax.set_title(
            f"Actual vs Predicted - {run_name}"
        )

        plot_path = (
            "artifacts/actual_vs_predicted.png"
        )

        fig.savefig(
            plot_path,
            bbox_inches="tight"
        )

        plt.close(fig)

        mlflow.log_artifact(
            plot_path,
            artifact_path="plots"
        )

        # 7. Log Preprocessing Metadata for Lineage
        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(metadata_path):
            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

        # 8. Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Save local backup
        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print(
            f"Run '{run_name}' successfully tracked!"
        )


if __name__ == "__main__":

    # Baseline Run
    train_and_track(
        run_name="RandomForest"
    )

    # Tuned Run
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