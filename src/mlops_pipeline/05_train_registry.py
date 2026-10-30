import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_and_register_model():

    print(
        "[INFO] --- Starting MLflow Run "
        "with Model Registry ---"
    )

    # 1. Set up MLflow
    mlflow.set_experiment(
        "Walmart_Weekly_Sales_Prediction"
    )

    # 2. Load Processed Data
    try:

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

    except FileNotFoundError:

        print(
            "[ERROR] Processed data not found. "
            "Please run the Lab 5 pipeline first."
        )

        return

    # 3. Define Model Parameters
    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "n_jobs": -1
    }

    with mlflow.start_run(
        run_name="RandomForest_Registry_V1"
    ) as run:

        # Log parameters
        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        # 4. Train Model
        print(
            "[INFO] Training RandomForest model..."
        )

        rf_model = RandomForestRegressor(
            **params
        )

        rf_model.fit(
            X_train,
            y_train
        )

        # 5. Evaluate
        print(
            "[INFO] Evaluating model..."
        )

        y_pred = rf_model.predict(
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

        mlflow.log_metrics(metrics)

        # 6. LOG PREPROCESSING METADATA (Lineage)
        print(
            "[INFO] Attaching preprocessing "
            "metadata to model artifacts..."
        )

        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(metadata_path):
            mlflow.log_artifact(
                metadata_path,
                artifact_path="preprocessing_pipeline"
            )

        # 7. LOG AND REGISTER THE MODEL
        print(
            "[INFO] Pushing model to MLflow Registry..."
        )

        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
            registered_model_name=
            "Walmart_Weekly_Sales_Production_Model"
        )

        print(
            f"[SUCCESS] Metrics logged: "
            f"R2 = {metrics['R2']:.4f} | "
            f"RMSE = {metrics['RMSE']:.4f}"
        )

        print(
            "[SUCCESS] Model successfully registered "
            "under name: "
            "'Walmart_Weekly_Sales_Production_Model'"
        )


if __name__ == "__main__":
    train_and_register_model()