# Lab 7 - Modular Walmart ML Pipeline

## Objective
Build a modular machine learning pipeline for the Walmart weekly sales dataset.

## Dataset
- File: `data/raw/Walmart.csv`
- Rows: 6,435
- Columns: 8
- Target: `Weekly_Sales`

## Pipeline Steps
1. `01_load_data.py` - Loads the dataset and checks required columns.
2. `02_validate_data.py` - Validates data quality.
3. `03_preprocess_data.py` - Creates date-based features and splits data.
4. `04_train_model.py` - Trains a Random Forest regression model.
5. `05_evaluate_model.py` - Calculates evaluation metrics.
6. `06_inference.py` - Generates sample predictions.
7. `07_log_artifacts.py` - Logs model artifacts using MLflow.

## Execution
Run from the project root:

```powershell
python .\pipelines\run_lab7_pipeline.py


### Verify the file

```powershell
Get-Item .\docs\LAB7_PIPELINE.md | Select-Object Name, Length
q



@'
# Lab 7 - Modular Walmart ML Pipeline

## Objective
Build a modular machine learning pipeline for the Walmart weekly sales dataset.

## Dataset
- File: `data/raw/Walmart.csv`
- Rows: 6,435
- Columns: 8
- Target: `Weekly_Sales`

## Pipeline Steps
1. `01_load_data.py` - Loads the dataset and checks required columns.
2. `02_validate_data.py` - Validates data quality.
3. `03_preprocess_data.py` - Creates date-based features and splits data.
4. `04_train_model.py` - Trains a Random Forest regression model.
5. `05_evaluate_model.py` - Calculates evaluation metrics.
6. `06_inference.py` - Generates sample predictions.
7. `07_log_artifacts.py` - Logs model artifacts using MLflow.

## Execution
Run from the project root:

    python .\pipelines\run_lab7_pipeline.py

## Preprocessing Results
- Training samples: 5,148
- Testing samples: 1,287
- Features: 10

## Recorded Evaluation Results
- MAE: 61,766.2215
- MSE: 12,831,605,765.0168
- RMSE: 113,276.6779
- R-squared: 0.9602

## Main Artifacts
- Model: `artifacts/lab7/walmart_random_forest.joblib`
- Metrics: `artifacts/lab7/evaluation_metrics.json`
- Predictions: `artifacts/lab7/sample_predictions.npy`

## Result
The modular pipeline completed successfully in the local environment, and the model artifacts were saved.
