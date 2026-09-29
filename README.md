# Walmart ML Project

## Project Overview

This project implements an end-to-end Machine Learning and MLOps workflow using the Walmart Weekly Sales dataset.

The project covers the complete workflow from data validation and preprocessing to model training, experiment tracking, reproducibility, model registration, lifecycle management, and production reporting.

---

## Problem Statement

The objective of this project is to predict Walmart's weekly sales using historical store, holiday, weather, fuel price, economic, and time-based information.

The target variable used for prediction is:

```text
Weekly_Sales

Dataset

The dataset used in this project is:

data/raw/Walmart.csv

Dataset Details
Number of records: 6435
Original features: 8
Target variable: Weekly_Sales
Original Columns

Column	Description
Store	Store identification number
Date	Date of weekly sales
Weekly_Sales	Weekly sales of the store
Holiday_Flag	Indicates whether the week is a holiday week
Temperature	Temperature during the week
Fuel_Price	Fuel price during the week
CPI	Consumer Price Index
Unemployment	Unemployment rate
Feature Engineering

The Date column is converted into time-based features:

Year
Month
Week
Quarter
Final Features
Store
Holiday_Flag
Temperature
Fuel_Price
CPI
Unemployment
Year
Month
Week
Quarter
Target
Weekly_Sales
Machine Learning

The project uses regression algorithms for Weekly Sales prediction.

The models implemented include:

Linear Regression
Ridge Regression
Lasso Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor

The main model used in the MLOps workflow is:

RandomForestRegressor
Evaluation Metrics

The models are evaluated using:

MAE — Mean Absolute Error
MSE — Mean Squared Error
RMSE — Root Mean Squared Error
R² — R-squared
MLOps Workflow
Lab 3 — Git-Based Version Control

Git is used for version control of the Walmart ML project.

The project is organized into separate directories for:

data/
src/
pipelines/
models/
artifacts/
outputs/
logs/
mlruns/
notebooks/

Git commits are used to track changes throughout the project.

Lab 4 — MLflow Experiment Tracking

Lab 4 implements experiment tracking using MLflow.

The workflow includes:

Data preprocessing
Random Forest model training
Experiment tracking
Parameter logging
Metric logging
Model logging
Dataset metadata logging
Actual vs Predicted visualization
Reproducibility validation
Baseline Random Forest
MAE  : 62106.9285
MSE  : 13170539565.5121
RMSE : 114762.9712
R²   : 0.9591
Tuned Random Forest
MAE  : 61766.2215
MSE  : 12831605765.0168
RMSE : 113276.6779
R²   : 0.9602
Reproducibility Test
Run 1 R² : 0.959117
Run 2 R² : 0.959117

Status : PASSED
Lab 5 — Production Data Pipeline

Lab 5 converts data validation and preprocessing into a production-style pipeline.

Pipeline
Walmart.csv
     ↓
Schema Validation
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Processed Data
     ↓
Output Validation
Schema Validation

Pandera is used to validate the Walmart dataset.

The validation checks:

Column names
Data types
Valid Store values
Weekly Sales values
Holiday Flag values
Fuel Price values
CPI values
Unemployment values
Train/Test Split
Training samples : 5148
Testing samples  : 1287
Features         : 10
Generated Processed Files
data/processed/X_train.npy
data/processed/X_test.npy
data/processed/y_train.npy
data/processed/y_test.npy
data/processed/dataset_metadata.json
Output Validation

The preprocessing outputs are checked for:

Missing values
Correct dimensions
Feature consistency
Row alignment

Lab 5 production pipeline completed successfully.

Lab 6 — Model Registry and Lifecycle Management

Lab 6 implements model registration and lifecycle management using MLflow Model Registry.

Workflow
Train Model
     ↓
Evaluate Model
     ↓
Register Model
     ↓
Staging
     ↓
Champion / Challenger Evaluation
     ↓
Production
     ↓
Production Report
Registered Model
Walmart_Weekly_Sales_Production_Model
Registered Version
Version : 1
R²      : 0.9602
RMSE    : 113276.6779
Model Lifecycle

The registered model was moved through:

None → Staging → Production

Version 1 was promoted to Production because there was no existing Production model.

Production Report

The registry workflow generates:

artifacts/production_model_report.json
Project Structure
Walmart-ML-Project/
│
├── data/
│   ├── raw/
│   │   └── Walmart.csv
│   └── processed/
│
├── notebooks/
│   └── Walmart.ipynb
│
├── pipelines/
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
│
├── src/
│   ├── validate_data.py
│   ├── preprocess.py
│   ├── preprocess_pipeline.py
│   ├── validate_outputs.py
│   ├── validate_reproducibility.py
│   ├── train_mlflow.py
│   ├── train_registry.py
│   ├── automate_lifecycle.py
│   └── generate_registry_report.py
│
├── artifacts/
├── models/
├── outputs/
├── logs/
├── mlruns/
│
├── .gitignore
├── README.md
└── requirements.txt
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Pandera
MLflow
Joblib
Git
GitHub
How to Run
Activate Virtual Environment
Windows PowerShell
.\.venv\Scripts\activate
Run Lab 5
.\.venv\Scripts\python.exe pipelines/run_lab5_pipeline.py
Run Lab 6
.\.venv\Scripts\python.exe pipelines/run_lab6_registry.py
Project Status
Lab	Status
Lab 3 — Version Control	Completed
Lab 4 — MLflow Experiment Tracking	Completed
Lab 5 — Production Data Pipeline	Completed
Lab 6 — Model Registry & Lifecycle	Completed
GitHub Repository	Completed


## Experiment 3 - Git Version Control

This experiment demonstrates branch creation, commit history, merging, rollback, revert, and version traceability for the Walmart ML project.  

Collaborative workflow: Walmart Branch B update.

Conclusion

The Walmart ML Project demonstrates an end-to-end Machine Learning and MLOps workflow.

The project progresses from raw data validation and preprocessing to model training, experiment tracking, reproducibility validation, model registration, lifecycle management, and production model reporting.