# Walmart ML Project

## Project Overview

This project implements an end-to-end Machine Learning and MLOps workflow using the Walmart Weekly Sales dataset.

It covers data validation, preprocessing, model training, experiment tracking, reproducibility, model registration, lifecycle management, monitoring, drift detection, retraining, and deployment using FastAPI, Docker, and Kubernetes.

## Problem Statement

The objective is to predict Walmart's weekly sales using historical store, holiday, weather, fuel price, economic, and time-based information.

**Target variable:** `Weekly_Sales`

## Dataset

**Dataset location:** `data/raw/Walmart.csv`

- Number of records: 6,435
- Original features: 8
- Target variable: `Weekly_Sales`

### Original Columns

| Column | Description |
|---|---|
| Store | Store identification number |
| Date | Date of weekly sales |
| Weekly_Sales | Weekly sales of the store |
| Holiday_Flag | Indicates whether the week is a holiday week |
| Temperature | Temperature during the week |
| Fuel_Price | Fuel price during the week |
| CPI | Consumer Price Index |
| Unemployment | Unemployment rate |

### Feature Engineering

The `Date` column is converted into time-based features:

- Year
- Month
- Week
- Quarter

### Final Features

- Store
- Holiday_Flag
- Temperature
- Fuel_Price
- CPI
- Unemployment
- Year
- Month
- Week
- Quarter

**Target:** `Weekly_Sales`

## Machine Learning

The project uses regression algorithms for weekly sales prediction.

Algorithms implemented:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The main model used in the MLOps workflow is `RandomForestRegressor`.

### Evaluation Metrics

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² — R-squared

---

# MLOps Workflow and Experiments

## Lab 3 — Git-Based Version Control

Git is used for version control of the Walmart ML project.

The project is organized into separate directories for data, source code, pipelines, models, artifacts, outputs, logs, MLflow runs, and notebooks.

Git commits are used to track project changes.

## Lab 4 — MLflow Experiment Tracking

Lab 4 implements experiment tracking using MLflow.

The workflow includes:

- Data preprocessing
- Random Forest model training
- Experiment tracking
- Parameter logging
- Metric logging
- Model logging
- Dataset metadata logging
- Actual vs. Predicted visualization
- Reproducibility validation

### Baseline Random Forest

| Metric | Value |
|---|---:|
| MAE | 62106.9285 |
| MSE | 13170539565.5121 |
| RMSE | 114762.9712 |
| R² | 0.9591 |

### Tuned Random Forest

| Metric | Value |
|---|---:|
| MAE | 61766.2215 |
| MSE | 12831605765.0168 |
| RMSE | 113276.6779 |
| R² | 0.9602 |

### Reproducibility Test

- Run 1 R²: 0.959117
- Run 2 R²: 0.959117
- Status: PASSED

## Lab 5 — Production Data Pipeline

Lab 5 converts data validation and preprocessing into a production-style pipeline.

### Pipeline

```text
Walmart.csv
     |
Schema Validation
     |
Data Preprocessing
     |
Train/Test Split
     |
Processed Data
     |
Output Validation
```

### Schema Validation

Pandera is used to validate the Walmart dataset.

Validation checks include:

- Column names
- Data types
- Valid Store values
- Weekly Sales values
- Holiday Flag values
- Fuel Price values
- CPI values
- Unemployment values

### Train/Test Split

- Training samples: 5148
- Testing samples: 1287
- Features: 10

### Generated Processed Files

```text
data/processed/X_train.npy
data/processed/X_test.npy
data/processed/y_train.npy
data/processed/y_test.npy
data/processed/dataset_metadata.json
```

Output validation checks missing values, dimensions, feature consistency, and row alignment.

## Lab 6 — Model Registry and Lifecycle Management

Lab 6 implements model registration and lifecycle management using MLflow Model Registry.

### Workflow

```text
Train Model
     |
Evaluate Model
     |
Register Model
     |
Staging
     |
Champion / Challenger Evaluation
     |
Production
     |
Production Report
```

### Registered Model

`Walmart_Weekly_Sales_Production_Model`

- Registered version: 1
- R²: 0.9602
- RMSE: 113276.6779

The registered model was moved through:

`None → Staging → Production`

Version 1 was promoted to Production because there was no existing Production model.

Generated report:

`artifacts/production_model_report.json`

## Lab 12 — Monitoring, Drift Detection, and Retraining

Lab 12 implements monitoring and drift management for the Walmart ML model.

### Key Components

- Data drift detection
- Concept drift analysis
- Prediction drift monitoring
- Model performance monitoring
- Drift data generation
- Model comparison
- Model retraining
- Model metadata tracking

### Workflow

```text
Reference Data + Current Data
             |
       Drift Detection
             |
      Performance Monitoring
             |
       Drift Analysis
             |
      Model Comparison
             |
       Model Retraining
             |
      Updated Model
             |
      Model Metadata
```

### Model Comparison Results

| Metric | Baseline Model | Retrained Model |
|---|---:|---:|
| R² | 0.7036 | 0.8871 |
| RMSE | 290174.47 | 179042.21 |

The retrained model achieved a higher R² and lower RMSE than the baseline model in this comparison.

### Important Files

```text
reports/
└── walmart_lab12_report.md

deployment/
├── model_metadata.json
└── walmart_retrained_model.pkl

data/
└── monitoring/
    └── drift/
        ├── current_data.csv
        └── reference_data.csv

src/
└── mlops_pipeline/
    ├── walmart_12_concept_drift.py
    ├── walmart_12_create_drift_data.py
    ├── walmart_12_data_drift.py
    ├── walmart_12_drift_analysis.py
    ├── walmart_12_model_comparison.py
    ├── walmart_12_performance_monitoring.py
    ├── walmart_12_prediction_drift.py
    ├── walmart_12_retrain_model.py
    └── walmart_12_utils.py
```

## Lab 13 — End-to-End Walmart MLOps System

Lab 13 integrates the Walmart ML model into an end-to-end deployment workflow.

### Key Components

- FastAPI-based prediction API
- Model loading
- Input validation using API schemas
- Health-check endpoint
- Prediction endpoint
- Prediction monitoring logs
- Automated API testing
- Docker containerization
- Kubernetes deployment and service configuration
- GitHub Actions workflow

### Architecture

```text
Walmart Dataset
       |
Trained ML Model
       |
Model Loader
       |
FastAPI Application
       |
Input Validation
       |
Weekly Sales Prediction
       |
Monitoring Logs
       |
Docker Container
       |
Kubernetes Deployment
       |
Kubernetes Service
```

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Health status and model availability |
| `/predict` | POST | Predict weekly sales |

Interactive API documentation is available at:

`http://localhost:8000/docs`

### Docker

Build the Docker image:

```bash
docker build -t walmart-ml-api:latest .
```

Run the container:

```bash
docker run -d --name walmart-api-container -p 8000:8000 walmart-ml-api:latest
```

### Kubernetes

The project includes Kubernetes deployment and service configuration files.

```text
k8s/
├── deployment.yaml
└── service.yaml
```

The Walmart API was validated using Kubernetes with:

- Deployment: `walmart-ml-api`
- Service: `walmart-ml-api-service`
- Service port: `8000`
- NodePort: `30081`

The API was also accessed through port forwarding:

```bash
kubectl port-forward service/walmart-ml-api-service 8001:8000
```

Swagger documentation:

`http://localhost:8001/docs`

### API Validation

The following validations were performed:

- Health endpoint returned HTTP 200.
- Model loading status was healthy.
- Prediction endpoint returned HTTP 200.
- Weekly sales prediction was generated successfully.

### GitHub Actions

The project includes the Lab 13 workflow:

`.github/workflows/walmart_lab13_end_to_end.yml`

It installs API dependencies and runs the dedicated API tests.

---

# Project Structure

```text
Walmart-ML-Project/
│
├── .github/
│   └── workflows/
│
├── api/
│   ├── main.py
│   ├── model_loader.py
│   ├── monitoring_logger.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── monitoring/
│