# MLflow and CI Setup

## Environment
Activate the project virtual environment and install dependencies
from requirements.txt.

## Experiment tracking
The project uses the Walmart_Weekly_Sales_Prediction experiment.
Training parameters and evaluation metrics should be logged to MLflow.

## Model registry
The intended registered model name is
Walmart_Weekly_Sales_Production_Model.

## Continuous integration
The GitHub Actions workflows are located in `.github/workflows/`.
Review their configured Python version, dependency installation,
test commands, and artifact paths before relying on CI results.

## Local verification
Run the pipeline and tests from the repository root.
Document actual outcomes; do not report unrun checks as passing.
