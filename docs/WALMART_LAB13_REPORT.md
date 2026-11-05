# Walmart Lab 13: End-to-End MLOps System Mini Project

## Aim
To integrate model deployment, input validation, API testing, Docker
containerization, CI automation, monitoring and Kubernetes configuration
into an end-to-end MLOps system.

## Model
Random Forest Regression model trained to predict Walmart Weekly Sales.

## Components
1. FastAPI prediction service.
2. Pydantic input validation.
3. Saved model loading using Joblib.
4. Prediction logging and latency monitoring.
5. Automated API tests using Pytest.
6. Docker containerization.
7. GitHub Actions continuous integration.
8. Kubernetes Deployment and Service manifests.

## API endpoints
- GET / : API information
- GET /health : Model health status
- POST /predict : Predict weekly sales
- GET /docs : Interactive API documentation

## Execution
Install requirements, start the FastAPI service, test the prediction endpoint,
build the Docker image, run automated tests and apply Kubernetes manifests.

## Conclusion
The project integrates the main stages of an end-to-end MLOps workflow.
