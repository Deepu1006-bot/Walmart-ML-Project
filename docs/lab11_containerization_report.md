# Lab 11: Containerization and Deployment Validation

## 1. Aim
To containerize the Walmart Weekly Sales prediction API using Docker and validate the containerized service.

## 2. Objectives
- Build a Docker image for the FastAPI application.
- Run the application inside a Docker container.
- Verify API health and model availability.
- Test the prediction endpoint.
- Compare local and containerized predictions.

## 3. Technology Stack
- Python 3.11
- FastAPI and Uvicorn
- Docker
- Scikit-learn and Joblib
- PowerShell and Requests

## 4. Container Configuration
- Docker image: `walmart-ml-api:lab11`
- Container name: `walmart-ml-container`
- Container port: 8000
- Host port: 8001
- API endpoint: `/predict`
- Health endpoint: `/health`

## 5. Implementation
A Dockerfile was created to install the API dependencies, copy the trained model and metadata, and start the FastAPI application using Uvicorn.

During initial testing, the container failed because pandas was missing from `requirements-api.txt`. The dependency was added and the image was rebuilt successfully.

## 6. Validation Results

| Test | Result |
|---|---|
| Docker image build | Passed |
| Container startup | Passed |
| Container health check | Passed |
| Model loading | Passed |
| Prediction endpoint | HTTP 200 |
| Local API prediction | 1610587.8967983348 |
| Docker API prediction | 1610587.8967983348 |
| Absolute prediction difference | 0.0 |
| Prediction consistency | Passed |

## 7. Performance Observation
- Local prediction request latency: 75.797 ms
- First measured Docker prediction request latency: 1302.761 ms

These are single-request measurements. More repeated tests are needed for a reliable performance comparison.

## 8. Conclusion
The Walmart Weekly Sales prediction API was successfully containerized using Docker. Both local and containerized APIs reported healthy status and returned HTTP 200 for the sample prediction. The predictions matched exactly, with an absolute difference of zero.

Kubernetes deployment and production-scale load testing are separate follow-up tasks and have not yet been validated by these results.