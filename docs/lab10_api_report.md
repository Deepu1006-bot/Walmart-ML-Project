
# Lab 10: ML Model Serving and API Engineering

## 1. Aim
To deploy the trained Walmart Weekly Sales prediction model through a FastAPI inference service.

## 2. Objectives
- Build health, metadata, single prediction and batch prediction endpoints.
- Validate incoming data using Pydantic schemas.
- Handle invalid inputs and model availability errors.
- Record request status and latency.
- Evaluate API performance.

## 3. Technology Stack
- Python 3.11
- FastAPI and Uvicorn
- Pydantic
- Scikit-learn and Joblib
- Pytest and HTTPX

## 4. Model Details
Model: Walmart Random Forest Regressor
Target: Weekly_Sales
Input features: Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, Week, Quarter.

## 5. API Endpoints
- GET / : API welcome response
- GET /health : Model health status
- GET /metadata : Model metadata and evaluation metrics
- POST /predict : Single prediction
- POST /predict/batch : Batch prediction

## 6. Testing
Six automated API tests passed:
1. Root endpoint
2. Health endpoint
3. Metadata endpoint
4. Single prediction
5. Batch prediction
6. Invalid input validation

Performance test: 30 out of 30 prediction requests succeeded.

## 7. Performance Results
- Mean latency: 10.151 ms
- Median latency: 9.967 ms
- Minimum latency: 6.987 ms
- Maximum latency: 16.458 ms
- Approximate sequential throughput: 98.513 requests/second

Results are from a local sequential test and may differ in production or under concurrent load.

## 8. Error Handling and Monitoring
Pydantic validates request fields. Invalid input returns HTTP 422. The API reports model unavailability with HTTP 503. Request status and latency are recorded in logs/api/api.log.

## 9. Conclusion
The trained Walmart sales model was served through FastAPI. Health, metadata, prediction, batch prediction and validation endpoints were tested successfully. Automated API tests and the sequential performance test passed.
