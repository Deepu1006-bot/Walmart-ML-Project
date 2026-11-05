
from fastapi import FastAPI, HTTPException
from datetime import datetime
import pandas as pd
import time

from api.schemas import (
    WalmartPredictionRequest,
    WalmartPredictionResponse
)
from api.model_loader import load_model
from api.monitoring_logger import log_prediction

app = FastAPI(
    title="Walmart Weekly Sales Prediction API",
    description="End-to-End Walmart MLOps API",
    version="1.0.0"
)

model = None

FEATURES = [
    "Store", "Holiday_Flag", "Temperature", "Fuel_Price",
    "CPI", "Unemployment", "Year", "Month", "Week"
]


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
    except Exception as exc:
        raise RuntimeError(f"Model loading failed: {exc}") from exc


@app.get("/")
def home():
    return {
        "message": "Walmart ML API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None
    }


@app.get("/metadata")
def metadata():
    return {
        "target": "Weekly_Sales",
        "model": "Walmart Retrained Random Forest",
        "features": FEATURES
    }


def generate_prediction(request: WalmartPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.Date:
        try:
            date_value = datetime.strptime(request.Date, "%d-%m-%Y")
            year = date_value.year
            month = date_value.month
            week = int(date_value.isocalendar().week)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid date. Use DD-MM-YYYY format."
            ) from exc
    else:
        if request.Year is None or request.Month is None or request.Week is None:
            raise HTTPException(
                status_code=422,
                detail="Provide Date or all of Year, Month and Week."
            )

        if not (1 <= request.Month <= 12 and 1 <= request.Week <= 53):
            raise HTTPException(
                status_code=422,
                detail="Month must be 1-12 and Week must be 1-53."
            )

        year = request.Year
        month = request.Month
        week = request.Week

    input_values = {
        "Store": request.Store,
        "Holiday_Flag": request.Holiday_Flag,
        "Temperature": request.Temperature,
        "Fuel_Price": request.Fuel_Price,
        "CPI": request.CPI,
        "Unemployment": request.Unemployment,
        "Year": year,
        "Month": month,
        "Week": week
    }

    input_df = pd.DataFrame([input_values], columns=FEATURES)

    start = time.perf_counter()
    prediction = float(model.predict(input_df)[0])
    latency = (time.perf_counter() - start) * 1000

    log_prediction(request.model_dump(), prediction, latency)

    return {
        "predicted_weekly_sales": prediction,
        "model": "Walmart Retrained Random Forest",
        "prediction": prediction,
        "latency_ms": latency
    }


@app.post("/predict", response_model=WalmartPredictionResponse)
def predict(request: WalmartPredictionRequest):
    try:
        return generate_prediction(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(payload: dict):
    records = payload.get("records")

    if not isinstance(records, list) or not records:
        raise HTTPException(
            status_code=422,
            detail="Provide a non-empty records list."
        )

    predictions = []

    for record in records:
        try:
            request = WalmartPredictionRequest(**record)
            result = generate_prediction(request)
            predictions.append(result["prediction"])
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "count": len(predictions),
        "predictions": predictions
    }
