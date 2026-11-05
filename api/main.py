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
    description="Lab 13 End-to-End MLOps deployment API",
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

@app.post("/predict", response_model=WalmartPredictionResponse)
def predict(request: WalmartPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        date_value = datetime.strptime(request.Date, "%d-%m-%Y")

        input_values = {
            "Store": request.Store,
            "Holiday_Flag": request.Holiday_Flag,
            "Temperature": request.Temperature,
            "Fuel_Price": request.Fuel_Price,
            "CPI": request.CPI,
            "Unemployment": request.Unemployment,
            "Year": date_value.year,
            "Month": date_value.month,
            "Week": int(date_value.isocalendar().week)
        }

        input_df = pd.DataFrame([input_values], columns=FEATURES)

        start = time.perf_counter()
        prediction = float(model.predict(input_df)[0])
        latency = (time.perf_counter() - start) * 1000

        log_prediction(request.model_dump(), prediction, latency)

        return {
            "predicted_weekly_sales": prediction,
            "model": "Walmart Retrained Random Forest"
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid date. Use DD-MM-YYYY format."
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
