
from contextlib import asynccontextmanager
from time import perf_counter

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.model_loader import load_model, get_model_info
from api.preprocessing import prepare_input, prepare_batch
from api.schemas import (
    WalmartFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
)
from api.monitoring_logger import log_request


model = None
model_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_error
    try:
        model = load_model()
        model_error = None
        print("Walmart model loaded successfully.")
    except Exception as exc:
        model = None
        model_error = str(exc)
        print(f"Model loading failed: {exc}")
    yield
    model = None


app = FastAPI(
    title="Walmart Weekly Sales Prediction API",
    description="Lab 10: ML Model Serving and API Engineering",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Walmart Sales Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
    }


@app.get("/metadata", response_model=MetadataResponse, tags=["Model"])
def metadata():
    try:
        return get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: WalmartFeatures):
    start = perf_counter()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_error or 'not loaded'}",
        )

    try:
        data = prepare_input(payload.model_dump())
        prediction = float(np.asarray(model.predict(data)).ravel()[0])
        latency_ms = (perf_counter() - start) * 1000

        log_request(
            "/predict", "success", latency_ms,
            {"prediction": prediction},
        )

        return PredictionResponse(
            prediction=prediction,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = (perf_counter() - start) * 1000
        log_request("/predict", "error", latency_ms, {"error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(payload: BatchPredictionRequest):
    start = perf_counter()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_error or 'not loaded'}",
        )

    try:
        records = [record.model_dump() for record in payload.records]
        data = prepare_batch(records)
        predictions = [
            float(value)
            for value in np.asarray(model.predict(data)).ravel()
        ]
        latency_ms = (perf_counter() - start) * 1000

        log_request(
            "/predict/batch", "success", latency_ms,
            {"count": len(predictions)},
        )

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = (perf_counter() - start) * 1000
        log_request(
            "/predict/batch", "error", latency_ms,
            {"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def unexpected_error_handler(request, exc):
    log_request(
        request.url.path,
        "unexpected_error",
        0,
        {"error": str(exc)},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
