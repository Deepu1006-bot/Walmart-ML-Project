
from pydantic import BaseModel, Field, ConfigDict
from typing import List


class WalmartFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Store: int = Field(gt=0, description="Walmart store ID")
    Holiday_Flag: int = Field(ge=0, le=1)
    Temperature: float
    Fuel_Price: float = Field(ge=0)
    CPI: float = Field(ge=0)
    Unemployment: float = Field(ge=0)
    Year: int = Field(ge=2000, le=2100)
    Month: int = Field(ge=1, le=12)
    Week: int = Field(ge=1, le=53)
    Quarter: int = Field(ge=1, le=4)


class PredictionResponse(BaseModel):
    prediction: float
    target: str = "Weekly_Sales"
    model_name: str = "Walmart Random Forest Regressor"
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    records: List[WalmartFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int
    target: str = "Weekly_Sales"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_name: str
    model_path: str
    target: str
    features: list[str]
    metrics: dict
