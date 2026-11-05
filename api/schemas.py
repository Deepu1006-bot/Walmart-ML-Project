
from typing import Optional
from pydantic import BaseModel, Field


class WalmartPredictionRequest(BaseModel):
    Store: int = Field(gt=0)
    Date: Optional[str] = None
    Holiday_Flag: int = Field(ge=0, le=1)
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float

    # Legacy Lab 10 fields
    Year: Optional[int] = None
    Month: Optional[int] = None
    Week: Optional[int] = None
    Quarter: Optional[int] = None


class WalmartPredictionResponse(BaseModel):
    predicted_weekly_sales: float
    model: str
    prediction: float
    latency_ms: float
