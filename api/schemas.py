from pydantic import BaseModel, Field

class WalmartPredictionRequest(BaseModel):
    Store: int = Field(gt=0)
    Date: str
    Holiday_Flag: int = Field(ge=0, le=1)
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float

class WalmartPredictionResponse(BaseModel):
    predicted_weekly_sales: float
    model: str
