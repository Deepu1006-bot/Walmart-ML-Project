from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/Walmart.csv"
DRIFT = ROOT / "data/monitoring/drift"
OUT = ROOT / "outputs/monitoring"
DEPLOY = ROOT / "deployment"

for folder in [DRIFT, OUT, DEPLOY]:
    folder.mkdir(parents=True, exist_ok=True)

FEATURES = ["Store", "Holiday_Flag", "Temperature", "Fuel_Price",
            "CPI", "Unemployment", "Year", "Month", "Week"]
TARGET = "Weekly_Sales"

def load_data():
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    return df

def split_data():
    df = load_data()
    split = int(len(df) * 0.70)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def get_model():
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

def metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }

def save_json(data, filename):
    import json
    with open(OUT / filename, "w") as f:
        json.dump(data, f, indent=4)

def train_model(df):
    model = get_model()
    model.fit(df[FEATURES], df[TARGET])
    return model
