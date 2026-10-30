
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from importlib.util import spec_from_file_location, module_from_spec

BASE_DIR = Path(__file__).parent
MODULE_PATH = BASE_DIR / "02_validate_data.py"

spec = spec_from_file_location("validate_module", MODULE_PATH)
validate_module = module_from_spec(spec)
spec.loader.exec_module(validate_module)

validate_data = validate_module.validate_data
load_data = validate_module.load_data

OUTPUT_DIR = Path("data/processed")

FEATURES = [
    "Store", "Holiday_Flag", "Temperature", "Fuel_Price",
    "CPI", "Unemployment", "Year", "Month", "Week", "Quarter"
]
TARGET = "Weekly_Sales"


def preprocess_data():
    df = load_data()
    validate_data(df)

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain non-finite values.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Target contains non-finite values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "lab7_X_train.npy", X_train.to_numpy())
    np.save(OUTPUT_DIR / "lab7_X_test.npy", X_test.to_numpy())
    np.save(OUTPUT_DIR / "lab7_y_train.npy", y_train.to_numpy())
    np.save(OUTPUT_DIR / "lab7_y_test.npy", y_test.to_numpy())

    metadata = {
        "dataset": "Walmart.csv",
        "rows": len(df),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": FEATURES,
        "target": TARGET,
        "test_size": 0.20,
        "random_state": 42
    }

    with open(OUTPUT_DIR / "lab7_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Lab 7 preprocessing completed.")
    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Test shape: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data()
