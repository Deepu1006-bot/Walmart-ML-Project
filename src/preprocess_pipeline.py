import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/Walmart.csv"
PROCESSED_DIR = "data/processed"

FEATURES = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
    "Quarter"
]

TARGET = "Weekly_Sales"


def preprocess_data():

    print("[INFO] Starting Walmart preprocessing pipeline...")

    df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Date conversion
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y"
    )

    # Temporal features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    # Feature and target selection
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save processed data
    np.save(
        f"{PROCESSED_DIR}/X_train.npy",
        X_train.values
    )

    np.save(
        f"{PROCESSED_DIR}/X_test.npy",
        X_test.values
    )

    np.save(
        f"{PROCESSED_DIR}/y_train.npy",
        y_train.values
    )

    np.save(
        f"{PROCESSED_DIR}/y_test.npy",
        y_test.values
    )

    # Save metadata
    metadata = {
        "dataset": "Walmart.csv",
        "original_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "test_size": 0.20,
        "random_state": 42,
        "date_format": "%d-%m-%Y"
    }

    with open(
        f"{PROCESSED_DIR}/dataset_metadata.json",
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("[SUCCESS] Walmart preprocessing pipeline completed.")
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Testing samples: {len(X_test)}")
    print(f"[INFO] Features used: {len(FEATURES)}")


if __name__ == "__main__":
    preprocess_data()