
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/Walmart.csv")

REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
]


def load_data(path=DATA_PATH):
    print("[INFO] Loading Walmart dataset...")

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Walmart dataset is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[SUCCESS] Dataset loaded: {df.shape}")
    return df


if __name__ == "__main__":
    load_data()
