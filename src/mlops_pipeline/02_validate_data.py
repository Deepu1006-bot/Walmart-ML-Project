
import pandas as pd

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

MODULE_PATH = Path(__file__).parent / "01_load_data.py"
spec = spec_from_file_location("load_data_module", MODULE_PATH)
load_module = module_from_spec(spec)
spec.loader.exec_module(load_module)

load_data = load_module.load_data


def validate_data(df):
    errors = []

    if df.empty:
        errors.append("Dataset is empty.")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        errors.append(f"Missing values: {missing.to_dict()}")

    if df.duplicated().any():
        errors.append(f"Duplicate rows: {int(df.duplicated().sum())}")

    numeric_columns = [
        "Store", "Weekly_Sales", "Holiday_Flag",
        "Temperature", "Fuel_Price", "CPI", "Unemployment"
    ]

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} must be numeric.")

    if "Weekly_Sales" in df.columns and (df["Weekly_Sales"] < 0).any():
        errors.append("Weekly_Sales contains negative values.")

    if "Date" in df.columns:
        parsed_dates = pd.to_datetime(
            df["Date"], format="%d-%m-%Y", errors="coerce"
        )
        if parsed_dates.isna().any():
            errors.append("Date contains invalid values.")

    if errors:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(errors))

    print("[SUCCESS] Data validation passed.")
    return True


if __name__ == "__main__":
    dataset = load_data()
    validate_data(dataset)
