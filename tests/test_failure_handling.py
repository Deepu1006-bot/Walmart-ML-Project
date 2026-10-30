
from pathlib import Path
import importlib.util

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "mlops_pipeline" / "02_validate_data.py"

spec = importlib.util.spec_from_file_location("validate_module", MODULE_PATH)
validate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_module)

validate_data = validate_module.validate_data


def valid_dataframe():
    return pd.DataFrame({
        "Store": [1, 2],
        "Date": ["05-02-2010", "12-02-2010"],
        "Weekly_Sales": [150000.0, 200000.0],
        "Holiday_Flag": [0, 1],
        "Temperature": [42.3, 38.5],
        "Fuel_Price": [2.5, 2.7],
        "CPI": [211.0, 212.0],
        "Unemployment": [8.1, 8.0],
    })


def test_missing_required_column_is_rejected_by_loader():
    loader_path = PROJECT_ROOT / "src" / "mlops_pipeline" / "01_load_data.py"
    loader_spec = importlib.util.spec_from_file_location("loader_module", loader_path)
    loader_module = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader_module)

    with pytest.raises(FileNotFoundError):
        loader_module.load_data(PROJECT_ROOT / "data" / "raw" / "missing.csv")


def test_schema_mismatch_is_detected():
    df = valid_dataframe().drop(columns=["Fuel_Price"])

    with pytest.raises(KeyError):
        validate_data(df)


def test_invalid_numeric_type_is_rejected():
    df = valid_dataframe()
    df["Temperature"] = ["hot", "cold"]

    with pytest.raises(ValueError, match="must be numeric"):
        validate_data(df)


def test_empty_dataframe_is_rejected():
    df = valid_dataframe().iloc[0:0]

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_data(df)
