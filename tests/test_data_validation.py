
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

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


def test_valid_data_passes():
    assert validate_data(valid_dataframe()) is True


def test_missing_values_fail():
    df = valid_dataframe()
    df.loc[0, "Weekly_Sales"] = float("nan")

    with pytest.raises(ValueError, match="Missing values"):
        validate_data(df)


def test_duplicate_rows_fail():
    df = valid_dataframe()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate rows"):
        validate_data(df)


def test_invalid_date_fails():
    df = valid_dataframe()
    df.loc[0, "Date"] = "invalid-date"

    with pytest.raises(ValueError, match="invalid values"):
        validate_data(df)


def test_negative_sales_fail():
    df = valid_dataframe()
    df.loc[0, "Weekly_Sales"] = -100

    with pytest.raises(ValueError, match="negative values"):
        validate_data(df)
