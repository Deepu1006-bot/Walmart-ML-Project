
import pandas as pd

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
    "Quarter",
]


def prepare_input(data: dict) -> pd.DataFrame:
    """Validate feature names and arrange them in training order."""
    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return pd.DataFrame(
        [[data[feature] for feature in FEATURES]],
        columns=FEATURES,
    )


def prepare_batch(records: list[dict]) -> pd.DataFrame:
    """Prepare multiple Walmart records for batch prediction."""
    if not records:
        raise ValueError("At least one record is required.")

    for index, record in enumerate(records):
        missing = [feature for feature in FEATURES if feature not in record]
        if missing:
            raise ValueError(
                f"Record {index} is missing features: {missing}"
            )

    return pd.DataFrame(
        [[record[feature] for feature in FEATURES] for record in records],
        columns=FEATURES,
    )
