import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def get_walmart_schema():
    """Defines the strict schema specification for the Walmart dataset."""
    return pa.DataFrameSchema({
        "Store": Column(
            pa.Int,
            Check.ge(1)
        ),
        "Date": Column(
            pa.String
        ),
        "Weekly_Sales": Column(
            pa.Float,
            Check.ge(0.0)
        ),
        "Holiday_Flag": Column(
            pa.Int,
            Check.isin([0, 1])
        ),
        "Temperature": Column(
            pa.Float
        ),
        "Fuel_Price": Column(
            pa.Float,
            Check.ge(0.0)
        ),
        "CPI": Column(
            pa.Float,
            Check.ge(0.0)
        ),
        "Unemployment": Column(
            pa.Float,
            Check.ge(0.0)
        )
    }, strict=True)


def validate_schema(
    df,
    output_report_name="schema_validation_errors.csv"
):
    """Validates the input DataFrame against the defined schema."""

    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_walmart_schema()

    try:
        schema.validate(df, lazy=True)
        print(
            "[SUCCESS] Schema Validation PASSED. "
            "Dataset is clean."
        )
        return True

    except pa.errors.SchemaErrors as err:
        print(
            "[ERROR] Schema Validation FAILED. "
            "Corruptions detected."
        )

        failures = err.failure_cases[
            [
                "schema_context",
                "column",
                "check",
                "failure_case",
                "index"
            ]
        ]

        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)

        report_path = os.path.join(
            "artifacts",
            output_report_name
        )

        failures.to_csv(
            report_path,
            index=False
        )

        print(
            f"[INFO] Detailed failure report saved "
            f"to '{report_path}'."
        )

        return False


if __name__ == "__main__":
    # When run directly, it only validates the clean production data
    data_path = "data/raw/Walmart.csv"

    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)

        validate_schema(
            raw_df,
            output_report_name="baseline_validation.csv"
        )
    else:
        print(
            f"[ERROR] Target file not found at: {data_path}"
        )