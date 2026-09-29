import os
import hashlib
import pandas as pd

from validate_data import validate_schema
from preprocess_pipeline import preprocess_data


DATA_PATH = "data/raw/Walmart.csv"


def file_hash(path):
    """Generate SHA256 hash for reproducibility comparison."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def test_datatype_mismatch(df):
    print("\n[TEST 1] Datatype Mismatch")

    test_df = df.copy()

    test_df["Store"] = test_df["Store"].astype(str)

    result = validate_schema(
        test_df,
        output_report_name="datatype_mismatch.csv"
    )

    print(
        "[PASS] Datatype mismatch detected."
        if not result
        else "[FAIL] Datatype mismatch was not detected."
    )

    return not result


def test_missing_attribute(df):
    print("\n[TEST 2] Missing Attribute")

    test_df = df.copy()

    test_df = test_df.drop(
        columns=["Fuel_Price"]
    )

    result = validate_schema(
        test_df,
        output_report_name="missing_attribute.csv"
    )

    print(
        "[PASS] Missing attribute detected."
        if not result
        else "[FAIL] Missing attribute was not detected."
    )

    return not result


def test_invalid_categorical_value(df):
    print("\n[TEST 3] Invalid Categorical Value")

    test_df = df.copy()

    test_df.loc[
        test_df.index[0],
        "Holiday_Flag"
    ] = 5

    result = validate_schema(
        test_df,
        output_report_name="invalid_categorical_value.csv"
    )

    print(
        "[PASS] Invalid categorical value detected."
        if not result
        else "[FAIL] Invalid categorical value was not detected."
    )

    return not result


def test_corrupted_record(df):
    print("\n[TEST 4] Corrupted Record")

    test_df = df.copy()

    test_df.loc[
        test_df.index[0],
        "Weekly_Sales"
    ] = -1000

    result = validate_schema(
        test_df,
        output_report_name="corrupted_record.csv"
    )

    print(
        "[PASS] Corrupted record detected."
        if not result
        else "[FAIL] Corrupted record was not detected."
    )

    return not result


def test_idempotency():
    print("\n[TEST 5] Idempotency")

    preprocess_data()

    files = [
        "data/processed/X_train.npy",
        "data/processed/X_test.npy",
        "data/processed/y_train.npy",
        "data/processed/y_test.npy",
        "data/processed/dataset_metadata.json"
    ]

    first_hashes = {
        file: file_hash(file)
        for file in files
    }

    preprocess_data()

    second_hashes = {
        file: file_hash(file)
        for file in files
    }

    if first_hashes == second_hashes:
        print(
            "[PASS] Preprocessing is idempotent. "
            "Repeated execution produced identical outputs."
        )
        return True

    print(
        "[FAIL] Preprocessing outputs changed "
        "between repeated executions."
    )

    return False


def main():

    print(
        "\n========================================"
    )
    print(
        "Experiment 5 Validation Tests"
    )
    print(
        "========================================"
    )

    if not os.path.exists(DATA_PATH):
        print(
            f"[ERROR] Dataset not found: {DATA_PATH}"
        )
        return

    df = pd.read_csv(DATA_PATH)

    results = []

    results.append(
        test_datatype_mismatch(df)
    )

    results.append(
        test_missing_attribute(df)
    )

    results.append(
        test_invalid_categorical_value(df)
    )

    results.append(
        test_corrupted_record(df)
    )

    results.append(
        test_idempotency()
    )

    print(
        "\n========================================"
    )

    if all(results):
        print(
            "[SUCCESS] All Experiment 5 validation "
            "tests PASSED."
        )
    else:
        print(
            "[ERROR] One or more Experiment 5 "
            "validation tests FAILED."
        )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()