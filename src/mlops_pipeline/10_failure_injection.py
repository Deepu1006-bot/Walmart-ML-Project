import pandas as pd


def main():
    print("WALMART FAILURE-INJECTION TEST")
    print("=" * 40)

    # Simulate corrupted data without changing Walmart.csv.
    corrupted_data = pd.DataFrame({
        "Store": [1, 2],
        "Date": ["05-02-2010", "invalid-date"],
        "Weekly_Sales": [250000.0, float("nan")],
        "Holiday_Flag": [0, 1],
        "Temperature": [42.3, 45.1],
        "Fuel_Price": [2.5, 2.6],
        "CPI": [211.1, 212.0],
        "Unemployment": [8.1, 7.9],
    })

    failures = []

    if corrupted_data["Weekly_Sales"].isna().any():
        failures.append("Missing Weekly_Sales value detected")

    parsed_dates = pd.to_datetime(
        corrupted_data["Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )
    if parsed_dates.isna().any():
        failures.append("Invalid Date value detected")

    for failure in failures:
        print(f"[EXPECTED FAILURE] {failure}")

    if len(failures) != 2:
        raise SystemExit("[FAIL] Failure-injection test did not detect all issues.")

    print("[PASS] Corrupted data was detected successfully.")
    print("[INFO] Original Walmart.csv was not modified.")


if __name__ == "__main__":
    main()
