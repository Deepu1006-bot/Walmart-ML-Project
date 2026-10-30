from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Walmart.csv"
REQUIRED_COLUMNS = {"Date", "Weekly_Sales"}

def validate_frame(frame):
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["Weekly_Sales"].isna().any():
        raise ValueError("Weekly_Sales contains missing values")
    dates = pd.to_datetime(frame["Date"], errors="coerce", dayfirst=True)
    if dates.isna().any():
        raise ValueError("Date contains invalid values")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    original = pd.read_csv(DATA_PATH)
    print("WALMART FAILURE SIMULATION")

    scenarios = {
        "missing target": original.assign(Weekly_Sales=float("nan")),
        "invalid date": original.assign(Date="INVALID_DATE"),
        "missing column": original.drop(columns=["Weekly_Sales"]),
    }

    for label, corrupted in scenarios.items():
        try:
            validate_frame(corrupted)
        except ValueError as exc:
            print(f"[EXPECTED FAILURE] {label}: {exc}")
        else:
            raise AssertionError(f"Failure was not detected: {label}")

    print("[PASS] All corruption scenarios were detected.")
    print("[INFO] Original CSV was read only and not modified.")

if __name__ == "__main__":
    main()
