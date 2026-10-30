from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FILES = [
    "lab7_X_train.npy",
    "lab7_X_test.npy",
    "lab7_y_train.npy",
    "lab7_y_test.npy",
    "lab7_metadata.json",
]


def main():
    print("WALMART OUTPUT VALIDATION")
    print("=" * 40)

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required output missing: {path}")
        print(f"[OK] Found {filename}")

    X_train = np.load(DATA_DIR / "lab7_X_train.npy")
    X_test = np.load(DATA_DIR / "lab7_X_test.npy")
    y_train = np.load(DATA_DIR / "lab7_y_train.npy")
    y_test = np.load(DATA_DIR / "lab7_y_test.npy")

    with open(DATA_DIR / "lab7_metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test feature counts do not match.")

    if X_train.shape[1] != 10:
        raise ValueError(f"Expected 10 features, found {X_train.shape[1]}.")

    if X_train.shape[0] != len(y_train):
        raise ValueError("Training features and targets do not match.")

    if X_test.shape[0] != len(y_test):
        raise ValueError("Test features and targets do not match.")

    for name, array in [
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} is empty or contains invalid values.")

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Metadata loaded: {bool(metadata)}")
    print("[SUCCESS] All processed outputs are valid.")


if __name__ == "__main__":
    main()
