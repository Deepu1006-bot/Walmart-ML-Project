import os
import numpy as np


PROCESSED_DIR = "data/processed"


def validate_outputs():
    print("[INFO] Validating preprocessing outputs...")

    required_files = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy"
    ]

    for file_name in required_files:
        file_path = os.path.join(
            PROCESSED_DIR,
            file_name
        )

        if not os.path.exists(file_path):
            print(f"[ERROR] Missing file: {file_path}")
            return False

    X_train = np.load(
        os.path.join(PROCESSED_DIR, "X_train.npy")
    )
    X_test = np.load(
        os.path.join(PROCESSED_DIR, "X_test.npy")
    )
    y_train = np.load(
        os.path.join(PROCESSED_DIR, "y_train.npy")
    )
    y_test = np.load(
        os.path.join(PROCESSED_DIR, "y_test.npy")
    )

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")

    # Walmart notebook uses 10 features
    if X_train.shape[1] != 10:
        print("[ERROR] Incorrect number of training features.")
        return False

    if X_test.shape[1] != 10:
        print("[ERROR] Incorrect number of testing features.")
        return False

    # Check X and y row alignment
    if X_train.shape[0] != y_train.shape[0]:
        print("[ERROR] Training data mismatch.")
        return False

    if X_test.shape[0] != y_test.shape[0]:
        print("[ERROR] Testing data mismatch.")
        return False

    # Check missing values
    if np.isnan(X_train).any():
        print("[ERROR] Missing values found in X_train.")
        return False

    if np.isnan(X_test).any():
        print("[ERROR] Missing values found in X_test.")
        return False

    if np.isnan(y_train).any():
        print("[ERROR] Missing values found in y_train.")
        return False

    if np.isnan(y_test).any():
        print("[ERROR] Missing values found in y_test.")
        return False

    print("[SUCCESS] Preprocessing output validation PASSED.")

    return True


if __name__ == "__main__":
    if not validate_outputs():
        raise SystemExit(1)