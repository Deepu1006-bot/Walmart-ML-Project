import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def train_once(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return r2_score(y_test, predictions)


def main():

    print("[INFO] Checking Walmart model reproducibility...")

    X_train = np.load(
        "data/processed/X_train.npy"
    )

    X_test = np.load(
        "data/processed/X_test.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    # First execution
    r2_run_1 = train_once(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Second execution
    r2_run_2 = train_once(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"Run 1 R2: {r2_run_1:.6f}")
    print(f"Run 2 R2: {r2_run_2:.6f}")

    if np.isclose(r2_run_1, r2_run_2):
        print(
            "[SUCCESS] Reproducibility validation PASSED."
        )
    else:
        print(
            "[ERROR] Reproducibility validation FAILED."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()