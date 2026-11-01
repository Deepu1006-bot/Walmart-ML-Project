
import json
import time
from pathlib import Path

import requests

LOCAL_URL = "http://127.0.0.1:8000"
CONTAINER_URL = "http://127.0.0.1:8001"

SAMPLE = {
    "Store": 1,
    "Holiday_Flag": 0,
    "Temperature": 42.31,
    "Fuel_Price": 2.572,
    "CPI": 211.096358,
    "Unemployment": 8.106,
    "Year": 2010,
    "Month": 2,
    "Week": 5,
    "Quarter": 1,
}


def test_health(base_url):
    response = requests.get(f"{base_url}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def test_prediction(base_url):
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/predict",
        json=SAMPLE,
        timeout=60,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    result = response.json()

    return {
        "status_code": response.status_code,
        "prediction": result["prediction"],
        "target": result["target"],
        "latency_ms": round(elapsed_ms, 3),
    }


def main():
    results = {}

    for name, base_url in [
        ("local", LOCAL_URL),
        ("container", CONTAINER_URL),
    ]:
        print(f"\nChecking {name} API: {base_url}")

        try:
            health = test_health(base_url)
            prediction = test_prediction(base_url)

            results[name] = {
                "health": health,
                "prediction_test": prediction,
            }

            print("Health:", health)
            print("Prediction:", prediction)

        except requests.RequestException as exc:
            results[name] = {"error": str(exc)}
            print(f"ERROR: {exc}")

    if "local" in results and "container" in results:
        local = results["local"].get("prediction_test")
        container = results["container"].get("prediction_test")

        if local and container:
            difference = abs(
                local["prediction"] - container["prediction"]
            )

            results["comparison"] = {
                "absolute_prediction_difference": round(difference, 6),
                "predictions_match": difference < 0.01,
            }

            print("\nComparison:", results["comparison"])

    output_dir = Path("artifacts/lab11")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "container_validation_report.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
