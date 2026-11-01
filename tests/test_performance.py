
import time
import statistics
from fastapi.testclient import TestClient
from api.main import app

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


def test_api_performance_report():
    latencies_ms = []

    with TestClient(app) as client:
        # Warm-up request
        warmup = client.post("/predict", json=SAMPLE)
        assert warmup.status_code == 200

        # Measure 30 sequential requests
        for _ in range(30):
            start = time.perf_counter()
            response = client.post("/predict", json=SAMPLE)
            elapsed = (time.perf_counter() - start) * 1000

            assert response.status_code == 200
            latencies_ms.append(elapsed)

    report = {
        "requests": len(latencies_ms),
        "successful_requests": len(latencies_ms),
        "mean_latency_ms": round(statistics.mean(latencies_ms), 3),
        "median_latency_ms": round(statistics.median(latencies_ms), 3),
        "min_latency_ms": round(min(latencies_ms), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
        "approx_sequential_throughput_rps": round(
            1000 / statistics.mean(latencies_ms), 3
        ),
    }

    import json
    from pathlib import Path

    output = Path("artifacts/lab10")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "api_performance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAPI PERFORMANCE REPORT")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {report_path.resolve()}")
