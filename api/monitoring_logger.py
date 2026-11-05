from pathlib import Path
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "api"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "prediction_logs.jsonl"

def log_prediction(input_data, prediction, latency_ms):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": float(prediction),
        "latency_ms": float(latency_ms),
        "status": "success"
    }
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
