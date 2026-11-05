from walmart_12_utils import *
import json

files = [
    "data_drift.json",
    "prediction_drift.json",
    "concept_drift.json",
    "performance_monitoring.json",
    "retrained_model_metrics.json",
    "model_comparison.json"
]

report = {
    "project": "Walmart Weekly Sales Prediction",
    "target": TARGET,
    "problem_type": "Regression",
    "monitoring_results": {}
}

for filename in files:
    path = OUT / filename
    if path.exists():
        with open(path) as f:
            report["monitoring_results"][filename] = json.load(f)

save_json(report, "walmart_lab12_summary.json")

report_path = ROOT / "reports/walmart_lab12_report.md"
with open(report_path, "w") as f:
    f.write("# Walmart Lab 12: Monitoring, Drift Detection, and Retraining Workflows\n\n")
    f.write("## Aim\nMonitor the Walmart Weekly Sales prediction model, detect drift, ")
    f.write("measure performance, retrain the model, and compare model results.\n\n")
    f.write("## Dataset\n- Rows: 6435\n- Target: Weekly_Sales\n- Problem: Regression\n\n")
    f.write("## Monitoring results\n")
    for filename in files:
        path = OUT / filename
        if path.exists():
            f.write(f"\n### {filename}\n```json\n")
            f.write(path.read_text())
            f.write("\n```\n")
    f.write("\n## Conclusion\nThe monitoring workflow was executed. ")
    f.write("Refer to the generated JSON results for measured drift and model metrics.\n")

print("Lab 12 report generated:", report_path)
