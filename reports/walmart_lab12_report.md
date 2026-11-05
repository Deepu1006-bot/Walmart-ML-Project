# Walmart Lab 12: Monitoring, Drift Detection, and Retraining Workflows

## Aim
Monitor the Walmart Weekly Sales prediction model, detect drift, measure performance, retrain the model, and compare model results.

## Dataset
- Rows: 6435
- Target: Weekly_Sales
- Problem: Regression

## Monitoring results

### data_drift.json
```json
{
    "Store": 2.7007537362339644e-05,
    "Holiday_Flag": 0.0,
    "Temperature": 0.07733490153297712,
    "Fuel_Price": 3.352185566409681,
    "CPI": 2.8938580240786997,
    "Unemployment": 0.5355412967298188,
    "Year": 4.376984112348762,
    "Month": 1.4318064837748998,
    "Week": 0.9381231081374547
}
```

### prediction_drift.json
```json
{
    "KS_statistic": 0.09381004789574236,
    "p_value": 8.349584931826737e-11,
    "drift_detected": true
}
```

### concept_drift.json
```json
{
    "MAE": 157575.3643355256,
    "RMSE": 280657.55185251316,
    "R2": 0.7267996968520645,
    "concept_drift_detected": true
}
```

### performance_monitoring.json
```json
{
    "MAE": 157575.3643355256,
    "RMSE": 280657.5518525132,
    "R2": 0.7267996968520645,
    "total_inference_time_seconds": 0.04386239999439567,
    "average_latency_ms": 0.022714862762504232,
    "throughput_predictions_per_second": 44024.038817910674,
    "prediction_count": 1931,
    "failed_predictions": 0
}
```

### retrained_model_metrics.json
```json
{
    "MAE": 102867.08446472396,
    "RMSE": 179042.21141468443,
    "R2": 0.8871448449872642
}
```

### model_comparison.json
```json
{
    "baseline_model": {
        "MAE": 167409.90375889663,
        "RMSE": 290174.4696010306,
        "R2": 0.7035653725865916
    },
    "retrained_model": {
        "MAE": 102867.08446472396,
        "RMSE": 179042.21141468445,
        "R2": 0.8871448449872642
    },
    "better_model_by_RMSE": "Retrained"
}
```

## Conclusion
The monitoring workflow was executed. Refer to the generated JSON results for measured drift and model metrics.
