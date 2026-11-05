from walmart_12_utils import *

reference, current = split_data()
model = train_model(reference)

start = time.perf_counter()
predictions = model.predict(current[FEATURES])
elapsed = time.perf_counter() - start

result = metrics(current[TARGET], predictions)
result["total_inference_time_seconds"] = float(elapsed)
result["average_latency_ms"] = float((elapsed / len(current)) * 1000)
result["throughput_predictions_per_second"] = float(len(current) / max(elapsed, 1e-9))
result["prediction_count"] = int(len(current))
result["failed_predictions"] = int(np.isnan(predictions).sum())

print("PERFORMANCE MONITORING")
print(result)
save_json(result, "performance_monitoring.json")
