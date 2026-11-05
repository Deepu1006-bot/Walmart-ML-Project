from walmart_12_utils import *

reference, current = split_data()
model = train_model(reference)

predictions = model.predict(current[FEATURES])
result = metrics(current[TARGET], predictions)

result["concept_drift_detected"] = bool(result["MAE"] > reference[TARGET].std() * 0.20)

print("CONCEPT DRIFT / MODEL PERFORMANCE")
print(result)
save_json(result, "concept_drift.json")
