from walmart_12_utils import *
from scipy.stats import ks_2samp

reference, current = split_data()
model = train_model(reference)

ref_predictions = model.predict(reference[FEATURES])
current_predictions = model.predict(current[FEATURES])

statistic, pvalue = ks_2samp(ref_predictions, current_predictions)

result = {
    "KS_statistic": float(statistic),
    "p_value": float(pvalue),
    "drift_detected": bool(pvalue < 0.05)
}

print("PREDICTION DRIFT")
print(result)
save_json(result, "prediction_drift.json")
