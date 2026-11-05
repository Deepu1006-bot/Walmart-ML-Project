from walmart_12_utils import *

df = load_data()
test = df.iloc[int(len(df) * 0.80):]
baseline_train = df.iloc[:int(len(df) * 0.70)]
retrain_train = df.iloc[:int(len(df) * 0.80)]

baseline = train_model(baseline_train)
retrained = train_model(retrain_train)

baseline_metrics = metrics(test[TARGET], baseline.predict(test[FEATURES]))
retrained_metrics = metrics(test[TARGET], retrained.predict(test[FEATURES]))

comparison = {
    "baseline_model": baseline_metrics,
    "retrained_model": retrained_metrics,
    "better_model_by_RMSE": (
        "Retrained" if retrained_metrics["RMSE"] < baseline_metrics["RMSE"]
        else "Baseline"
    )
}

print("MODEL COMPARISON")
print(comparison)
save_json(comparison, "model_comparison.json")
