from walmart_12_utils import *

df = load_data()
split = int(len(df) * 0.80)
train, test = df.iloc[:split], df.iloc[split:]

model = train_model(train)
predictions = model.predict(test[FEATURES])
result = metrics(test[TARGET], predictions)

joblib.dump(model, DEPLOY / "walmart_retrained_model.pkl")
save_json(result, "retrained_model_metrics.json")

print("Retraining completed.")
print("New model saved to deployment/walmart_retrained_model.pkl")
print("Metrics:", result)
