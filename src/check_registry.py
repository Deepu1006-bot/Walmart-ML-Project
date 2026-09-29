from mlflow.tracking import MlflowClient

MODEL_NAME = "Walmart_Weekly_Sales_Production_Model"

client = MlflowClient()

versions = client.search_model_versions(
    "name='Walmart_Weekly_Sales_Production_Model'"
)

print("\n========================================")
print("Final Model Registry State")
print("========================================")

for v in versions:
    print(
        f"Version {v.version} | "
        f"Stage: {v.current_stage} | "
        f"Run: {v.run_id}"
    )

print("========================================")