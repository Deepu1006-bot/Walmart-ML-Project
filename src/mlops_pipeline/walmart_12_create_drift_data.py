from walmart_12_utils import *

reference, current = split_data()
reference.to_csv(DRIFT / "reference_data.csv", index=False)
current.to_csv(DRIFT / "current_data.csv", index=False)

print("Reference data:", reference.shape)
print("Current data:", current.shape)
print("Drift datasets created successfully.")
