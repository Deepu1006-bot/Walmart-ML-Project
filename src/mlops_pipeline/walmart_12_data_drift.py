from walmart_12_utils import *

reference = pd.read_csv(DRIFT / "reference_data.csv")
current = pd.read_csv(DRIFT / "current_data.csv")

columns = FEATURES
results = {}

def calculate_psi(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 0.0001, None)
    a = np.clip(a, 0.0001, None)
    return float(np.sum((a - e) * np.log(a / e)))

for col in columns:
    results[col] = calculate_psi(reference[col], current[col])

print("DATA DRIFT RESULTS")
for col, value in results.items():
    status = "Drift detected" if value >= 0.20 else "No significant drift"
    print(f"{col}: PSI={value:.4f} - {status}")

save_json(results, "data_drift.json")
