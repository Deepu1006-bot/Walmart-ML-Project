
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "src/mlops_pipeline/01_load_data.py",
    "src/mlops_pipeline/02_validate_data.py",
    "src/mlops_pipeline/03_preprocess_data.py",
    "src/mlops_pipeline/04_train_model.py",
    "src/mlops_pipeline/05_evaluate_model.py",
    "src/mlops_pipeline/06_inference.py",
    "src/mlops_pipeline/07_log_artifacts.py",
]


def run_pipeline():
    print("=" * 60)
    print("WALMART LAB 7 - MODULAR ML PIPELINE")
    print("=" * 60)

    for script in SCRIPTS:
        print(f"\n[INFO] Running {script}")

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            print(f"[ERROR] Pipeline stopped at {script}")
            return result.returncode

    print("\n[SUCCESS] Lab 7 pipeline completed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
