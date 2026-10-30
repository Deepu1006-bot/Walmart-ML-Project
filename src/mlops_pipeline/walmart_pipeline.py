import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "src/mlops_pipeline/01_load_data.py",
    "src/mlops_pipeline/02_validate_data.py",
    "src/mlops_pipeline/03_preprocess_data.py",
    "src/mlops_pipeline/04_validate_outputs.py",
    "src/mlops_pipeline/05_train_registry.py",
    "src/mlops_pipeline/06_model_evaluation.py",
    "src/mlops_pipeline/07_quality_gate.py",
    "src/mlops_pipeline/09_inference.py",
]

def run_pipeline():
    print("=" * 60, flush=True)
    print("WALMART LAB 7 - MODULAR ML PIPELINE", flush=True)
    print("=" * 60, flush=True)

    for script in SCRIPTS:
        script_path = PROJECT_ROOT / script

        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}", flush=True)
            return 1

        print(f"\n[INFO] Running {script}", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:
            print(f"[ERROR] Pipeline stopped at {script}", flush=True)
            return result.returncode

    print("\n[SUCCESS] Lab 7 pipeline completed.", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(run_pipeline())
