from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "walmart_pipeline.py"

def main():
    print("=" * 60, flush=True)
    print("WALMART LAB 7 - PIPELINE RUNNER", flush=True)
    print("=" * 60, flush=True)

    if not PIPELINE_SCRIPT.exists():
        print(f"[ERROR] Pipeline script not found: {PIPELINE_SCRIPT}")
        return 1

    result = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False
    )

    if result.returncode != 0:
        print("[ERROR] Lab 7 pipeline failed.")
        return result.returncode

    print("[SUCCESS] Lab 7 pipeline runner completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
