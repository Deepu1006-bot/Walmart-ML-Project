import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "mlops_pipeline"
    / "walmart_pipeline.py"
)


def main():
    if not PIPELINE_SCRIPT.exists():
        print(f"[ERROR] Pipeline not found: {PIPELINE_SCRIPT}")
        return 1

    print("=" * 60, flush=True)
    print("WALMART PREDICTION PIPELINE", flush=True)
    print("=" * 60, flush=True)

    result = subprocess.run(
        [sys.executable, "-u", str(PIPELINE_SCRIPT)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode == 0:
        print("[SUCCESS] Walmart prediction pipeline completed.")
    else:
        print(
            f"[ERROR] Pipeline failed with exit code "
            f"{result.returncode}"
        )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
