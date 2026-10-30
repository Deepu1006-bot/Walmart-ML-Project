from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "mlops_pipeline"
    / "walmart_pipeline.py"
)


def main():
    if not PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Walmart pipeline not found: {PIPELINE_SCRIPT}"
        )

    print("=" * 50, flush=True)
    print("WALMART LIFECYCLE AUTOMATION", flush=True)
    print("=" * 50, flush=True)

    result = subprocess.run(
        [sys.executable, "-u", str(PIPELINE_SCRIPT)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(
            f"[FAIL] Lifecycle stopped with exit code "
            f"{result.returncode}",
            flush=True,
        )
        return result.returncode

    print("[SUCCESS] Walmart lifecycle automation completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
