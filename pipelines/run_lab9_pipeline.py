from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    print("=" * 60, flush=True)
    print("WALMART LAB 9 - FAILURE HANDLING TESTS", flush=True)
    print("=" * 60, flush=True)

    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        print("[ERROR] Tests directory not found.")
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v"],
        cwd=PROJECT_ROOT,
        check=False
    )

    if result.returncode != 0:
        print("[ERROR] Lab 9 tests failed.", flush=True)
        return result.returncode

    print("[SUCCESS] Lab 9 failure-handling tests completed.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
