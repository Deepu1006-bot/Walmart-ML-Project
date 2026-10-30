from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"

def run_command(command, description):
    print(f"\n[INFO] {description}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed.", flush=True)
        return result.returncode
    print(f"[SUCCESS] {description} completed.", flush=True)
    return 0

def main():
    print("=" * 60)
    print("WALMART LAB 8 - CI AND AUTOMATED TESTS")
    print("=" * 60)

    if not TESTS_DIR.exists():
        print("[ERROR] tests directory not found.")
        return 1

    steps = [
        (
            [sys.executable, "-m", "pytest", "tests", "-v"],
            "Run automated tests"
        )
    ]

    for command, description in steps:
        code = run_command(command, description)
        if code != 0:
            return code

    print("\n[SUCCESS] Lab 8 runner completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
