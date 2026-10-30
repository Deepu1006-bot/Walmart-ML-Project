# Lab 9 - Failure Handling Report

## Objective
Verify that invalid or corrupted Walmart input data is detected safely.

## Failure scenarios
- Missing Weekly_Sales values.
- Invalid Date values.
- Missing required columns.

## Expected behavior
Validation should report invalid input and stop unsafe processing.
The original data file must remain unchanged.

## Verification
Run `python src/mlops_pipeline/10_failure_injection.py`.

Record the actual command output and test result before submission.
