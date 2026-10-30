# Lab 8 - Continuous Integration and Automated Testing

## Objective
Set up a continuous integration workflow and automated tests for the Walmart ML pipeline.

## CI Workflow
Workflow file: `.github/workflows/lab8_ci.yml`

The workflow is configured to:
1. Check out the repository.
2. Set up Python 3.11.
3. Install the required Python dependencies.
4. Load and validate the Walmart dataset.
5. Preprocess the data.
6. Train the Random Forest model.
7. Evaluate the model.
8. Generate predictions.
9. Run automated tests.
10. Upload Lab 7 artifacts when available.

## Automated Test Results
The local test suite collected and passed 13 tests.

Test categories:
- Data validation
- Missing-value handling
- Duplicate-row detection
- Invalid-date detection
- Negative-sales detection
- Failure handling
- Preprocessed data verification
- Model prediction checks

## Local Execution
Run the automated tests:

    python .\pipelines\run_lab8_pipeline.py

## Result
The local automated test suite completed successfully with 13 passed tests.

Note: The workflow file has been created locally. A successful GitHub Actions run must be verified separately after pushing the changes.
