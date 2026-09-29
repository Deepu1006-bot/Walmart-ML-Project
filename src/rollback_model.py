import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "Walmart_Weekly_Sales_Production_Model"


def get_versions(client):
    return client.search_model_versions(
        "name='" + MODEL_NAME + "'"
    )


def rollback_demo():

    print(
        "\n========================================"
    )
    print(
        "Experiment 6 - Model Rollback Demonstration"
    )
    print(
        "========================================"
    )

    client = MlflowClient()

    versions = get_versions(client)

    production_models = [
        v for v in versions
        if v.current_stage == "Production"
    ]

    archived_models = [
        v for v in versions
        if v.current_stage == "Archived"
    ]

    if not production_models:
        print(
            "[ERROR] No Production model found."
        )
        return

    if not archived_models:
        print(
            "[ERROR] No Archived model available "
            "for rollback demonstration."
        )
        return

    original_production = production_models[0]

    rollback_target = sorted(
        archived_models,
        key=lambda v: int(v.version)
    )[-1]

    print(
        f"[INFO] Current Production: "
        f"Version {original_production.version}"
    )

    print(
        f"[INFO] Rollback test target: "
        f"Version {rollback_target.version}"
    )

    # Step 1: Temporarily promote archived version
    print(
        f"\n[INFO] Temporarily promoting Version "
        f"{rollback_target.version} to Production..."
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=rollback_target.version,
        stage="Production",
        archive_existing_versions=False
    )

    # Archive the original production version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=original_production.version,
        stage="Archived",
        archive_existing_versions=False
    )

    print(
        f"[INFO] Version {rollback_target.version} "
        "is temporarily in Production."
    )

    # Step 2: Roll back to original Production version
    print(
        f"\n[INFO] Rolling back to original "
        f"Production Version {original_production.version}..."
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=original_production.version,
        stage="Production",
        archive_existing_versions=False
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=rollback_target.version,
        stage="Archived",
        archive_existing_versions=False
    )

    # Step 3: Verify final state
    final_versions = get_versions(client)

    final_production = [
        v for v in final_versions
        if v.current_stage == "Production"
    ]

    if (
        len(final_production) == 1
        and str(final_production[0].version)
        == str(original_production.version)
    ):

        print(
            f"\n[SUCCESS] Rollback completed successfully."
        )

        print(
            f"[SUCCESS] Version "
            f"{original_production.version} "
            "is restored to Production."
        )

    else:

        print(
            "\n[ERROR] Rollback verification failed."
        )

    print(
        "\n========================================"
    )


if __name__ == "__main__":
    rollback_demo()