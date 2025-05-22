from mlflow.tracking import MlflowClient

def deploy_model(**kwargs):

    should_deploy = kwargs['ti'].xcom_pull(task_ids='validate_model', key='should_deploy')
    run_id = kwargs['ti'].xcom_pull(task_ids='validate_model', key='validated_run_id')
    client = MlflowClient()

    if should_deploy:
        model_uri = f"runs:/{run_id}/model"
        registered = client.register_model(model_uri, "DeepFM")
        client.transition_model_version_stage(
            name="DeepFM",
            version=registered.version,
            stage="Production",
            archive_existing_versions=True
        )
        print("New model deployed to Production")
    else:
        print("Model did not outperform current Production. Skipping deployment.")
