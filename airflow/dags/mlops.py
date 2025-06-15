from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from train import train_mlp_model
from validate import validate_model
from deploy import deploy_model

from airflow.operators.empty import EmptyOperator
from mlflow_provider.operators.registry import (
    CreateModelVersionOperator,
    TransitionModelVersionStageOperator,
)

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def choose_deploy_path(**kwargs):
    should_deploy = kwargs['ti'].xcom_pull(task_ids='validate_model', key='should_deploy')
    return "create_model_version" if should_deploy else "skip_deploy"

with DAG(
    dag_id='MLOps',
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_mlp_model
    )

    validate_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model
    )

    branch_task = BranchPythonOperator(
        task_id="choose_deploy_path",
        python_callable=choose_deploy_path
    )

    create_model_version = CreateModelVersionOperator(
        task_id="create_model_version",
        mlflow_conn_id="mlflow_conn",
        name="MLP_practice",
        source="runs:/{{ ti.xcom_pull(task_ids='validate_model', key='validated_run_id') }}/model",
        run_id="{{ ti.xcom_pull(task_ids='validate_model', key='validated_run_id') }}"
    )

    transition_stage = TransitionModelVersionStageOperator(
        task_id="transition_model_stage",
        mlflow_conn_id="mlflow_conn",
        name="MLP_practice",
        version="{{ ti.xcom_pull(task_ids='create_model_version')['model_version']['version'] }}",
        stage="Production",
        archive_existing_versions=True
    )

    skip_deploy = EmptyOperator(
        task_id="skip_deploy"
    )
    
    train_task >> validate_task >> branch_task
    branch_task >> create_model_version >> transition_stage
    branch_task >> skip_deploy