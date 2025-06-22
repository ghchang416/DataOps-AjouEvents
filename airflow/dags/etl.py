from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
from etl.extract import raw_data_upload_to_minio
from etl.load import build_feature_store

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='ETL',
    default_args=default_args,
    start_date=datetime(2025, 4, 1, 0, 0, 0),
    end_date=datetime(2025, 4, 1, 0, 5, 0),
    schedule_interval="*/5 * * * *",
    catchup=True,
    max_active_runs=1,
) as dag:

    filter_and_upload = PythonOperator(
        task_id='raw_data_upload',
        python_callable=raw_data_upload_to_minio
    )
    
    spark_load_to_postgres = SparkSubmitOperator(
        task_id='spark_load_to_postgres',
        name='Data_Transform',
        executor_memory='1G',
        total_executor_cores=2,
        application='/opt/etl/transform.py',
        conn_id='spark_conn', 
        application_args=["{{ execution_date }}"],
        packages='org.postgresql:postgresql:42.7.2,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262',
    )
    
    build_feature_store_task = PythonOperator(
        task_id='build_feature_store',
        python_callable=build_feature_store
    )

    filter_and_upload >> spark_load_to_postgres >> build_feature_store_task
