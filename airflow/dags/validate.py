from mlflow.tracking import MlflowClient
from mlflow_provider.hooks.client import MLflowClientHook
from mlflow.exceptions import MlflowException
import logging

def validate_model(**context):
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')
    if not run_id:
        raise ValueError("❌ run_id not found in XCom")

    # MLflowClient 직접 사용
    mlflow_hook = MLflowClientHook(mlflow_conn_id='mlflow_conn')
    mlflow_hook.get_conn()  # base_url 설정용
    client = MlflowClient(tracking_uri=mlflow_hook.base_url)

    # 1. 새 모델 AUC 가져오기
    try:
        run_data = client.get_run(run_id).data
        new_auc = float(run_data.metrics.get("test_auc", -1.0))
        logging.info(f"✅ New model AUC: {new_auc}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to get new model AUC: {str(e)}")

    # 2. 프로덕션 모델 AUC 가져오기
    try:
        versions = client.get_latest_versions("DeepFM", stages=["Production"])
        prod_run_id = versions[0].run_id

        prod_data = client.get_run(prod_run_id).data
        prod_auc = float(prod_data.metrics.get("test_auc", -1.0))
        logging.info(f"✅ Production model AUC: {prod_auc}")
    except (IndexError, KeyError, MlflowException):
        logging.info("⚠️ No production model found — accepting new model by default.")
        prod_auc = -1.0
    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch production model AUC: {str(e)}")

    # 3. 비교
    is_better = new_auc > prod_auc
    logging.info(f"📊 is_better = {is_better} (new > prod)")

    # 4. 결과 XCom 저장
    context['ti'].xcom_push(key="should_deploy", value=is_better)
    context['ti'].xcom_push(key="validated_run_id", value=run_id)

if __name__ == "__main__":
    validate_model()
