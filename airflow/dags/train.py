from airflow.providers.postgres.hooks.postgres import PostgresHook
from mlflow_provider.hooks.client import MLflowClientHook

from mlflow.tracking import MlflowClient
import mlflow.pytorch
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

cast_type = {
    "id": "int64",
    "click": "int64",
    "C1": "int64",
    "banner_pos": "int64",
    "site_id": "int32",
    "site_domain": "int32",
    "site_category": "int32",
    "app_id": "int32",
    "app_domain": "int32",
    "app_category": "int32",
    "device_id": "int32",
    "device_ip": "int32",
    "device_model": "int32",
    "device_type": "int64",
    "device_conn_type": "int64",
    "C14": "int64",
    "C15": "int64",
    "C16": "int64",
    "C17": "int64",
    "C18": "int64",
    "C19": "int64",
    "C20": "int64",
    "C21": "int64",
    "hour": "float64"
}

def train_deepfm_model(**kwargs):
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "admin_password"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    # 1. PostgreSQL에서 feature store 불러오기
    pg_hook = PostgresHook(postgres_conn_id='postgres_conn')
    df = pg_hook.get_pandas_df("SELECT * FROM feature_store_table")
    df = df.astype(cast_type)

    target = ['click']
    dense_features = ['hour']
    sparse_features = [col for col in df.columns if col not in dense_features + target]

    fixlen_feature_columns = (
        [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=8) for feat in sparse_features] +
        [DenseFeat(feat, 1) for feat in dense_features]
    )
    feature_names = get_feature_names(fixlen_feature_columns)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_input = {name: train[name].values for name in feature_names}
    test_input = {name: test[name].values for name in feature_names}

    # 2. MLflow Hook 및 Client 준비
    mlflow_hook = MLflowClientHook(mlflow_conn_id='mlflow_conn')
    mlflow_hook.get_conn()
    tracking_uri = mlflow_hook.base_url
    client = MlflowClient(tracking_uri=tracking_uri)

    # 3. Experiment 확인 및 생성
    experiment_name = "DeepFM_practice"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location="s3://mlflow-artifacts/DeepFM"
        )

    # 4. Run 생성
    run = client.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id

    # 5. 모델 학습
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepFM(
        linear_feature_columns=fixlen_feature_columns,
        dnn_feature_columns=fixlen_feature_columns,
        dnn_dropout=0.7,
        dnn_use_bn=True,
        task='binary',
        device=device
    )

    model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
    model.fit(train_input, train[target].values,
              batch_size=1024, epochs=5, verbose=2, validation_split=0.1)

    # 6. 예측 및 평가
    pred_ans = model.predict(test_input)
    auc = roc_auc_score(test[target].values, pred_ans)

    # 7. Metric, Param, Tag 기록
    client.log_metric(run_id, "test_auc", auc)
    client.log_param(run_id, "batch_size", 1024)
    client.log_param(run_id, "epochs", 5)
    client.set_tag(run_id, "model", "DeepFM")

    # 8. 모델 artifact 저장
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.pytorch.log_model(
            model,
            artifact_path="model"
        )

    # 9. XCom으로 run_id 전달
    kwargs['ti'].xcom_push(key="run_id", value=run_id)

if __name__ == "__main__":
    train_deepfm_model()
