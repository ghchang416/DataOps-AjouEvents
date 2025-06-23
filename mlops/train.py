import logging
from airflow.providers.postgres.hooks.postgres import PostgresHook
from mlflow_provider.hooks.client import MLflowClientHook

from mlflow.tracking import MlflowClient
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

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

def train_mlp_model(**kwargs):
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "admin_password"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    # 1. 데이터 로드
    pg_hook = PostgresHook(postgres_conn_id='postgres_conn')
    df = pg_hook.get_pandas_df("SELECT * FROM feature_store_table")
    df = df.astype(cast_type)
    
    
    target = ['click']
    dense_features = ['hour']
    sparse_features = [col for col in df.columns if col not in dense_features + target]

    # One-hot encoding
    df = pd.get_dummies(df, columns=sparse_features)
    df = df.astype('float32') 
    feature_names = [col for col in df.columns if col not in target]
    X = df[feature_names].values
    y = df[target].values
    
    feature_names = [col for col in df.columns if col not in target]
    import joblib
    joblib.dump(feature_names, "/opt/preprocessor/feature_names.pkl")
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 2. MLflow client
    mlflow_hook = MLflowClientHook(mlflow_conn_id='mlflow_conn')
    mlflow_hook.get_conn()
    tracking_uri = mlflow_hook.base_url
    client = MlflowClient(tracking_uri=tracking_uri)

    # 3. Experiment 생성
    experiment_name = "MLP_practice"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location="s3://mlflow-artifacts/MLP"
        )

    # 4. Run
    run = client.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id

    # 5. 모델 정의 (기초 MLP)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 학습
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

    # 7. 평가
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device)).cpu().numpy()
    auc = roc_auc_score(y_test.numpy(), pred)

    # 8. 기록
    client.log_metric(run_id, "test_auc", auc)
    client.log_param(run_id, "epochs", 5)
    client.log_param(run_id, "model_type", "MLP")

    # 9. 저장
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.pytorch.log_model(
            model,
            artifact_path="model"
        )
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="MLP_practice"
    )
    # 10. XCom 전달
    kwargs['ti'].xcom_push(key="run_id", value=run_id)

if __name__ == "__main__":
    train_mlp_model()
