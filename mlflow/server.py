# serve_model.py
import mlflow.pytorch
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import time
from prometheus_client import Counter, Histogram
import requests
from prometheus_fastapi_instrumentator import Instrumentator
import torch

# MLflow 설정
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "MLP_practice"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 글로벌 모델 객체
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_latest_model():
    global model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()
    print(f"✅ Latest model loaded from: {model_uri}")

# FastAPI 앱 생성
app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Prometheus 메트릭 정의
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_OUTPUT = Histogram("prediction_output", "Distribution of prediction values")
PREDICTION_LATENCY = Histogram("prediction_latency", "Latency of prediction")

# 모델 입력 스키마 정의 (client.py와 동일)
class InputData(BaseModel):
    id: int
    C1: int
    banner_pos: int
    site_id: int
    site_domain: int
    site_category: int
    app_id: int
    app_domain: int
    app_category: int
    device_id: int
    device_ip: int
    device_model: int
    device_type: int
    device_conn_type: int
    C14: int
    C15: int
    C16: int
    C17: int
    C18: int
    C19: int
    C20: int
    C21: int
    hour: float

@app.on_event("startup")
async def startup_event():
    try:
        load_latest_model()
    except Exception as e:
        print(f"❌ Failed to load model at startup: {e}")

@app.post("/alert")
async def receive_alert(req: Request):
    data = await req.json()

    response = requests.post(
        "http://airflow-webserver:8080/api/v1/dags/MLOps/dagRuns",
        auth=("airflow", "airflow"),
        json={"conf": {}, "dag_run_id": "alert-triggered"}
    )
    return {"status": response.status_code}


@app.post("/predict")
async def predict(input: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start_time = time.time()
    REQUEST_COUNT.inc()

    df = pd.DataFrame([input.dict()])

    # 입력 데이터 정제
    # int32_feats = [
    #     "site_id", "site_domain", "site_category",
    #     "app_id", "app_domain", "app_category",
    #     "device_id", "device_ip", "device_model"
    # ]
    # int64_feats = list(set(df.columns) - set(int32_feats) - {"hour"})
    # float64_feats = ["hour"]

    # df = df.astype({
    #     **{col: "int32" for col in int32_feats},
    #     **{col: "int64" for col in int64_feats},
    #     **{col: "float64" for col in float64_feats}
    # })
    import joblib
    with open("preprocessors/feature_names.pkl", "rb") as f:
        feature_names = joblib.load(f)
        # ✅ One-hot encoding

    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        # 0.0으로 채운 열들을 한 번에 생성
        missing_df = pd.DataFrame(0.0, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_df], axis=1)

    # ✅ 컬럼 순서 학습 기준에 맞춤
    df = df[feature_names]

    # ✅ float32로 변환
    df = df.astype("float32")

    # 예측
    input_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()[0][0]

    # Prometheus 기록
    PREDICTION_OUTPUT.observe(pred)
    PREDICTION_LATENCY.observe(time.time() - start_time)

    return {"prediction": float(pred)}

@app.post("/reload-model")
async def reload_model():
    try:
        load_latest_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
