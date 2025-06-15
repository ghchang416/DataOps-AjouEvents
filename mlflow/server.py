# serve_model.py
import mlflow.pytorch
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time, os
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# MLflow 설정
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "DeepFM"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 글로벌 모델 객체
model = None

def load_latest_model():
    global model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pytorch.load_model(model_uri)
    print(f"✅ Latest model loaded from: {model_uri}")

# FastAPI 앱 생성
app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Prometheus 메트릭 정의
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_OUTPUT = Histogram("prediction_output", "Distribution of prediction values")
PREDICTION_LATENCY = Histogram("prediction_latency", "Latency of prediction")

# 모델 입력 스키마 정의
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

# 서버 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    try:
        load_latest_model()
    except Exception as e:
        print(f"❌ Failed to load model at startup: {e}")

# 예측 API
@app.post("/predict")
async def predict(input: InputData):
    print("🔍 Received input:", input)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start_time = time.time()
    REQUEST_COUNT.inc()

    df = pd.DataFrame([input.dict()])

    # 타입 정제
    int32_feats = [
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category",
        "device_id", "device_ip", "device_model"
    ]
    int64_feats = list(set(df.columns) - set(int32_feats) - {"hour"})
    float64_feats = ["hour"]

    df = df.astype({
        **{col: "int32" for col in int32_feats},
        **{col: "int64" for col in int64_feats},
        **{col: "float64" for col in float64_feats}
    })

    from deepctr_torch.inputs import get_feature_names
    feature_names = get_feature_names(model.feature_dim_dict["feature_columns"])  # 또는 따로 저장해둔 fixlen_feature_columns
    model_input = [df[feat].values for feat in feature_names]
    # 예측
    pred_arr = model.predict(model_input)
    prediction = pred_arr[0].item() if hasattr(pred_arr[0], "item") else float(pred_arr[0])
    print(prediction)

    # Prometheus 기록
    PREDICTION_OUTPUT.observe(prediction)
    PREDICTION_LATENCY.observe(time.time() - start_time)

    return {"prediction": prediction}

# 모델 재로드 API
@app.post("/reload-model")
async def reload_model():
    try:
        load_latest_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
