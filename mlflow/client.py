import pandas as pd
import requests
import joblib
import os

int_fields = [
    "id", "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"
]

# 📄 테스트 데이터 로딩 (1줄씩 API로 보내기 위함)
data = pd.read_csv("data.csv", nrows=1, usecols=lambda col: col != "timestamp")
# 💡 모델 전처리용 정보 정의 (학습과 동일하게 맞춰야 함)
target = ['click']
dense_features = ['hour']
sparse_features = [col for col in data.columns if col not in dense_features + target]

# ✅ 저장된 전처리기 로드
ENCODER_DIR = "preprocessors"
encoders = {}
for feat in sparse_features:
    encoders[feat] = joblib.load(os.path.join(ENCODER_DIR, f"{feat}_encoder.pkl"))

scaler = joblib.load(os.path.join(ENCODER_DIR, "minmax_scaler.pkl"))

# ✅ MinMax Scaling (hour은 float로 유지)
data[dense_features] = scaler.transform(data[dense_features])
# ✅ 전처리 적용
for feat in sparse_features:
    data[feat] = encoders[feat].transform(data[feat].astype(str))

        
# # ✅ 예측 API 주소
API_URL = "http://localhost:8000/predict"

# ✅ 한 줄씩 API로 요청
for i, row in data.iterrows():
    input_dict = row[sparse_features + dense_features].to_dict()
    try:
        response = requests.post(API_URL, json=input_dict)
        print(f"[RESPONSE #{i}] Status Code: {response.status_code}")
        result = response.json()
    except Exception as e:
        print(f"[ERROR #{i}] {e}")
        print()

    # (옵션) 느린 API 서버 방지용 지연
    import time; time.sleep(0.1)
