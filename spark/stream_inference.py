from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, FloatType
import joblib
import requests
import redis
import json
from concurrent.futures import ThreadPoolExecutor
import os

KAFKA_BOOTSTRAP_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "ajou-ad-clicks-topic")  # default값 지정도 가능
KAFKA_SECURITY_PROTOCOL = os.environ["KAFKA_SECURITY_PROTOCOL"]
KAFKA_SASL_MECHANISM = os.environ["KAFKA_SASL_MECHANISM"]
KAFKA_JAAS_CONFIG = os.environ["KAFKA_JAAS_CONFIG"]

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# SparkSession 생성
spark = SparkSession.builder \
    .appName("KafkaPredictionConsumer") \
    .getOrCreate()

# Redis 연결
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ✅ 전처리기 로딩
ENCODER_DIR = "/opt/preprocessors"  # 볼륨 경로
sparse_features = [
    "id", "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"
] # csv와 동일하게 작성
dense_features = ['hour']

encoders = {feat: joblib.load(f"{ENCODER_DIR}/{feat}_encoder.pkl") for feat in sparse_features}
scaler = joblib.load(f"{ENCODER_DIR}/minmax_scaler.pkl")

# ✅ 메시지 스키마 정의 (Kafka 메시지는 JSON 문자열로 들어올 것)
schema = StructType()
for feat in sparse_features + dense_features:
    schema = schema.add(feat, StringType() if feat != 'hour' else FloatType())

# ✅ Kafka 스트리밍 읽기
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("kafka.security.protocol", KAFKA_SECURITY_PROTOCOL) \
    .option("kafka.sasl.mechanism", KAFKA_SASL_MECHANISM) \
    .option("kafka.sasl.jaas.config", KAFKA_JAAS_CONFIG) \
    .load()

# ✅ Kafka value는 JSON 문자열 → struct로 변환
json_df = df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

from concurrent.futures import ThreadPoolExecutor
import threading

# 🔧 병렬 요청 보낼 쓰레드 수
MAX_WORKERS = 8

def send_prediction_request(row):
    try:
        row_dict = row.to_dict()
        raw_id = str(row_dict.pop("raw_id")).split(".")[0]  # ✅ request에서는 제거
        response = requests.post("http://mlflow:8000/predict", json=row_dict, timeout=2)
        result = response.json()
        redis_client.set(f"user:{raw_id}", json.dumps(result))
    except Exception as e:
        print(f"[ERROR] Failed for user_id={row.get('raw_id')}: {e}")

def process_batch(batch_df, epoch_id):
    pdf = batch_df.toPandas()
    if pdf.empty:
        return

    # ✅ 원본 ID 따로 저장 (request에 포함 X)
    raw_ids = pdf["id"].copy()

    # 🔁 인코딩
    for feat in sparse_features:
        pdf[feat] = encoders[feat].transform(pdf[feat])
    pdf[dense_features] = scaler.transform(pdf[dense_features])

    # ✅ row에 raw_id 함께 넘기기
    pdf["raw_id"] = raw_ids

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(send_prediction_request, [row for _, row in pdf.iterrows()])



# ✅ foreachBatch로 실행
query = json_df.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()