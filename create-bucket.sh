#!/bin/sh
set -e

# MinIO 서버를 백그라운드로 시작
minio server /data --console-address ":9001" &
MINIO_PID=$!

# health check (최대 30초 기다림)
echo "Waiting for MinIO to be healthy..."
for i in $(seq 1 30); do
    if curl -s http://localhost:9000/minio/health/live >/dev/null; then
        break
    fi
    sleep 1
done

# mc 세팅 및 bucket 자동 생성
mc alias set myminio http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb -p myminio/raw-data || true            # 이미 있으면 무시
mc mb -p myminio/mlflow-artifacts || true    # mlflow-artifacts 버킷 생성

# MLP "디렉토리" (prefix) 미리 생성
mc cp --no-color /etc/hosts myminio/mlflow-artifacts/MLP/.keep || true
# (빈 파일을 /MLP/ 아래에 하나 두면 prefix가 생성됨. 나중에 자동으로 없어져도 상관 없음)

wait $MINIO_PID
