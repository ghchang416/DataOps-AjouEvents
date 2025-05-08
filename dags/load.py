from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import timedelta

def build_feature_store(**context):
    # 1. execution_date 받아오기
    execution_date = context['execution_date']
    start_time = pd.to_datetime(str(execution_date)).tz_convert(None)

    # 2. schedule_interval을 이용해 end_time 계산
    dag = context['dag']
    schedule_interval = dag.schedule_interval  # ex) timedelta(minutes=5)

    if isinstance(schedule_interval, str):
        # cron 표현식이라면 따로 처리해야 하지만, 우리는 timedelta(minutes=5)니까 그대로 사용
        raise ValueError("schedule_interval이 timedelta 형식이 아닙니다. cron형이면 추가 파싱 필요.")

    end_time = start_time + schedule_interval

    # 3. PostgresHook 연결
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    # 4. start_time ~ end_time 사이 데이터 조회
    query = f"""
    SELECT * FROM raw_data_table
    WHERE timestamp >= '{start_time}' AND timestamp < '{end_time}'
    """
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()

    raw_data = pd.DataFrame(data, columns=columns)

    if raw_data.empty:
        print(f"No data found between {start_time} and {end_time}")
        return

    # 5. Feature Engineering
    target = ['click']
    dense_features = ['hour']
    sparse_features = [col for col in raw_data.columns if col not in dense_features + target + ['timestamp']]

    os.makedirs("preprocessors", exist_ok=True)

    for feat in sparse_features:
        encoder_path = f"preprocessors/{feat}_encoder.pkl"
        if os.path.exists(encoder_path):
            lbe = joblib.load(encoder_path)
            raw_data[feat] = lbe.transform(raw_data[feat].astype(str))
        else:
            lbe = LabelEncoder()
            raw_data[feat] = lbe.fit_transform(raw_data[feat].astype(str))
            joblib.dump(lbe, encoder_path)

    scaler_path = "preprocessors/minmax_scaler.pkl"
    if os.path.exists(scaler_path):
        mms = joblib.load(scaler_path)
        raw_data[dense_features] = mms.transform(raw_data[dense_features])
    else:
        mms = MinMaxScaler()
        raw_data[dense_features] = mms.fit_transform(raw_data[dense_features])
        joblib.dump(mms, scaler_path)

    # 6. Feature Store 테이블에 저장
    raw_data.to_sql('feature_store_table', con=pg_hook.get_sqlalchemy_engine(), if_exists='append', index=False)
