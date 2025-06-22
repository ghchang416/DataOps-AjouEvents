import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from io import StringIO
from datetime import datetime, timedelta

def raw_data_upload_to_minio(**context):
    try:
        # Get execution date from context
        start_time_str = str(context['execution_date'])[:16]  # '2025-04-30 17:10'
        
        # 문자열을 다시 datetime 객체로 파싱
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
        end_time = start_time + timedelta(minutes=5)

        # Read source data
        df = pd.read_csv('/opt/data/raw.csv', parse_dates=['timestamp'])

        # Filter data for the current interval
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            print(f"[{start_time}] No data for this interval.")
            return

        # Initialize S3 hook
        s3_hook = S3Hook(aws_conn_id='minio_conn')

        # Prepare upload parameters
        bucket_name = 'raw-data'
        object_key = f"{start_time.strftime('%Y%m%d_%H%M')}.csv"

        # Convert DataFrame to CSV
        csv_buffer = StringIO()
        filtered_df.to_csv(csv_buffer, index=False)

        # Upload to MinIO
        s3_hook.load_string(
            string_data=csv_buffer.getvalue(),
            key=object_key,
            bucket_name=bucket_name,
            replace=True
        )

        print(f"[{start_time}] Successfully uploaded {len(filtered_df)} rows to {bucket_name}/{object_key}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
