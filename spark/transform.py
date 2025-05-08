from pyspark.sql import SparkSession
import sys
from datetime import datetime
import logging
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_IP_ADDRESS = socket.gethostbyname("minio")
logging.info(f"MINIO_IP_ADDRESS: {MINIO_IP_ADDRESS}")

def main():
    try:
        # Get execution date from arguments
        execution_date_str = sys.argv[1]
        execution_date = datetime.fromisoformat(execution_date_str.replace("Z", "+00:00"))
        filename = execution_date.strftime("%Y%m%d_%H%M")
        
        logger.info(f"Processing file for date: {filename}")
        s3_path = f"s3a://raw-data/{filename}.csv"

        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("ETL Transform") \
            .config("fs.s3a.access.key", "admin") \
            .config("fs.s3a.secret.key", "admin") \
            .config("fs.s3a.endpoint",  f"http://{MINIO_IP_ADDRESS}:9000") \
            .config("fs.s3a.connection.ssl.enabled", "false") \
            .config("fs.s3a.path.style.access", "true") \
            .config("fs.s3a.connection.timeout", "30000") \
            .config("fs.s3a.connection.establish.timeout", "30000") \
            .getOrCreate()

        # Read CSV from MinIO
        logger.info(f"Reading CSV from: {s3_path}")
        df = spark.read.option("header", "true").csv(s3_path)

        if df.rdd.isEmpty():
            logger.info("CSV file is empty - stopping")
            return

        # Transform data (add your transformations here)
        # Example: df = df.withColumn("new_column", ...)

        # Write to PostgreSQL
        logger.info("Starting PostgreSQL save")
        df.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://postgres:5432/airflow") \
            .option("dbtable", "transformed_data") \
            .option("user", "airflow") \
            .option("password", "airflow") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()

        logger.info("PostgreSQL save completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
