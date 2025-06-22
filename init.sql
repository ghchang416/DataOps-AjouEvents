DO
$do$
BEGIN
   IF NOT EXISTS (
       SELECT FROM pg_database WHERE datname = 'mlflow'
   ) THEN
       CREATE DATABASE mlflow;
   END IF;
END
$do$;
