# Databricks notebook source
CATALOG = "world_bank_ai"
SCHEMAS = ["bronze", "silver", "gold", "evaluation", "monitoring"]

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.monitoring.app_logs (
  event_time TIMESTAMP,
  request_id STRING,
  question STRING,
  route STRING,
  latency_ms DOUBLE,
  status STRING,
  error_message STRING
) USING DELTA
""")

print(f"Created catalog {CATALOG} and schemas: {', '.join(SCHEMAS)}")
