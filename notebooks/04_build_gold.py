# Databricks notebook source
CATALOG = "world_bank_ai"

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.gold.country_indicators
USING DELTA
AS
SELECT
  country_code,
  country_name,
  indicator_code,
  indicator_name,
  year,
  value
FROM {CATALOG}.silver.indicators_clean
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.gold.india_gdp_growth AS
SELECT year, value AS gdp_growth_pct
FROM {CATALOG}.gold.country_indicators
WHERE country_code = 'IND'
  AND indicator_code = 'NY.GDP.MKTP.KD.ZG'
ORDER BY year DESC
""")

display(spark.table(f"{CATALOG}.gold.india_gdp_growth").limit(10))
