# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4a - Structured path: Bronze → Silver → Gold
# MAGIC
# MAGIC | Layer | Table | Grain | Purpose |
# MAGIC |---|---|---|---|
# MAGIC | silver | `countries` | country | Real countries only (aggregates like "South Asia" removed), region + income group |
# MAGIC | silver | `indicators` | indicator | Definitions + the short column name used in gold |
# MAGIC | silver | `indicator_values` | country × indicator × year | Parsed, typed, nulls removed, latest ingest wins |
# MAGIC | gold | `country_year_facts` | country × year | **Wide table the SQL agent and dashboards query** |
# MAGIC | gold | `indicator_definitions` | indicator | Lets the agent explain what a column means and cite the source |
# MAGIC
# MAGIC All tables are rebuilt with `CREATE OR REPLACE` - Delta keeps every version (time travel), and
# MAGIC Unity Catalog records table- and column-level lineage from bronze to gold automatically.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
log_event("silver_gold_indicators", "started")

def sql_str(s: str | None) -> str:
    """Escape a Python string for a SQL single-quoted literal."""
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")

# COMMAND ----------

# MAGIC %md ## Silver

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE silver.countries
COMMENT 'World Bank countries (aggregates removed) with region, income group and lending type'
AS SELECT
  country_iso3,
  get_json_object(raw_json, '$.iso2Code')                  AS country_iso2,
  get_json_object(raw_json, '$.name')                      AS country_name,
  trim(get_json_object(raw_json, '$.region.value'))        AS region,
  trim(get_json_object(raw_json, '$.incomeLevel.value'))   AS income_group,
  trim(get_json_object(raw_json, '$.lendingType.value'))   AS lending_type,
  nullif(get_json_object(raw_json, '$.capitalCity'), '')   AS capital_city,
  CAST(nullif(get_json_object(raw_json, '$.latitude'), '')  AS DOUBLE) AS latitude,
  CAST(nullif(get_json_object(raw_json, '$.longitude'), '') AS DOUBLE) AS longitude
FROM bronze.countries_raw
WHERE trim(get_json_object(raw_json, '$.region.value')) <> 'Aggregates'
""")

# COMMAND ----------

spark.createDataFrame(list(INDICATORS.items()), "indicator_id string, column_name string").createOrReplaceTempView("indicator_columns")

spark.sql("""
CREATE OR REPLACE TABLE silver.indicators
COMMENT 'World Bank indicator definitions for the indicators in scope, with their gold column name'
AS SELECT
  m.indicator_id,
  c.column_name,
  get_json_object(m.raw_json, '$.name')               AS indicator_name,
  get_json_object(m.raw_json, '$.sourceNote')         AS definition,
  get_json_object(m.raw_json, '$.sourceOrganization') AS source_organization,
  get_json_object(m.raw_json, '$.source.value')       AS source_database
FROM bronze.indicators_meta_raw m
JOIN indicator_columns c USING (indicator_id)
""")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE silver.indicator_values
COMMENT 'One row per country, indicator and year. Typed, nulls removed, latest ingestion wins.'
AS
WITH latest AS (
  SELECT *
  FROM bronze.indicators_raw
  WHERE country_iso3 IS NOT NULL AND country_iso3 <> ''
  QUALIFY row_number() OVER (PARTITION BY country_iso3, indicator_id, year ORDER BY ingest_ts DESC) = 1
)
SELECT
  l.country_iso3,
  l.indicator_id,
  i.column_name,
  get_json_object(l.raw_json, '$.indicator.value')              AS indicator_name,
  CAST(l.year AS INT)                                           AS year,
  CAST(get_json_object(l.raw_json, '$.value') AS DOUBLE)        AS value,
  nullif(get_json_object(l.raw_json, '$.obs_status'), '')       AS obs_status,
  l.ingest_ts
FROM latest l
JOIN silver.indicators i USING (indicator_id)
WHERE get_json_object(l.raw_json, '$.value') IS NOT NULL
""")

display(spark.sql("""
  SELECT column_name, count(*) AS values, count(DISTINCT country_iso3) AS countries, min(year) AS first_year, max(year) AS last_year
  FROM silver.indicator_values GROUP BY ALL ORDER BY column_name
"""))

# COMMAND ----------

# MAGIC %md ## Gold
# MAGIC The pivot is generated from `INDICATORS` in `00_config`, so adding an indicator there adds a column here.

# COMMAND ----------

pivot_cols = ",\n  ".join(f"max(CASE WHEN v.indicator_id = '{ind}' THEN v.value END) AS {col}" for ind, col in INDICATORS.items())

spark.sql(f"""
CREATE OR REPLACE TABLE gold.country_year_facts
COMMENT 'One row per country and year with World Bank development indicators as columns. Source: World Bank World Development Indicators API. Use this table to answer numeric questions about GDP, growth, poverty, inflation, population, health, education, energy, trade and debt.'
AS SELECT
  v.country_iso3,
  c.country_name,
  c.region,
  c.income_group,
  v.year,
  {pivot_cols}
FROM silver.indicator_values v
JOIN silver.countries c USING (country_iso3)
GROUP BY ALL
""")

# COMMAND ----------

# Column comments - Genie and the SQL agent read these to pick the right column
base_comments = {
    "country_iso3": "ISO-3 country code, e.g. IND, BGD, KEN",
    "country_name": "Country name as published by the World Bank, e.g. India, Bangladesh",
    "region": "World Bank region, e.g. South Asia, Sub-Saharan Africa",
    "income_group": "World Bank income classification, e.g. Lower middle income",
    "year": "Calendar year of the observation",
}
for col, comment in base_comments.items():
    spark.sql(f"ALTER TABLE gold.country_year_facts ALTER COLUMN {col} COMMENT '{sql_str(comment)}'")

for r in spark.table("silver.indicators").collect():
    comment = f"{r.indicator_name} [World Bank indicator {r.indicator_id}]"
    spark.sql(f"ALTER TABLE gold.country_year_facts ALTER COLUMN {r.column_name} COMMENT '{sql_str(comment)}'")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE gold.indicator_definitions
COMMENT 'What each column of gold.country_year_facts means: indicator name, definition and source. Use it to explain or cite a metric.'
AS SELECT column_name, indicator_id, indicator_name, definition, source_organization, source_database
FROM silver.indicators
""")

# COMMAND ----------

# MAGIC %md ## Governance extras: primary key, tags

# COMMAND ----------

governance = [
    "ALTER TABLE gold.country_year_facts ALTER COLUMN country_iso3 SET NOT NULL",
    "ALTER TABLE gold.country_year_facts ALTER COLUMN year SET NOT NULL",
    "ALTER TABLE gold.country_year_facts ADD CONSTRAINT country_year_pk PRIMARY KEY (country_iso3, year)",
    "ALTER TABLE gold.country_year_facts SET TAGS ('layer' = 'gold', 'domain' = 'development_indicators', 'source' = 'world_bank_wdi')",
    "ALTER TABLE gold.indicator_definitions SET TAGS ('layer' = 'gold', 'domain' = 'development_indicators')",
    "ALTER TABLE silver.indicator_values SET TAGS ('layer' = 'silver')",
    "ALTER TABLE silver.countries SET TAGS ('layer' = 'silver')",
]
for stmt in governance:
    try:
        spark.sql(stmt)
    except Exception as e:
        print("⚠", stmt, "→", str(e).splitlines()[0][:150])

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

facts = spark.table("gold.country_year_facts")
n_rows, n_countries = facts.count(), facts.select("country_iso3").distinct().count()
assert n_rows > 0, "gold.country_year_facts is empty"
print(f"gold.country_year_facts: {n_rows} rows, {n_countries} countries")

# Share of non-null values per indicator column - shows data coverage at a glance
coverage = ", ".join(f"round(100 * count({c}) / count(*), 0) AS {c}" for c in INDICATORS.values())
display(spark.sql(f"SELECT {coverage} FROM gold.country_year_facts"))

# COMMAND ----------

# The first half of the demo question, answered from gold
display(spark.sql("""
  SELECT year, country_name, round(gdp_growth_pct, 2) AS gdp_growth_pct
  FROM gold.country_year_facts
  WHERE country_iso3 IN ('IND', 'BGD') AND year >= 2010
  ORDER BY year, country_name
"""))

# COMMAND ----------

log_event("silver_gold_indicators", "complete", gold_rows=n_rows, countries=n_countries,
          silver_values=spark.table("silver.indicator_values").count())
