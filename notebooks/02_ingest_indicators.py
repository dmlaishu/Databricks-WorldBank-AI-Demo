# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2 - Ingest structured data → Bronze (Delta Lake)
# MAGIC Pulls indicator values plus country and indicator metadata from the
# MAGIC [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392) and lands them
# MAGIC **as raw JSON** in Delta tables. No cleaning here - that is the silver layer's job (Step 4).
# MAGIC
# MAGIC | Table | Contents | Write mode |
# MAGIC |---|---|---|
# MAGIC | `bronze.indicators_raw` | one row per API record (country × indicator × year) | append - every run is kept, silver dedupes |
# MAGIC | `bronze.countries_raw` | country region / income group | overwrite - reference snapshot |
# MAGIC | `bronze.indicators_meta_raw` | indicator names and definitions | overwrite - reference snapshot |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json, datetime, requests
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://api.worldbank.org/v2"
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))

def fetch_all(path: str, params: dict | None = None) -> list[dict]:
    """GET every page of a World Bank API endpoint. Response shape is [meta, data]."""
    params = {"format": "json", "per_page": 1000, **(params or {})}
    rows, page = [], 1
    while True:
        resp = session.get(f"{BASE_URL}/{path}", params={**params, "page": page}, timeout=120)  # API is occasionally slow
        resp.raise_for_status()
        payload = resp.json()
        if len(payload) == 1:                      # API reports errors as [{"message": [...]}]
            raise ValueError(payload[0].get("message"))
        meta, data = payload
        rows.extend(data or [])
        if page >= int(meta.get("pages") or 1):
            return rows
        page += 1

ingest_ts = datetime.datetime.utcnow()
log_event("ingest_indicators", "started", indicators=len(INDICATORS), countries=len(COUNTRIES))

# COMMAND ----------

# MAGIC %md ## 1. Create bronze tables (with comments - Genie, agents and Catalog Explorer use them)

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.indicators_raw (
    indicator_id STRING    COMMENT 'World Bank indicator code, e.g. NY.GDP.MKTP.CD',
    country_iso3 STRING    COMMENT 'ISO-3 country code',
    year         STRING    COMMENT 'Year as returned by the API (string)',
    raw_json     STRING    COMMENT 'Full API record, unmodified',
    source_url   STRING    COMMENT 'API endpoint the record came from',
    ingest_ts    TIMESTAMP COMMENT 'UTC time of the ingestion run'
  )
  COMMENT 'Raw World Bank indicator values (append-only, one row per API record per run)'
""")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.countries_raw (
    country_iso3 STRING, raw_json STRING, ingest_ts TIMESTAMP
  ) COMMENT 'Raw World Bank country metadata (region, income group, capital, lat/long)'
""")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.indicators_meta_raw (
    indicator_id STRING, raw_json STRING, ingest_ts TIMESTAMP
  ) COMMENT 'Raw World Bank indicator definitions (name, source note, source organization)'
""")

# COMMAND ----------

# MAGIC %md ## 2. Indicator values
# MAGIC One API call per indicator covers all countries (`IND;BGD;...`), so ~14 calls in total.

# COMMAND ----------

country_path = ";".join(COUNTRIES)
records, failures = [], []

for ind_id in INDICATORS:
    path = f"country/{country_path}/indicator/{ind_id}"
    try:
        data = fetch_all(path, {"date": f"{START_YEAR}:{END_YEAR}"})
        records += [
            (ind_id, rec.get("countryiso3code"), rec.get("date"), json.dumps(rec), f"{BASE_URL}/{path}", ingest_ts)
            for rec in data
        ]
        print(f"✓ {ind_id:<24} {len(data):>5} records")
    except Exception as e:
        failures.append(ind_id)
        print(f"✗ {ind_id:<24} {e}")
        log_event("ingest_indicators", "indicator_failed", level="WARNING", indicator=ind_id, error=str(e))

indicators_df = spark.createDataFrame(
    records, "indicator_id string, country_iso3 string, year string, raw_json string, source_url string, ingest_ts timestamp"
)
indicators_df.write.mode("append").saveAsTable(f"{CATALOG}.bronze.indicators_raw")
print(f"\nAppended {len(records)} rows; {len(failures)} indicator(s) failed: {failures}")

# COMMAND ----------

# MAGIC %md ## 3. Country and indicator metadata

# COMMAND ----------

countries = fetch_all("country")   # ~300 rows incl. aggregates like 'South Asia'; silver keeps real countries
spark.createDataFrame(
    [(c["id"], json.dumps(c), ingest_ts) for c in countries], "country_iso3 string, raw_json string, ingest_ts timestamp"
).write.mode("overwrite").saveAsTable(f"{CATALOG}.bronze.countries_raw")

ind_meta = []
for ind_id in INDICATORS:
    try:
        ind_meta += [(ind_id, json.dumps(m), ingest_ts) for m in fetch_all(f"indicator/{ind_id}")]
    except Exception as e:
        print(f"✗ metadata {ind_id}: {e}")
spark.createDataFrame(
    ind_meta, "indicator_id string, raw_json string, ingest_ts timestamp"
).write.mode("overwrite").saveAsTable(f"{CATALOG}.bronze.indicators_meta_raw")

print(f"countries_raw: {len(countries)} rows, indicators_meta_raw: {len(ind_meta)} rows")

# COMMAND ----------

# MAGIC %md ## 4. Quick checks

# COMMAND ----------

display(spark.sql(f"""
  SELECT indicator_id,
         count(*)                                                  AS records,
         count_if(get_json_object(raw_json, '$.value') IS NOT NULL) AS non_null_values,
         count(DISTINCT country_iso3)                              AS countries,
         min(year) AS first_year, max(year) AS last_year
  FROM {CATALOG}.bronze.indicators_raw
  WHERE ingest_ts = '{ingest_ts}'
  GROUP BY indicator_id ORDER BY indicator_id
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta Lake features to show in the demo
# MAGIC Every write is an ACID transaction with a version number. Re-run this notebook and the
# MAGIC history grows; `VERSION AS OF` reads the table as it was before (time travel).

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.bronze.indicators_raw").select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

display(spark.sql(f"""
  SELECT 'version 0' AS snapshot, count(*) AS rows FROM {CATALOG}.bronze.indicators_raw VERSION AS OF 0
  UNION ALL
  SELECT 'current', count(*) FROM {CATALOG}.bronze.indicators_raw
"""))

# COMMAND ----------

log_event("ingest_indicators", "complete", rows=len(records), failed_indicators=failures,
          countries=len(countries), indicator_meta=len(ind_meta))
if failures:
    print(f"⚠ Finished with failures: {failures}. The job still succeeds - silver uses whatever landed.")
