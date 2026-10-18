# Databricks notebook source
# MAGIC %md
# MAGIC # 00_config - shared settings
# MAGIC Every other notebook starts with `%run ./00_config`, so names live in one place.
# MAGIC Change `CATALOG` here if `wb_demo` is taken in your metastore.

# COMMAND ----------

CATALOG = "wb_demo"

SCHEMAS = {
    "bronze": "Raw data exactly as received from World Bank APIs (JSON rows, PDFs).",
    "silver": "Cleaned, typed, de-duplicated data (one row per country/indicator/year; parsed doc text).",
    "gold":   "Business-ready tables consumed by BI and AI (country_year_facts, doc_chunks).",
    "ai":     "AI assets: UC functions (agent tools), vector indexes, registered agent models.",
    "ops":    "Operational data: app logs, eval sets/results, inference tables.",
}

RAW_DOCS_VOLUME = f"{CATALOG}.bronze.raw_docs"
RAW_DOCS_PATH = f"/Volumes/{CATALOG}/bronze/raw_docs"

# Account-level groups (UC grants only work on account groups, not workspace-local ones)
GROUP_ANALYSTS = "wb_analysts"
GROUP_AI_APP = "wb_ai_app"

# Demo scope - kept small so every step runs fast live
COUNTRIES = [
    "IND", "BGD", "PAK", "LKA", "NPL",   # South Asia
    "CHN", "IDN", "VNM", "PHL",          # East Asia
    "NGA", "KEN", "ETH", "ZAF", "GHA",   # Sub-Saharan Africa
    "BRA", "MEX", "COL",                 # Latin America
    "EGY", "TUR", "USA",                 # MENA / Europe-Central Asia / benchmark
]
INDICATORS = {
    "NY.GDP.MKTP.CD":    "gdp_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NY.GDP.PCAP.CD":    "gdp_per_capita_usd",
    "FP.CPI.TOTL.ZG":    "inflation_pct",
    "SL.UEM.TOTL.ZS":    "unemployment_pct",
    "SI.POV.DDAY":       "poverty_rate_pct",
    "SP.DYN.LE00.IN":    "life_expectancy_years",
    "SP.POP.TOTL":       "population",
    "SE.ADT.LITR.ZS":    "literacy_rate_pct",
    "EN.GHG.CO2.PC.CE.AR5": "co2_per_capita_t",
    "EG.ELC.ACCS.ZS":    "electricity_access_pct",
    "NE.TRD.GNFS.ZS":    "trade_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
    "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
}
START_YEAR, END_YEAR = 2000, 2025

# Models / serving (Databricks-hosted Foundation Model APIs)
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"   # verified in this workspace; any chat endpoint under Serving works
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
VS_ENDPOINT = "wb_demo_vs_endpoint"
VS_INDEX = f"{CATALOG}.ai.doc_chunks_index"
AGENT_MODEL_NAME = f"{CATALOG}.ai.worldbank_agent"

print(f"Config loaded: catalog={CATALOG}, {len(COUNTRIES)} countries, {len(INDICATORS)} indicators")

# COMMAND ----------

# MAGIC %md
# MAGIC ### App logger → `ops.app_logs`
# MAGIC Used by every notebook so key events land in a governed Delta table (Step 14).

# COMMAND ----------

import logging, json, uuid, datetime

_RUN_ID = str(uuid.uuid4())

def log_event(component: str, event: str, level: str = "INFO", **details):
    """Write one structured event to Python logging and to ops.app_logs."""
    logging.getLogger(component).log(getattr(logging, level, logging.INFO), f"{event} {details}")
    try:
        spark.createDataFrame(
            [(datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None), _RUN_ID, component, level, event,
              json.dumps(details, default=str))],
            "event_ts timestamp, run_id string, component string, level string, event string, details string",
        ).write.mode("append").saveAsTable(f"{CATALOG}.ops.app_logs")
    except Exception as e:  # logging must never break the pipeline (e.g. before Step 1 has run)
        print(f"[log_event skipped] {e}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
