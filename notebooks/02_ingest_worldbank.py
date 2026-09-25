# Databricks notebook source
import requests
from datetime import datetime, timezone
from pyspark.sql import functions as F

CATALOG = "world_bank_ai"
COUNTRIES = ["IND", "USA", "CHN"]
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "SP.POP.TOTL": "Population",
    "SL.UEM.TOTL.ZS": "Unemployment",
    "FP.CPI.TOTL.ZG": "Inflation",
}

records = []
for country in COUNTRIES:
    for indicator_code, indicator_name in INDICATORS.items():
        url = (
            f"https://api.worldbank.org/v2/country/{country}/indicator/"
            f"{indicator_code}?format=json&date=2000:2025&per_page=1000"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        for row in (payload[1] if len(payload) > 1 and payload[1] else []):
            records.append({
                "country_code": row["countryiso3code"],
                "country_name": row["country"]["value"],
                "indicator_code": indicator_code,
                "indicator_name": indicator_name,
                "year": int(row["date"]),
                "value": row["value"],
                "source_url": url,
                "ingested_at": datetime.now(timezone.utc),
            })

df = spark.createDataFrame(records)
(df.write.format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.indicators_raw"))

display(df.orderBy(F.desc("year")).limit(20))
