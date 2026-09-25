# Databricks notebook source
# Ingest unstructured World Bank indicator metadata as trusted public text.
# This keeps the demo fully reproducible without bundling copyrighted reports.

import requests
from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

CATALOG = "world_bank_ai"
INDICATORS = [
    "NY.GDP.MKTP.KD.ZG",
    "SP.POP.TOTL",
    "SL.UEM.TOTL.ZS",
    "FP.CPI.TOTL.ZG",
]

rows = []
for indicator in INDICATORS:
    url = f"https://api.worldbank.org/v2/indicator/{indicator}?format=json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    item = payload[1][0]
    text = "\n\n".join([
        f"Indicator: {item.get('name', indicator)}",
        f"Definition and methodology: {item.get('sourceNote') or ''}",
        f"Source organization: {item.get('sourceOrganization') or ''}",
    ]).strip()
    rows.append({
        "document_id": indicator,
        "title": f"World Bank indicator metadata — {item.get('name', indicator)}",
        "source_url": url,
        "report_text": text,
        "ingested_at": datetime.now(timezone.utc),
    })

docs = spark.createDataFrame(rows)
(docs.write.format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.bronze.documents_raw"))

@F.udf(ArrayType(StringType()))
def chunk_text(text):
    if not text:
        return []
    words = text.split()
    size, overlap = 250, 40
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += max(1, size - overlap)
    return chunks

chunks = (
    docs.withColumn("chunks", chunk_text("report_text"))
        .select(
            "document_id", "title", "source_url",
            F.posexplode("chunks").alias("chunk_id", "chunk_text")
        )
)

(chunks.write.format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.silver.document_chunks"))

display(chunks)
