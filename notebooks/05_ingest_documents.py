# Databricks notebook source
# Demo ingestion for unstructured World Bank report text.
# In production, populate report_text from approved World Bank report files/URLs.

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

CATALOG = "world_bank_ai"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.documents_raw (
  document_id STRING,
  title STRING,
  source_url STRING,
  report_text STRING,
  ingested_at TIMESTAMP
) USING DELTA
""")

# Simple deterministic chunker for demo purposes.
@F.udf(ArrayType(StringType()))
def chunk_text(text):
    if not text:
        return []
    words = text.split()
    size, overlap = 350, 50
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks

docs = spark.table(f"{CATALOG}.bronze.documents_raw")

chunks = (
    docs
    .withColumn("chunks", chunk_text("report_text"))
    .select(
        "document_id", "title", "source_url",
        F.posexplode("chunks").alias("chunk_id", "chunk_text")
    )
)

(chunks.write.format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.silver.document_chunks"))

display(chunks.limit(20))

# Next workspace step:
# Create a Databricks Vector Search index over silver.document_chunks.
# Configure the embedding endpoint/index name in src/rag_agent.py.
