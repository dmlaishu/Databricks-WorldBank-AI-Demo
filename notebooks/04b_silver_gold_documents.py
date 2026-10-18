# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4b - Unstructured path: PDFs → Silver text → Gold chunks
# MAGIC
# MAGIC | Layer | Table | Grain | Purpose |
# MAGIC |---|---|---|---|
# MAGIC | silver | `doc_text` | document × page | Text extracted from each PDF page (pypdf, run in parallel with Spark) |
# MAGIC | gold | `doc_chunks` | chunk (~500 words ≈ 650 tokens, 75-word overlap) | **Source table for the Vector Search index** - CDF on, metadata on every chunk |
# MAGIC
# MAGIC **Incremental:** only PDFs not yet in `silver.doc_text` are parsed, and `doc_chunks` is updated
# MAGIC with `MERGE`, so the Delta Sync index (Step 6) only re-embeds what changed.
# MAGIC
# MAGIC *Alternative parser:* `ai_parse_document()` gives layout-aware parsing (tables, figures) but is
# MAGIC billed per page. pypdf is free and good enough for these text-heavy reports.

# COMMAND ----------

# MAGIC %pip install -q "pypdf>=5,<6"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F

spark.sql(f"USE CATALOG {CATALOG}")
PDF_DIR = f"{RAW_DOCS_PATH}/pdfs"
CHUNK_WORDS, OVERLAP_WORDS, MIN_CHUNK_WORDS = 500, 75, 40

log_event("silver_gold_documents", "started", chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS)

# COMMAND ----------

# MAGIC %md ## Silver: extract text per page

# COMMAND ----------

spark.sql("""
  CREATE TABLE IF NOT EXISTS silver.doc_text (
    doc_id     STRING    COMMENT 'Document ID (joins to bronze.documents_meta)',
    page_num   INT       COMMENT '1-based page number in the PDF',
    page_text  STRING    COMMENT 'Extracted text with whitespace normalized and line-break hyphenation removed',
    n_words    INT       COMMENT 'Word count of page_text',
    parser     STRING    COMMENT 'Library used to extract the text',
    parsed_ts  TIMESTAMP COMMENT 'When the page was parsed'
  )
  COMMENT 'Text extracted from World Bank report PDFs, one row per page'
""")

@F.udf("array<struct<page_num:int, page_text:string>>")
def extract_pages(content: bytearray):
    import io, re
    from pypdf import PdfReader
    pages = []
    try:
        reader = PdfReader(io.BytesIO(bytes(content)))
    except Exception:
        return pages
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)   # re-join words hyphenated across lines
        text = re.sub(r"\s+", " ", text).strip()
        pages.append((i, text))
    return pages

# Only PDFs that downloaded successfully and have not been parsed yet
to_parse = spark.sql("""
  SELECT m.doc_id
  FROM bronze.documents_meta m
  LEFT ANTI JOIN (SELECT DISTINCT doc_id FROM silver.doc_text) t ON m.doc_id = t.doc_id
  WHERE m.download_status NOT LIKE 'failed%'
""")
n_to_parse = to_parse.count()
print(f"{n_to_parse} new document(s) to parse")

# COMMAND ----------

if n_to_parse:
    pdfs = (
        spark.read.format("binaryFile").load(PDF_DIR)
        .withColumn("doc_id", F.regexp_extract("path", r"([^/]+)\.pdf$", 1))
        .join(to_parse, "doc_id")
        .repartition(max(4, min(n_to_parse, 32)), "doc_id")   # one PDF per task → parallel parsing
    )
    pages = (
        pdfs.select("doc_id", F.explode(extract_pages("content")).alias("p"))
        .select("doc_id", "p.page_num", "p.page_text")
        .withColumn("n_words", F.size(F.split("page_text", " ")))
        .where(F.length("page_text") > 0)
        .withColumn("parser", F.lit("pypdf"))
        .withColumn("parsed_ts", F.current_timestamp())
    )
    pages.write.mode("append").saveAsTable("silver.doc_text")

display(spark.sql("""
  SELECT count(DISTINCT doc_id) AS documents, count(*) AS pages, sum(n_words) AS words,
         round(avg(n_words)) AS avg_words_per_page
  FROM silver.doc_text
"""))

# COMMAND ----------

# Documents where almost nothing was extracted are probably scanned images (would need OCR / ai_parse_document)
display(spark.sql("""
  SELECT m.doc_id, m.title, count(t.page_num) AS pages, coalesce(sum(t.n_words), 0) AS words
  FROM bronze.documents_meta m LEFT JOIN silver.doc_text t USING (doc_id)
  WHERE m.download_status NOT LIKE 'failed%'
  GROUP BY ALL HAVING words < 1000
"""))

# COMMAND ----------

# MAGIC %md ## Gold: chunk with overlap, keep metadata on every chunk
# MAGIC Chunks span page boundaries (so no tiny end-of-page fragments) but remember `page_start`/`page_end`
# MAGIC for citations. `search_text` prefixes the chunk with title/country/date - that is what gets
# MAGIC embedded, so a query like *"Kenya risks"* matches chunks that never repeat the word Kenya.

# COMMAND ----------

@F.udf("array<struct<chunk_index:int, page_start:int, page_end:int, chunk_text:string>>")
def chunk_document(pages):
    words = []   # (word, page_num)
    for p in sorted(pages, key=lambda p: p["page_num"]):
        words.extend((w, p["page_num"]) for w in p["page_text"].split())
    chunks, start, step = [], 0, CHUNK_WORDS - OVERLAP_WORDS
    while start < len(words):
        window = words[start:start + CHUNK_WORDS]
        if len(window) >= MIN_CHUNK_WORDS or not chunks:
            chunks.append((len(chunks), window[0][1], window[-1][1], " ".join(w for w, _ in window)))
        if start + CHUNK_WORDS >= len(words):
            break
        start += step
    return chunks

chunks = (
    spark.table("silver.doc_text")
    .groupBy("doc_id").agg(F.collect_list(F.struct("page_num", "page_text")).alias("pages"))
    .select("doc_id", F.explode(chunk_document("pages")).alias("c"))
    .select("doc_id", "c.*")
    .join(spark.table("bronze.documents_meta"), "doc_id")
    .select(
        F.concat_ws("_", "doc_id", F.format_string("%04d", "chunk_index")).alias("chunk_id"),
        "doc_id", "chunk_index", "page_start", "page_end", "chunk_text",
        F.size(F.split("chunk_text", " ")).alias("n_words"),
        F.concat(
            F.lit("Title: "), "title",
            F.lit(" | Scope: "), F.coalesce("country_name", "region"),
            F.lit(" | Date: "), F.date_format("doc_date", "yyyy-MM"),
            F.lit("\n"), "chunk_text",
        ).alias("search_text"),
        "title", "doc_type", "doc_date", F.year("doc_date").alias("doc_year"),
        "scope", "country_iso3", "country_name", "region", "landing_url", "pdf_url",
    )
)
chunks.createOrReplaceTempView("new_chunks")

# COMMAND ----------

spark.sql("""
  CREATE TABLE IF NOT EXISTS gold.doc_chunks (
    chunk_id     STRING NOT NULL COMMENT 'Primary key: <doc_id>_<chunk_index>',
    doc_id       STRING    COMMENT 'Source document ID',
    chunk_index  INT       COMMENT 'Position of the chunk within the document',
    page_start   INT       COMMENT 'PDF page where the chunk starts (for citations)',
    page_end     INT       COMMENT 'PDF page where the chunk ends',
    chunk_text   STRING    COMMENT 'Chunk text shown to users and passed to the LLM',
    n_words      INT       COMMENT 'Word count of chunk_text',
    search_text  STRING    COMMENT 'Title/scope/date header + chunk_text. This column is embedded by Vector Search.',
    title        STRING    COMMENT 'Document title',
    doc_type     STRING    COMMENT 'Document type',
    doc_date     DATE      COMMENT 'Publication date',
    doc_year     INT       COMMENT 'Publication year (handy vector search filter)',
    scope        STRING    COMMENT 'country | regional | global',
    country_iso3 STRING    COMMENT 'ISO-3 code for country reports, NULL for regional/global reports',
    country_name STRING    COMMENT 'Country name for country reports',
    region       STRING    COMMENT 'World Bank region the report covers',
    landing_url  STRING    COMMENT 'Document web page, used as the citation link',
    pdf_url      STRING    COMMENT 'Direct PDF link',
    CONSTRAINT doc_chunks_pk PRIMARY KEY (chunk_id)
  )
  COMMENT 'Overlapping text chunks of World Bank reports with document metadata. Source for the Vector Search index used by the Research agent.'
  TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

merge = spark.sql("""
  MERGE INTO gold.doc_chunks t
  USING new_chunks s ON t.chunk_id = s.chunk_id
  WHEN MATCHED AND t.search_text <> s.search_text THEN UPDATE SET *
  WHEN NOT MATCHED THEN INSERT *
  WHEN NOT MATCHED BY SOURCE THEN DELETE
""")
display(merge)

# COMMAND ----------

for stmt in [
    "ALTER TABLE gold.doc_chunks SET TAGS ('layer' = 'gold', 'domain' = 'world_bank_reports', 'usage' = 'vector_search_source')",
    "ALTER TABLE silver.doc_text SET TAGS ('layer' = 'silver')",
]:
    try:
        spark.sql(stmt)
    except Exception as e:
        print("⚠", stmt, "→", str(e).splitlines()[0][:150])

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

stats = spark.sql("""
  SELECT count(*) AS chunks, count(DISTINCT doc_id) AS documents,
         round(avg(n_words)) AS avg_words, min(n_words) AS min_words, max(n_words) AS max_words
  FROM gold.doc_chunks
""").first()
assert stats.chunks > 0, "gold.doc_chunks is empty"
display(spark.createDataFrame([stats]))

display(spark.sql("""
  SELECT scope, coalesce(country_name, region) AS subject, count(DISTINCT doc_id) AS docs, count(*) AS chunks
  FROM gold.doc_chunks GROUP BY ALL ORDER BY scope, subject
"""))

# COMMAND ----------

# Peek at a chunk relevant to the demo question (keyword search - semantic search comes in Step 6)
display(spark.sql("""
  SELECT title, page_start, page_end, left(chunk_text, 600) AS preview
  FROM gold.doc_chunks
  WHERE region = 'South Asia' AND scope = 'regional' AND lower(chunk_text) LIKE '%risk%'
  LIMIT 3
"""))

# COMMAND ----------

display(spark.sql("DESCRIBE HISTORY gold.doc_chunks").select("version", "timestamp", "operation", "operationMetrics").limit(5))

# COMMAND ----------

log_event("silver_gold_documents", "complete", parsed_new_docs=n_to_parse,
          chunks=stats.chunks, documents=stats.documents, avg_words=stats.avg_words)
