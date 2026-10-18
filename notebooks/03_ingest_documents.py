# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3 - Ingest unstructured data → Bronze (Volume + metadata table)
# MAGIC Finds recent World Bank reports through the
# MAGIC [Documents & Reports API](https://documents.worldbank.org/en/publication/documents-reports/api),
# MAGIC downloads the PDFs into a **Unity Catalog Volume**, and records one row per document in
# MAGIC `bronze.documents_meta`. The PDFs are governed by the same grants, lineage and audit log as tables.
# MAGIC
# MAGIC | What | Where |
# MAGIC |---|---|
# MAGIC | ~2–3 newest reports per country (Economic/Development Updates, Economic Prospects, Poverty Assessments, Climate & Development Reports) | `/Volumes/wb_demo/bronze/raw_docs/pdfs/<doc_id>.pdf` |
# MAGIC | Regional & global flagships (South Asia Economic Focus, Africa's Pulse, EAP & MENA Updates, Global Economic Prospects) | same |
# MAGIC | Metadata: title, type, date, country, region, abstract, URLs, file path | `wb_demo.bronze.documents_meta` (MERGE - re-runs don't duplicate) |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os, re, json, datetime, requests
import concurrent.futures as cf
from requests.adapters import HTTPAdapter, Retry

DOCS_API = "https://search.worldbank.org/api/v2/wds"
PDF_DIR = f"{RAW_DOCS_PATH}/pdfs"
DOCS_PER_COUNTRY = 3
DOCS_PER_REGION = 2
MIN_DOC_DATE = "2015-01-01"
MAX_PDF_MB = 60

# Short queries work best - the API ANDs every word, so each query is tried separately and results are merged.
COUNTRY_QUERIES = ["economic update", "development update", "economic monitor", "economic prospects",
                   "country climate and development report", "country economic memorandum", "poverty assessment"]
COUNTRY_TITLE_RE = (r"(economic|development) (update|monitor|prospects|review)|taking stock|poverty (and equity )?assessment"
                    r"|country economic memorandum|country climate and development report")
EXCLUDE_TITLE_RE = r"\(vol\. [2-9]\)|growth data|daily|presentation|brief|annex|executive summary|background note|overview"
NAME_ALIASES = {"VNM": "Vietnam|Viet Nam", "TUR": "Turkey|T.rkiye", "EGY": "Egypt"}

# (region label, scope, query, title regex)
REGIONAL_REPORTS = [
    ("South Asia",                 "regional", "South Asia Economic Focus",                    r"south asia (economic focus|development update)"),
    ("Sub-Saharan Africa",         "regional", "Africa's Pulse",                               r"africa.s pulse"),
    ("East Asia & Pacific",        "regional", "East Asia and Pacific Economic Update",        r"east asia and (the )?pacific economic update"),
    ("Middle East & North Africa", "regional", "Middle East and North Africa Economic Update", r"(middle east and north africa|mena) economic update"),
    ("World",                      "global",   "Global Economic Prospects",                    r"^global economic prospects"),
]

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
session.mount("http://", session.get_adapter("https://"))

ingest_ts = datetime.datetime.utcnow()
log_event("ingest_documents", "started", per_country=DOCS_PER_COUNTRY, per_region=DOCS_PER_REGION)

# COMMAND ----------

# MAGIC %md ## 1. Metadata table

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.documents_meta (
    doc_id          STRING    COMMENT 'World Bank Documents & Reports ID (also the PDF file name)',
    title           STRING    COMMENT 'Document title',
    doc_type        STRING    COMMENT 'Document type, e.g. Report, Working Paper',
    major_doc_type  STRING    COMMENT 'Major document type, e.g. Economic & Sector Work',
    doc_date        DATE      COMMENT 'Publication date',
    scope           STRING    COMMENT 'country | regional | global',
    country_iso3    STRING    COMMENT 'ISO-3 code for country reports, NULL for regional/global',
    country_name    STRING    COMMENT 'Country name for country reports',
    region          STRING    COMMENT 'World Bank region (for country reports: the country''s region)',
    abstract        STRING    COMMENT 'Abstract from the API',
    pdf_url         STRING    COMMENT 'Source PDF URL',
    landing_url     STRING    COMMENT 'Human-readable document page (used for citations)',
    file_path       STRING    COMMENT 'Location of the PDF in the Unity Catalog volume',
    file_size_bytes BIGINT    COMMENT 'Downloaded PDF size',
    download_status STRING    COMMENT 'downloaded | already_present | failed: <reason>',
    search_query    STRING    COMMENT 'API query that found the document',
    ingest_ts       TIMESTAMP COMMENT 'UTC time of the ingestion run'
  )
  COMMENT 'Metadata for World Bank report PDFs stored in the bronze.raw_docs volume'
  TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# MAGIC %md ## 2. Discover documents
# MAGIC ISO-2 codes, names and regions come from `bronze.countries_raw` (Step 2) - the lakehouse feeding itself.

# COMMAND ----------

countries_df = spark.sql(f"""
  SELECT country_iso3,
         get_json_object(raw_json, '$.iso2Code')           AS iso2,
         get_json_object(raw_json, '$.name')               AS name,
         trim(get_json_object(raw_json, '$.region.value')) AS region
  FROM {CATALOG}.bronze.countries_raw
  WHERE country_iso3 IN ({",".join(f"'{c}'" for c in COUNTRIES)})
""")
countries = {r.country_iso3: r.asDict() for r in countries_df.collect()}
print(f"{len(countries)} countries loaded from bronze.countries_raw")

def search_docs(qterm: str, iso2: str | None = None, rows: int = 50) -> list[dict]:
    params = {"format": "json", "lang_exact": "English", "rows": rows, "qterm": qterm,
              "fl": "id,display_title,docdt,docty,majdocty,count,pdfurl,url,abstracts"}
    if iso2:
        params["countrycode_exact"] = iso2
    resp = session.get(DOCS_API, params=params, timeout=120)
    resp.raise_for_status()
    return [d for k, d in resp.json().get("documents", {}).items() if k != "facets"]

def clean_title(doc: dict) -> str:
    return " ".join(doc.get("display_title", "").split())

def pick(docs: list[tuple[str, dict]], title_re: str, n: int, name_re: str | None = None) -> list[tuple[str, dict]]:
    """Filter to real reports with a PDF, dedupe by title, return the n newest."""
    by_title = {}
    for query, d in docs:
        t = clean_title(d)
        if (d.get("pdfurl") and d.get("docdt", "") >= MIN_DOC_DATE
                and re.search(title_re, t, re.I) and not re.search(EXCLUDE_TITLE_RE, t, re.I)
                and (name_re is None or re.search(name_re, t, re.I))):
            by_title.setdefault(t.lower(), (query, d))
    return sorted(by_title.values(), key=lambda qd: qd[1].get("docdt", ""), reverse=True)[:n]

def discover_country(iso3: str) -> list[dict]:
    c = countries[iso3]
    name_re = NAME_ALIASES.get(iso3, re.escape(c["name"].split(",")[0]))
    found = []
    for q in COUNTRY_QUERIES:
        found += [(q, d) for d in search_docs(q, c["iso2"])]
    return [dict(doc=d, query=q, scope="country", country_iso3=iso3, country_name=c["name"], region=c["region"])
            for q, d in pick(found, COUNTRY_TITLE_RE, DOCS_PER_COUNTRY, name_re)]

def discover_region(region: str, scope: str, query: str, title_re: str) -> list[dict]:
    return [dict(doc=d, query=q, scope=scope, country_iso3=None, country_name=None, region=region)
            for q, d in pick([(query, d) for d in search_docs(query, rows=100)], title_re, DOCS_PER_REGION)]

candidates, discovery_errors = [], {}
with cf.ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(discover_country, iso3): iso3 for iso3 in countries}
    futures |= {pool.submit(discover_region, *r): r[0] for r in REGIONAL_REPORTS}
    for fut in cf.as_completed(futures):
        label = futures[fut]
        try:
            docs = fut.result()
            candidates += docs
            print(f"{label:<28} {len(docs)} docs")
        except Exception as e:
            discovery_errors[label] = str(e)
            print(f"{label:<28} ✗ {e}")

# The same report can surface for a country and a region - keep one copy
candidates = list({c["doc"]["id"]: c for c in candidates}.values())
print(f"\n{len(candidates)} unique documents selected")

# COMMAND ----------

# MAGIC %md ## 3. Download PDFs into the Volume
# MAGIC Idempotent: files already in the volume are skipped, so re-runs only fetch new reports.

# COMMAND ----------

os.makedirs(PDF_DIR, exist_ok=True)

def download(c: dict) -> dict:
    d = c["doc"]
    path = f"{PDF_DIR}/{d['id']}.pdf"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return {**c, "file_path": path, "size": os.path.getsize(path), "status": "already_present"}
    try:
        url = d["pdfurl"].replace("http://", "https://", 1)
        with session.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            body = r.content
        if not body.startswith(b"%PDF"):
            raise ValueError("response is not a PDF")
        if len(body) > MAX_PDF_MB * 1024 * 1024:
            raise ValueError(f"larger than {MAX_PDF_MB} MB")
        with open(path, "wb") as f:
            f.write(body)
        return {**c, "file_path": path, "size": len(body), "status": "downloaded"}
    except Exception as e:
        return {**c, "file_path": None, "size": None, "status": f"failed: {str(e)[:200]}"}

with cf.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(download, candidates))

status_counts = {}
for r in results:
    key = r["status"].split(":")[0]
    status_counts[key] = status_counts.get(key, 0) + 1
print(status_counts)
for r in results:
    if r["status"].startswith("failed"):
        print("✗", clean_title(r["doc"])[:70], "→", r["status"])

# COMMAND ----------

# MAGIC %md ## 4. Upsert metadata (Delta MERGE)

# COMMAND ----------

rows = []
for r in results:
    d = r["doc"]
    abstract = (d.get("abstracts") or {}).get("cdata!", "")
    rows.append((
        d["id"], clean_title(d), d.get("docty"), d.get("majdocty"),
        datetime.date.fromisoformat(d["docdt"][:10]), r["scope"], r["country_iso3"], r["country_name"], r["region"],
        " ".join(abstract.split()) or None, d["pdfurl"], d.get("url"), r["file_path"], r["size"], r["status"],
        r["query"], ingest_ts,
    ))

spark.createDataFrame(rows, spark.table(f"{CATALOG}.bronze.documents_meta").schema).createOrReplaceTempView("new_docs")
merge_result = spark.sql(f"""
  MERGE INTO {CATALOG}.bronze.documents_meta t
  USING new_docs s ON t.doc_id = s.doc_id
  WHEN MATCHED THEN UPDATE SET *
  WHEN NOT MATCHED THEN INSERT *
""")
display(merge_result)

# COMMAND ----------

# MAGIC %md ## 5. Checks

# COMMAND ----------

display(spark.sql(f"""
  SELECT scope, coalesce(country_name, region) AS subject, count(*) AS docs,
         max(doc_date) AS newest, round(sum(file_size_bytes) / 1e6, 1) AS mb
  FROM {CATALOG}.bronze.documents_meta
  WHERE download_status NOT LIKE 'failed%'
  GROUP BY ALL ORDER BY scope, subject
"""))

# COMMAND ----------

# MAGIC %md The PDFs are ordinary files in a governed Volume - list them with SQL, browse them in Catalog Explorer.

# COMMAND ----------

display(spark.sql(f"LIST '{PDF_DIR}'").orderBy("size", ascending=False).limit(10))

# COMMAND ----------

log_event("ingest_documents", "complete", selected=len(candidates), status=status_counts,
          discovery_errors=discovery_errors)
