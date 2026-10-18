# Databricks notebook source
# MAGIC %md
# MAGIC # Step 6 - Vector Search index (RAG retrieval)
# MAGIC Creates a Vector Search **endpoint** and a **Delta Sync index** on `gold.doc_chunks`.
# MAGIC Databricks computes embeddings with `databricks-gte-large-en` from the `search_text` column,
# MAGIC and because the source table has Change Data Feed on, a sync only re-embeds changed chunks.
# MAGIC
# MAGIC The index is a Unity Catalog object (`wb_demo.ai.doc_chunks_index`): it has grants, lineage and audit
# MAGIC like any table.
# MAGIC
# MAGIC Uses the **Databricks SDK that ships with serverless** - no `%pip install`. (The separate
# MAGIC `databricks-vectorsearch` package downgrades protobuf and crashes the serverless Python kernel.)
# MAGIC
# MAGIC First run: endpoint ~5–15 min + initial embedding ~5–15 min. Re-runs just trigger a sync.
# MAGIC The endpoint is billed per hour while it exists - see the cleanup cell at the bottom.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json, time
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn, EndpointType, PipelineType, VectorIndexType,
)

w = WorkspaceClient()
SOURCE_TABLE = f"{CATALOG}.gold.doc_chunks"
COLUMNS_TO_SYNC = ["chunk_id", "doc_id", "chunk_text", "title", "doc_type", "doc_year", "page_start", "page_end",
                   "scope", "country_iso3", "country_name", "region", "landing_url"]

log_event("vector_index", "started", endpoint=VS_ENDPOINT, index=VS_INDEX)

def wait_for(check, what: str, timeout_min: int = 20, every_s: int = 60) -> bool:
    """Poll check() -> (done, state). Returns False (no error) on timeout: the work continues server-side,
    so just re-run the cell later."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        done, state = check()
        print(f"{time.strftime('%H:%M:%S')}  {what}: {state}")
        if done:
            return True
        time.sleep(every_s)
    print(f"⏳ {what} still building after {timeout_min} min - it continues in the background. Re-run this cell later.")
    return False

# COMMAND ----------

# MAGIC %md ## 1. Endpoint

# COMMAND ----------

if VS_ENDPOINT not in {e.name for e in w.vector_search_endpoints.list_endpoints()}:
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD)
    print("Creating endpoint", VS_ENDPOINT)

def endpoint_state():
    status = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT).endpoint_status
    state = status.state.value if status and status.state else "UNKNOWN"
    if state in ("OFFLINE", "RED_STATE"):
        raise RuntimeError(f"Endpoint {VS_ENDPOINT} is {state}: {status.message}")
    return state == "ONLINE", state

assert wait_for(endpoint_state, f"endpoint {VS_ENDPOINT}", timeout_min=45), "Endpoint not online yet - re-run this cell later"

# COMMAND ----------

# MAGIC %md ## 2. Delta Sync index (managed embeddings)

# COMMAND ----------

def index_state():
    status = w.vector_search_indexes.get_index(VS_INDEX).status
    msg = (status.message or "")[:120]
    if "fail" in msg.lower():
        raise RuntimeError(f"Index failed: {status.message}")
    return bool(status.ready), f"ready={status.ready} · rows indexed={status.indexed_row_count} · {msg}"

existing_indexes = {i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=VS_ENDPOINT)}

if VS_INDEX not in existing_indexes:
    w.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_TABLE,
            pipeline_type=PipelineType.TRIGGERED,          # sync on demand (the weekly job triggers it)
            embedding_source_columns=[EmbeddingSourceColumn(name="search_text",
                                                            embedding_model_endpoint_name=EMBEDDING_ENDPOINT)],
            columns_to_sync=COLUMNS_TO_SYNC,
        ),
    )
    print("Creating index", VS_INDEX, "- first embedding of all chunks can take 30–90 min on a trial")
elif index_state()[0]:
    w.vector_search_indexes.sync_index(index_name=VS_INDEX)   # incremental via Change Data Feed
    print("Index exists and is ready → incremental sync triggered")
else:
    print("Index exists and is still syncing:", index_state()[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for the index
# MAGIC Checks once a minute for up to 20 min, then stops **without failing**. The embedding keeps running
# MAGIC server-side - re-run this cell (only this one) until it prints `ready=True`.
# MAGIC Progress is also visible in **Catalog → wb_demo → ai → doc_chunks_index**.

# COMMAND ----------

time.sleep(15)
index_ready = wait_for(index_state, "index")

# COMMAND ----------

status = w.vector_search_indexes.get_index(VS_INDEX).status
indexed = status.indexed_row_count
source_rows = spark.table(SOURCE_TABLE).count()
print(f"Indexed rows: {indexed} / source rows: {source_rows}")
if not status.ready:
    dbutils.notebook.exit(f"Index not ready yet ({indexed}/{source_rows}). Re-run the 'Wait for the index' cell later, then continue from here.")

# COMMAND ----------

# MAGIC %md ## 3. Test retrieval
# MAGIC Three styles the Research agent will use: plain semantic search, a metadata filter, and hybrid (semantic + keyword).

# COMMAND ----------

RESULT_COLUMNS = ["title", "country_name", "region", "doc_year", "page_start", "chunk_text", "landing_url"]

def search(query: str, filters: dict | None = None, k: int = 5, query_type: str = "ANN") -> pd.DataFrame:
    kwargs = dict(index_name=VS_INDEX, columns=RESULT_COLUMNS, query_text=query, num_results=k,
                  filters_json=json.dumps(filters) if filters else None)
    try:
        res = w.vector_search_indexes.query_index(**kwargs, query_type=query_type)
    except TypeError:   # older SDK without query_type → falls back to semantic search
        res = w.vector_search_indexes.query_index(**kwargs)
    cols = [c.name for c in res.manifest.columns]
    df = pd.DataFrame(res.result.data_array or [], columns=cols)
    if not df.empty:
        df["chunk_text"] = df["chunk_text"].str.slice(0, 250) + "…"
    return df

print("① Semantic: 'South Asia growth risks'")
display(search("South Asia growth risks"))

# COMMAND ----------

print("② Filtered to regional South Asia reports")
display(search("main risks to growth outlook", filters={"region": "South Asia", "scope": "regional"}))

# COMMAND ----------

print("③ Hybrid (semantic + keyword), Kenya reports from 2023 onwards")
display(search("inflation and public debt pressures", filters={"country_iso3": "KEN", "doc_year >=": 2023}, query_type="HYBRID"))

# COMMAND ----------

# MAGIC %md ## 4. Governance: the index is a UC securable

# COMMAND ----------

for stmt in [f"GRANT SELECT ON TABLE {VS_INDEX} TO `{GROUP_AI_APP}`",
             f"ALTER TABLE {VS_INDEX} SET TAGS ('layer' = 'ai', 'usage' = 'rag_retrieval')"]:
    try:
        spark.sql(stmt)
        print("✓", stmt)
    except Exception as e:
        print("⚠", stmt, "→", str(e).splitlines()[0][:150])

log_event("vector_index", "complete", indexed_rows=indexed, source_rows=source_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (run only after the demo)
# MAGIC Deleting the endpoint stops hourly billing. Re-running this notebook rebuilds endpoint and index.

# COMMAND ----------

# w.vector_search_indexes.delete_index(index_name=VS_INDEX)
# w.vector_search_endpoints.delete_endpoint(endpoint_name=VS_ENDPOINT)
