# Databricks notebook source
# MAGIC %md
# MAGIC # 00_verify_workspace - pre-flight checks
# MAGIC Run once on a new workspace (serverless compute). Every row in the final table should be ✅
# MAGIC before running `01_setup_unity_catalog`. Read-only, except a test catalog that is dropped immediately.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
results = []

def check(name, fn):
    try:
        detail = fn()
        results.append((name, "✅", str(detail)[:150]))
    except Exception as e:
        results.append((name, "❌", str(e).splitlines()[0][:150]))

# COMMAND ----------

check("Current user", lambda: spark.sql("SELECT current_user()").first()[0])
check("Unity Catalog metastore", lambda: spark.sql("SELECT current_metastore()").first()[0])
check("Existing catalogs", lambda: [r[0] for r in spark.sql("SHOW CATALOGS").collect()])

def can_create_catalog():
    spark.sql("CREATE CATALOG IF NOT EXISTS wb_preflight_test")
    spark.sql("DROP CATALOG wb_preflight_test CASCADE")
    return "CREATE CATALOG works → keep CATALOG = 'wb_demo'"
check("Can create catalog", can_create_catalog)

check("System table: access.audit", lambda: spark.sql("SELECT count(*) FROM system.access.audit WHERE event_date >= current_date() - 7").first()[0])
check("System table: billing.usage", lambda: spark.sql("SELECT count(*) FROM system.billing.usage").first()[0])
check("System table: lakeflow.job_run_timeline", lambda: spark.sql("SELECT count(*) FROM system.lakeflow.job_run_timeline").first()[0])

# COMMAND ----------

endpoints = []
check("Serving endpoints", lambda: endpoints.extend(e.name for e in w.serving_endpoints.list()) or f"{len(endpoints)} found")
check("Embedding model (databricks-gte-large-en)",
      lambda: len(spark.sql("SELECT ai_query('databricks-gte-large-en', 'hello') AS v").first()[0]))
chat_models = [e for e in endpoints if any(k in e for k in ("claude", "llama", "gpt", "gemma", "qwen"))]
results.append(("Chat LLM endpoints", "✅" if chat_models else "❌", ", ".join(chat_models) or "none - check Serving page"))

check("SQL warehouses", lambda: [(wh.name, str(wh.state).split(".")[-1]) for wh in w.warehouses.list()])
check("Vector Search API", lambda: f"{len(list(w.vector_search_endpoints.list_endpoints()))} endpoints (0 is fine for now)")
check("Groups visible", lambda: [g.display_name for g in w.groups.list(attributes="displayName")][:10])

# COMMAND ----------

display(spark.createDataFrame(results, "check string, status string, detail string"))
