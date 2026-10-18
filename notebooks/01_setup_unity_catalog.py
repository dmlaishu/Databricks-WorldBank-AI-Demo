# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1 - Workspace & Unity Catalog governance
# MAGIC Creates the catalog, medallion schemas, the raw-docs Volume, the ops log table,
# MAGIC and the two demo groups with **different** permissions:
# MAGIC
# MAGIC | Group | bronze | silver | gold | ai | ops |
# MAGIC |---|---|---|---|---|---|
# MAGIC | `wb_analysts` | ❌ denied | ❌ | ✅ read | ✅ execute functions | ❌ |
# MAGIC | `wb_ai_app` (agent / app service principal) | ❌ | ❌ | ✅ read | ✅ read + execute | ✅ write logs |
# MAGIC
# MAGIC **Run as:** a metastore admin (or someone with `CREATE CATALOG`).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md ## 1. Catalog and schemas

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'World Bank AI assistant demo: data, AI assets and ops logs'")
spark.sql(f"USE CATALOG {CATALOG}")

for schema, comment in SCHEMAS.items():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema} COMMENT '{comment}'")

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# COMMAND ----------

# MAGIC %md ## 2. Volume for raw PDFs (unstructured data governed like a table)

# COMMAND ----------

spark.sql(f"""
  CREATE VOLUME IF NOT EXISTS {RAW_DOCS_VOLUME}
  COMMENT 'Raw World Bank report PDFs downloaded from the Documents & Reports API'
""")
dbutils.fs.mkdirs(RAW_DOCS_PATH + "/pdfs")
display(dbutils.fs.ls(RAW_DOCS_PATH))

# COMMAND ----------

# MAGIC %md ## 3. Ops table for application logs

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {CATALOG}.ops.app_logs (
    event_ts  TIMESTAMP COMMENT 'UTC time the event was logged',
    run_id    STRING    COMMENT 'Notebook/app run identifier',
    component STRING    COMMENT 'Pipeline stage or service that emitted the event',
    level     STRING    COMMENT 'INFO / WARNING / ERROR',
    event     STRING    COMMENT 'Short event name, e.g. ingest_complete',
    details   STRING    COMMENT 'JSON payload with event-specific details'
  )
  COMMENT 'Application logs from notebooks, jobs, the agent and the chat app'
  TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
log_event("setup", "unity_catalog_setup_started", catalog=CATALOG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Groups
# MAGIC Unity Catalog grants need **account-level** groups. This cell tries to create them with the
# MAGIC Databricks SDK (works if you are an account admin with identity federation). If it fails,
# MAGIC create them manually: **Account console → User management → Groups → Add group**, then
# MAGIC add them to this workspace. Add yourself to neither - you need a separate test user
# MAGIC (or a service principal) in `wb_analysts` for the access-denied demo in Step 15.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
# Not created via API on purpose: the workspace API makes *workspace-local* groups, which Unity Catalog can't use.
groups = {g.display_name: g for g in w.groups.list(attributes="displayName,meta")}
for group in (GROUP_ANALYSTS, GROUP_AI_APP):
    if group in groups:
        print(f"✓ {group} found - check its Source is 'Account' (Settings → Identity and access → Groups)")
    else:
        print(f"✗ {group} missing → Settings → Identity and access → Groups → Add group (Source must be 'Account')")

# COMMAND ----------

# MAGIC %md ## 5. Grants - least privilege per group

# COMMAND ----------

grants = [
    # Both groups can see the catalog, but only the schemas granted below
    f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{GROUP_ANALYSTS}`",
    f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{GROUP_AI_APP}`",

    # Analysts: read gold, call governed AI tool functions. No bronze/silver/ops.
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {CATALOG}.gold TO `{GROUP_ANALYSTS}`",
    f"GRANT USE SCHEMA, EXECUTE ON SCHEMA {CATALOG}.ai TO `{GROUP_ANALYSTS}`",

    # AI app / agent: read gold + ai assets, write logs to ops
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {CATALOG}.gold TO `{GROUP_AI_APP}`",
    f"GRANT USE SCHEMA, SELECT, EXECUTE ON SCHEMA {CATALOG}.ai TO `{GROUP_AI_APP}`",
    f"GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA {CATALOG}.ops TO `{GROUP_AI_APP}`",
]

for stmt in grants:
    try:
        spark.sql(stmt)
        print("✓", stmt)
    except Exception as e:
        print("✗", stmt, "\n   ", str(e).splitlines()[0])

# COMMAND ----------

# MAGIC %md ## 6. Verify

# COMMAND ----------

for schema in ("bronze", "gold", "ai", "ops"):
    print(f"--- {CATALOG}.{schema}")
    display(spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOG}.{schema}"))

# COMMAND ----------

log_event("setup", "unity_catalog_setup_complete", catalog=CATALOG, schemas=list(SCHEMAS), groups=[GROUP_ANALYSTS, GROUP_AI_APP])
display(spark.table(f"{CATALOG}.ops.app_logs").orderBy("event_ts", ascending=False).limit(5))
