# Databricks notebook source
# MAGIC %md
# MAGIC # Step 15 - Audit logs, lineage and the live governance demo
# MAGIC
# MAGIC 1. **Groups & grants** - make sure `wb_analysts` / `wb_ai_app` exist and re-apply the least-privilege grants:
# MAGIC
# MAGIC    | | bronze | silver | gold | ai functions | vector index | ops |
# MAGIC    |---|---|---|---|---|---|---|
# MAGIC    | `wb_analysts` | ❌ | ❌ | ✅ SELECT | ✅ EXECUTE | ❌ → `search_reports` is denied | ❌ |
# MAGIC    | `wb_ai_app` | ❌ | ❌ | ✅ SELECT | ✅ EXECUTE | ✅ SELECT | ✅ SELECT, MODIFY |
# MAGIC
# MAGIC 2. **Audit views** over `system.access.audit`: who touched which `wb_demo` object, who queried the model
# MAGIC    endpoint / app, who changed permissions, and every **denied** request.
# MAGIC 3. **Lineage** over `system.access.table_lineage`: bronze → silver → gold → vector index (→ agent → endpoint in
# MAGIC    Catalog Explorer).
# MAGIC 4. **The live demo:** a test user in `wb_analysts` reads gold ✅, tries bronze ❌ - and the denial shows up here.
# MAGIC
# MAGIC Audit events arrive with a delay (typically minutes, can be up to ~1 hour).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
C = CATALOG
ENDPOINT_NAME, APP_NAME = "wb-demo-agent", "wb-demo-chat"
TEST_USER_EMAIL = ""     # optional: the second (analyst) user's email, to filter the audit views to them
log_event("audit", "started")

# COMMAND ----------

# MAGIC %md ## 1. Groups

# COMMAND ----------

def principal_exists_in_uc(name: str) -> bool:
    """UC only accepts account-level principals - test with a harmless grant."""
    try:
        spark.sql(f"GRANT USE CATALOG ON CATALOG {C} TO `{name}`")
        return True
    except Exception as e:
        if "PRINCIPAL_DOES_NOT_EXIST" in str(e):
            return False
        raise

# Groups are NOT created here: the workspace API creates *workspace-local* groups, which Unity Catalog ignores.
# They must be account groups (Settings → Identity and access → Groups → Add group → Source shows "Account").
missing = []
for g in (GROUP_ANALYSTS, GROUP_AI_APP):
    ok = principal_exists_in_uc(g)
    print(f"{'✓' if ok else '✗'} {g} {'is an account group usable in Unity Catalog' if ok else 'not usable in Unity Catalog'}")
    if not ok:
        missing.append(g)

if missing:
    displayHTML(f"""<h3>⚠ Create these as ACCOUNT groups, then re-run this notebook: {', '.join(missing)}</h3>
    <ol><li>Profile icon (top right) → <b>Settings</b> → <b>Identity and access</b> → <b>Groups</b> → <b>Manage</b>.
    If the group is listed with Source <b>Workspace</b>, delete it first.</li>
    <li><b>Add group</b> → Source must show <b>Account</b>
    (otherwise: <b>Manage account</b> → User management → Groups → Add group, then assign it to this workspace)</li>
    <li>Name it exactly <code>{GROUP_ANALYSTS}</code> / <code>{GROUP_AI_APP}</code></li>
    <li>Open <code>{GROUP_ANALYSTS}</code> → <b>Add members</b> → your second test user (not yourself)</li></ol>""")

# COMMAND ----------

# MAGIC %md ## 2. Least-privilege grants (idempotent)

# COMMAND ----------

grants = [
    # analysts: gold + governed functions only
    f"GRANT USE CATALOG ON CATALOG {C} TO `{GROUP_ANALYSTS}`",
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {C}.gold TO `{GROUP_ANALYSTS}`",
    f"GRANT USE SCHEMA, EXECUTE ON SCHEMA {C}.ai TO `{GROUP_ANALYSTS}`",
    # app / agent identity: gold, all AI assets, write logs
    f"GRANT USE CATALOG ON CATALOG {C} TO `{GROUP_AI_APP}`",
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {C}.gold TO `{GROUP_AI_APP}`",
    f"GRANT USE SCHEMA, SELECT, EXECUTE ON SCHEMA {C}.ai TO `{GROUP_AI_APP}`",
    f"GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA {C}.ops TO `{GROUP_AI_APP}`",
]
for stmt in grants:
    try:
        spark.sql(stmt)
        print("✓", stmt)
    except Exception as e:
        print("⚠", stmt, "→", str(e).splitlines()[0][:120])

for schema in ("bronze", "gold", "ai"):
    print(f"\n── grants on {C}.{schema}")
    try:
        for r in spark.sql(f"SHOW GRANTS ON SCHEMA {C}.{schema}").collect():
            print(f"   {r[0]:<40} {r[1]}")
    except Exception as e:
        print("  ", str(e).splitlines()[0][:120])

# COMMAND ----------

# MAGIC %md ## 3. Audit views over `system.access.audit`

# COMMAND ----------

TOUCHES_DEMO = ("(array_join(map_values(request_params), ' ') ILIKE '%wb_demo%' "
                f"OR array_join(map_values(request_params), ' ') ILIKE '%{ENDPOINT_NAME}%' "
                f"OR array_join(map_values(request_params), ' ') ILIKE '%{APP_NAME}%')")
OBJECT = ("coalesce(request_params['full_name_arg'], request_params['table_full_name'], request_params['securable_full_name'], "
          "request_params['name'], request_params['name_arg'], request_params['endpoint_name'], request_params['app_name'])")

audit_views = {
    "v_audit_events": f"""
        SELECT event_time, event_date, user_identity.email AS user, service_name, action_name,
               {OBJECT} AS object, response.status_code AS status_code, response.error_message AS error,
               source_ip_address, request_params
        FROM system.access.audit
        WHERE event_date >= current_date() - 30 AND {TOUCHES_DEMO}""",
    "v_audit_denied": f"""
        SELECT event_time, user, service_name, action_name, object, status_code, error
        FROM {C}.ops.v_audit_events
        WHERE status_code IN (401, 403) OR error ILIKE '%PERMISSION_DENIED%' OR error ILIKE '%does not have%'
              OR error ILIKE '%insufficient%'""",
    "v_audit_permission_changes": f"""
        SELECT event_time, user, action_name, object, request_params['changes'] AS changes, status_code
        FROM {C}.ops.v_audit_events
        WHERE action_name ILIKE '%permission%' OR action_name ILIKE '%grant%'""",
    "v_lineage": f"""
        SELECT source_table_full_name AS source, target_table_full_name AS target,
               source_type, target_type, entity_type, max(event_time) AS last_seen, count(*) AS events
        FROM system.access.table_lineage
        WHERE event_date >= current_date() - 30
          AND (source_table_full_name LIKE '{C}.%' OR target_table_full_name LIKE '{C}.%')
        GROUP BY ALL""",
}
for name, body in audit_views.items():
    try:
        spark.sql(f"CREATE OR REPLACE VIEW {C}.ops.{name} AS {body}")
        print(f"✓ ops.{name:<28} {spark.table(f'{C}.ops.{name}').count()} rows")
    except Exception as e:
        print(f"⚠ ops.{name:<28} → {str(e).splitlines()[0][:160]}")

# COMMAND ----------

# MAGIC %md ### Who accessed which `wb_demo` table?

# COMMAND ----------

display(spark.sql(f"""
  SELECT user, object, collect_set(action_name) AS actions, count(*) AS events, max(event_time) AS last_access
  FROM {C}.ops.v_audit_events
  WHERE service_name = 'unityCatalog' AND object LIKE '{C}.%'
  GROUP BY user, object ORDER BY last_access DESC LIMIT 50"""))

# COMMAND ----------

# MAGIC %md ### Who queried the model endpoint and the chat app?

# COMMAND ----------

display(spark.sql(f"""
  SELECT user, service_name, action_name, count(*) AS events, max(event_time) AS last_event
  FROM {C}.ops.v_audit_events
  WHERE service_name ILIKE '%serving%' OR service_name ILIKE '%apps%'
  GROUP BY ALL ORDER BY last_event DESC LIMIT 50"""))

# COMMAND ----------

# MAGIC %md ### Who changed permissions?

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {C}.ops.v_audit_permission_changes ORDER BY event_time DESC LIMIT 50"))

# COMMAND ----------

# MAGIC %md ## 4. Lineage: from raw data to the AI layer

# COMMAND ----------

display(spark.sql(f"""
  SELECT source, '→' AS flow, target, entity_type, last_seen
  FROM {C}.ops.v_lineage
  WHERE source IS NOT NULL AND target IS NOT NULL
  ORDER BY CASE WHEN source LIKE '%bronze%' THEN 1 WHEN source LIKE '%silver%' THEN 2 ELSE 3 END, source"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Show it visually:** Catalog → `wb_demo.gold.doc_chunks` → **Lineage** → *See lineage graph* (upstream: silver.doc_text
# MAGIC ← bronze.documents_meta; downstream: `ai.doc_chunks_index`). Then Catalog → `wb_demo.ai.worldbank_agent` → versions
# MAGIC → the model's lineage and the serving endpoint it's deployed to.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The live governance demo (access denied → audit log)
# MAGIC
# MAGIC **Setup (once):** your second user is a member of `wb_analysts` (section 1). Open an **incognito window**, sign in to
# MAGIC this workspace as that user, open the **SQL Editor** with the serverless warehouse. *(If they can't see the warehouse:
# MAGIC SQL Warehouses → the warehouse → Permissions → add `wb_analysts` with **Can use**.)*
# MAGIC
# MAGIC **As the analyst, run:**
# MAGIC ```sql
# MAGIC SELECT * FROM wb_demo.gold.country_year_facts LIMIT 5;                          -- ✅ allowed
# MAGIC SELECT * FROM wb_demo.ai.compare_countries('India, Bangladesh', 'gdp_growth_pct', 2015); -- ✅ governed tool
# MAGIC SELECT * FROM wb_demo.bronze.indicators_raw LIMIT 5;                             -- ❌ PERMISSION_DENIED
# MAGIC SELECT * FROM wb_demo.ai.search_reports('growth risks', 'Kenya');               -- ❌ no access to the vector index
# MAGIC ```
# MAGIC **Then, back here as admin** (wait a few minutes for audit delivery), run the next cell.

# COMMAND ----------

user_filter = f"AND user = '{TEST_USER_EMAIL}'" if TEST_USER_EMAIL else ""
display(spark.sql(f"""
  SELECT event_time, user, service_name, action_name, object, status_code, left(error, 200) AS error
  FROM {C}.ops.v_audit_denied
  WHERE event_time >= current_timestamp() - INTERVAL 1 DAY {user_filter}
  ORDER BY event_time DESC LIMIT 50"""))

# COMMAND ----------

log_event("audit", "complete", groups_missing=missing)
