# Databricks notebook source
# MAGIC %md
# MAGIC # Step 12 - Chat interface as a Databricks App (Streamlit)
# MAGIC
# MAGIC The app `wb-demo-chat` calls the `wb-demo-agent` serving endpoint and shows, for every answer:
# MAGIC * the **answer**,
# MAGIC * the **agent path** (e.g. *Supervisor → Data Analyst Agent + Research Agent*) and latency,
# MAGIC * the **sources** - report passages with title, year, pages and link; SQL tools with their arguments,
# MAGIC * "How this answer was produced" - the supervisor's routing and every Unity Catalog tool call,
# MAGIC * 👍 / 👎 feedback.
# MAGIC
# MAGIC **Security model:** the app runs as its own **service principal**. It gets exactly two resources -
# MAGIC `CAN_QUERY` on the agent endpoint and `CAN_USE` on the SQL warehouse - plus `MODIFY` on `ops.app_logs`,
# MAGIC where it writes one row per request/feedback (application logs, Step 14). Users sign in with Databricks SSO.
# MAGIC
# MAGIC This notebook writes the app files into `./app/`, creates the app, grants its principal, and deploys.
# MAGIC Source of truth in the repo: `app/`. First run ~5–10 min.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.60" "protobuf>=6,<7"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os
os.makedirs("app", exist_ok=True)
APP_NAME = "wb-demo-chat"
ENDPOINT_NAME = "wb-demo-agent"
print("App source folder:", os.path.join(os.getcwd(), "app"))

# COMMAND ----------

# {{APP_PY}}

# COMMAND ----------

# {{APP_YAML}}

# COMMAND ----------

# {{APP_REQUIREMENTS}}

# COMMAND ----------

# MAGIC %md ## 1. Create the app with its resources

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App, AppDeployment, AppResource, AppResourceServingEndpoint, AppResourceServingEndpointServingEndpointPermission,
    AppResourceSqlWarehouse, AppResourceSqlWarehouseSqlWarehousePermission,
)

w = WorkspaceClient()
warehouses = list(w.warehouses.list())
WAREHOUSE_ID = next((wh.id for wh in warehouses if wh.enable_serverless_compute), warehouses[0].id)

resources = [
    AppResource(name="serving-endpoint", description="World Bank multi-agent endpoint",
                serving_endpoint=AppResourceServingEndpoint(
                    name=ENDPOINT_NAME, permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY)),
    AppResource(name="sql-warehouse", description="Writes app logs to wb_demo.ops.app_logs",
                sql_warehouse=AppResourceSqlWarehouse(
                    id=WAREHOUSE_ID, permission=AppResourceSqlWarehouseSqlWarehousePermission.CAN_USE)),
]

existing = {a.name for a in w.apps.list()}
if APP_NAME not in existing:
    print(f"Creating app {APP_NAME} (starts its compute, ~2–5 min)…")
    app = w.apps.create(App(name=APP_NAME, description="World Bank AI Assistant - multi-agent chat",
                            resources=resources)).result()
else:
    app = w.apps.update(APP_NAME, App(name=APP_NAME, description="World Bank AI Assistant - multi-agent chat",
                                      resources=resources))
    if str(app.compute_status and app.compute_status.state).endswith("STOPPED"):
        app = w.apps.start(APP_NAME).result()
print("Service principal:", app.service_principal_name, "| client id:", app.service_principal_client_id)

# COMMAND ----------

# MAGIC %md ## 2. Least-privilege grants for the app's service principal

# COMMAND ----------

sp = app.service_principal_client_id
for stmt in [f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{sp}`",
             f"GRANT USE SCHEMA ON SCHEMA {CATALOG}.ops TO `{sp}`",
             f"GRANT SELECT, MODIFY ON TABLE {CATALOG}.ops.app_logs TO `{sp}`"]:
    try:
        spark.sql(stmt)
        print("✓", stmt)
    except Exception as e:
        print("⚠", stmt, "→", str(e).splitlines()[0][:200])

# COMMAND ----------

# MAGIC %md ## 3. Deploy the code

# COMMAND ----------

import time

# Only one deployment may run at a time - wait for any in-progress one (e.g. from an earlier run) to finish
deadline = time.time() + 15 * 60
while time.time() < deadline:
    pending = w.apps.get(APP_NAME).pending_deployment
    state = pending.status.state.value if pending and pending.status and pending.status.state else None
    if state != "IN_PROGRESS":
        break
    print(f"{time.strftime('%H:%M:%S')}  waiting for the previous deployment to finish…")
    time.sleep(30)

source_path = os.path.join(os.getcwd(), "app")
print("Deploying from", source_path, "…")
deployment = w.apps.deploy(APP_NAME, AppDeployment(source_code_path=source_path)).result()
print("Deployment:", deployment.status.state if deployment.status else "?", "-", deployment.status.message if deployment.status else "")

app = w.apps.get(APP_NAME)
displayHTML(f'<h3>✅ <a href="{app.url}" target="_blank">Open the World Bank AI Assistant</a></h3>'
            f'<p>Logs: Compute → Apps → {APP_NAME} → Logs · Every request is also in <code>{CATALOG}.ops.app_logs</code></p>')
log_event("chat_app", "deployed", app=APP_NAME, url=app.url, endpoint=ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md ## 4. After you chat a bit - the app's requests in the governed log table

# COMMAND ----------

display(spark.sql(f"""
  SELECT event_ts, event, get_json_object(details, '$.user') AS user,
         get_json_object(details, '$.question') AS question,
         get_json_object(details, '$.agent_path') AS agent_path,
         get_json_object(details, '$.latency_s') AS latency_s
  FROM {CATALOG}.ops.app_logs WHERE component = 'chat_app'
  ORDER BY event_ts DESC LIMIT 20
"""))
