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

# MAGIC %%writefile app/app.py
# MAGIC """World Bank AI Assistant - Databricks App (Streamlit).
# MAGIC
# MAGIC Calls the Model Serving endpoint of the multi-agent system and shows, for every answer:
# MAGIC the answer, the agent path taken, the sources (report passages with links, SQL tools with arguments),
# MAGIC and how long it took. Every request is written to wb_demo.ops.app_logs (application log, Step 14)
# MAGIC and to the app's own logs (Apps → wb-demo-chat → Logs).
# MAGIC """
# MAGIC import json
# MAGIC import logging
# MAGIC import os
# MAGIC import time
# MAGIC import uuid
# MAGIC
# MAGIC import streamlit as st
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.sql import StatementParameterListItem
# MAGIC
# MAGIC ENDPOINT = os.environ["SERVING_ENDPOINT"]            # injected from the app's serving-endpoint resource
# MAGIC WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID")
# MAGIC LOG_TABLE = os.environ.get("LOG_TABLE", "wb_demo.ops.app_logs")
# MAGIC
# MAGIC logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# MAGIC log = logging.getLogger("wb-demo-chat")
# MAGIC
# MAGIC EXAMPLES = {
# MAGIC     "🔢 Numeric": "What was Kenya's GDP growth in 2019?",
# MAGIC     "📄 Document": "What risks does the latest Nigeria Development Update highlight?",
# MAGIC     "🔀 Mixed": ("How has India's GDP growth compared to Bangladesh since 2010, and what does the World Bank say "
# MAGIC                 "are the main risks to South Asia's growth?"),
# MAGIC }
# MAGIC
# MAGIC
# MAGIC @st.cache_resource
# MAGIC def workspace() -> WorkspaceClient:
# MAGIC     return WorkspaceClient()   # authenticates as the app's service principal
# MAGIC
# MAGIC
# MAGIC def current_user() -> str:
# MAGIC     try:
# MAGIC         return st.context.headers.get("X-Forwarded-Email") or st.context.headers.get("X-Forwarded-Preferred-Username") or "unknown"
# MAGIC     except Exception:
# MAGIC         return "unknown"
# MAGIC
# MAGIC
# MAGIC def ask_agent(messages: list[dict]) -> dict:
# MAGIC     return workspace().api_client.do("POST", f"/serving-endpoints/{ENDPOINT}/invocations", body={"input": messages})
# MAGIC
# MAGIC
# MAGIC def log_request(event: str, level: str = "INFO", **details):
# MAGIC     """App log → stdout (Apps UI) and → Delta table in Unity Catalog (queried by the monitoring dashboard)."""
# MAGIC     log.log(getattr(logging, level), "%s %s", event, json.dumps(details, default=str))
# MAGIC     if not WAREHOUSE_ID:
# MAGIC         return
# MAGIC     try:
# MAGIC         workspace().statement_execution.execute_statement(
# MAGIC             warehouse_id=WAREHOUSE_ID, wait_timeout="0s",   # fire-and-forget: never slows the chat down
# MAGIC             statement=f"INSERT INTO {LOG_TABLE} VALUES (current_timestamp(), :run_id, 'chat_app', :level, :event, :details)",
# MAGIC             parameters=[StatementParameterListItem(name="run_id", value=st.session_state.session_id),
# MAGIC                         StatementParameterListItem(name="level", value=level),
# MAGIC                         StatementParameterListItem(name="event", value=event),
# MAGIC                         StatementParameterListItem(name="details", value=json.dumps(details, default=str))],
# MAGIC         )
# MAGIC     except Exception as e:
# MAGIC         log.warning("could not write to %s: %s", LOG_TABLE, e)
# MAGIC
# MAGIC
# MAGIC def split_sources(text: str) -> str:
# MAGIC     """The agent appends a '**Sources**' list; the app renders sources itself from custom_outputs."""
# MAGIC     return text.split("\n\n**Sources**")[0]
# MAGIC
# MAGIC
# MAGIC def render_details(meta: dict):
# MAGIC     path = meta.get("agent_path") or "-"
# MAGIC     st.caption(f"🧭 **{path}**   ·   ⏱ {meta.get('latency_s', '?')} s")
# MAGIC
# MAGIC     sources = meta.get("sources") or []
# MAGIC     docs = [s for s in sources if s.get("type") == "document"]
# MAGIC     tables = [s for s in sources if s.get("type") == "table"]
# MAGIC     if docs or tables:
# MAGIC         with st.expander(f"📚 Sources ({len(docs)} report passages, {len(tables)} data queries)", expanded=True):
# MAGIC             for s in docs:
# MAGIC                 title = f"[{s.get('id')}] {s.get('title')} ({s.get('year')}), pp. {s.get('pages')}"
# MAGIC                 st.markdown(f"- [{title}]({s['url']})" if s.get("url") else f"- {title}")
# MAGIC             for s in tables:
# MAGIC                 st.markdown(f"- 🗄️ {s.get('label')}")
# MAGIC
# MAGIC     with st.expander("🔍 How this answer was produced"):
# MAGIC         routing = meta.get("routing") or {}
# MAGIC         st.markdown("**Supervisor routing**")
# MAGIC         st.json({k: v for k, v in routing.items() if v})
# MAGIC         calls = meta.get("tool_calls") or []
# MAGIC         if calls:
# MAGIC             st.markdown("**Tool calls** (Unity Catalog functions)")
# MAGIC             st.dataframe([{"agent": c.get("agent"), "tool": c.get("tool"), "arguments": json.dumps(c.get("args")),
# MAGIC                            "rows": c.get("rows", c.get("error"))} for c in calls], width="stretch")
# MAGIC
# MAGIC
# MAGIC # ── UI ─────────────────────────────────────────────────────────────────────────────────────────
# MAGIC
# MAGIC st.set_page_config(page_title="World Bank AI Assistant", page_icon="🌍", layout="wide")
# MAGIC if "session_id" not in st.session_state:
# MAGIC     st.session_state.session_id = str(uuid.uuid4())
# MAGIC     st.session_state.history = []          # [{"role", "content", "meta"}]
# MAGIC
# MAGIC with st.sidebar:
# MAGIC     st.header("🌍 World Bank AI Assistant")
# MAGIC     st.write("Ask about development indicators for 20 countries (2000 → latest) or about what World Bank "
# MAGIC              "reports say. A supervisor agent routes each question to a **Data Analyst** agent (SQL over "
# MAGIC              "governed gold tables) and/or a **Research** agent (Vector Search over World Bank reports).")
# MAGIC     st.markdown("**Try an example**")
# MAGIC     for label, q in EXAMPLES.items():
# MAGIC         if st.button(label, width="stretch"):
# MAGIC             st.session_state.pending = q
# MAGIC     st.divider()
# MAGIC     if st.button("🗑️ New conversation", width="stretch"):
# MAGIC         st.session_state.history = []
# MAGIC         st.rerun()
# MAGIC     st.caption(f"Endpoint: `{ENDPOINT}` · signed in as {current_user()}")
# MAGIC
# MAGIC st.title("World Bank AI Assistant")
# MAGIC
# MAGIC def give_feedback(i: int, rating: str):
# MAGIC     st.session_state.history[i]["rated"] = rating
# MAGIC     log_request("feedback", user=current_user(), question=st.session_state.history[i - 1]["content"], rating=rating,
# MAGIC                 agent_path=(st.session_state.history[i].get("meta") or {}).get("agent_path"))
# MAGIC
# MAGIC for i, msg in enumerate(st.session_state.history):
# MAGIC     with st.chat_message(msg["role"]):
# MAGIC         st.markdown(msg["content"])
# MAGIC         if msg.get("meta"):
# MAGIC             render_details(msg["meta"])
# MAGIC             if msg.get("rated"):
# MAGIC                 st.caption("Thanks for the feedback " + ("👍" if msg["rated"] == "up" else "👎"))
# MAGIC             else:
# MAGIC                 c1, c2, _ = st.columns([1, 1, 12])
# MAGIC                 c1.button("👍", key=f"up{i}", on_click=give_feedback, args=(i, "up"))
# MAGIC                 c2.button("👎", key=f"down{i}", on_click=give_feedback, args=(i, "down"))
# MAGIC
# MAGIC question = st.chat_input("Ask about GDP, poverty, inflation… or what the World Bank says about a country") \
# MAGIC            or st.session_state.pop("pending", None)
# MAGIC
# MAGIC if question:
# MAGIC     st.session_state.history.append({"role": "user", "content": question})
# MAGIC     with st.chat_message("user"):
# MAGIC         st.markdown(question)
# MAGIC
# MAGIC     with st.chat_message("assistant"):
# MAGIC         with st.spinner("Supervisor is routing your question to the specialist agents…"):
# MAGIC             conversation = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history[-8:]]
# MAGIC             t0 = time.time()
# MAGIC             try:
# MAGIC                 resp = ask_agent(conversation)
# MAGIC                 text = split_sources(resp["output"][0]["content"][0]["text"])
# MAGIC                 meta = {**(resp.get("custom_outputs") or {}), "latency_s": round(time.time() - t0, 1)}
# MAGIC                 log_request("chat_request", user=current_user(), question=question, latency_s=meta["latency_s"],
# MAGIC                             agent_path=meta.get("agent_path"), sources=len(meta.get("sources") or []),
# MAGIC                             tool_calls=len(meta.get("tool_calls") or []))
# MAGIC             except Exception as e:
# MAGIC                 text, meta = f"⚠️ The assistant is unavailable right now ({type(e).__name__}). Please try again.", None
# MAGIC                 log_request("chat_error", level="ERROR", user=current_user(), question=question, error=str(e)[:500],
# MAGIC                             latency_s=round(time.time() - t0, 1))
# MAGIC         st.markdown(text)
# MAGIC         if meta:
# MAGIC             render_details(meta)
# MAGIC     st.session_state.history.append({"role": "assistant", "content": text, "meta": meta})
# MAGIC     st.rerun()   # redraw from history so the new answer gets its feedback buttons

# COMMAND ----------

# MAGIC %%writefile app/app.yaml
# MAGIC command: ["streamlit", "run", "app.py"]
# MAGIC env:
# MAGIC   - name: SERVING_ENDPOINT
# MAGIC     valueFrom: serving-endpoint
# MAGIC   - name: DATABRICKS_WAREHOUSE_ID
# MAGIC     valueFrom: sql-warehouse
# MAGIC   - name: LOG_TABLE
# MAGIC     value: wb_demo.ops.app_logs
# MAGIC   - name: STREAMLIT_BROWSER_GATHER_USAGE_STATS
# MAGIC     value: "false"

# COMMAND ----------

# MAGIC %%writefile app/requirements.txt
# MAGIC streamlit>=1.46
# MAGIC databricks-sdk>=0.50

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
