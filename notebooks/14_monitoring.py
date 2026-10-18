# Databricks notebook source
# MAGIC %md
# MAGIC # Step 14 - Monitoring: application logs, agent quality, jobs, serving and cost
# MAGIC
# MAGIC One place to answer "is the assistant healthy?". Governed views in `wb_demo.ops` over:
# MAGIC
# MAGIC | View | Source | What it shows |
# MAGIC |---|---|---|
# MAGIC | `v_chat_requests` | `ops.app_logs` (written by the chat app) | requests, latency, errors, 👍/👎, agent path per question |
# MAGIC | `v_pipeline_events` | `ops.app_logs` (written by every notebook/job task) | ingestion counts, index syncs, deploys |
# MAGIC | `v_eval_scores` | `ops.eval_results` (Step 10) | judge scores per evaluation run, vs thresholds |
# MAGIC | `v_job_runs` | `system.lakeflow.job_run_timeline` | pipeline runs, duration, result |
# MAGIC | `v_serving_usage` | `system.serving.endpoint_usage` | requests and tokens per endpoint (agent, LLM, embeddings) |
# MAGIC | `v_cost` | `system.billing.usage` + `list_prices` | DBUs and estimated list cost per product, last 30 days |
# MAGIC
# MAGIC …and publishes the **"World Bank Assistant - Monitoring"** AI/BI dashboard on top of them.
# MAGIC Agent **traces** (per-request spans, tokens, latency) are in MLflow: **Experiments → wb_demo_agent → Traces**.
# MAGIC
# MAGIC Views over system tables that this workspace can't read are skipped (with the reason) instead of failing.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
C = CATALOG
ENDPOINT_NAME, JOB_NAME = "wb-demo-agent", "wb_demo_pipeline"
DASHBOARD_NAME = "World Bank Assistant - Monitoring"
job_id = next((j.job_id for j in w.jobs.list(name=JOB_NAME)), None)
print("Job id:", job_id)
log_event("monitoring", "started")

# COMMAND ----------

# MAGIC %md ## 1. Monitoring views

# COMMAND ----------

views = {
    "v_chat_requests": f"""
        SELECT event_ts, date(event_ts) AS day, event, level,
               get_json_object(details, '$.user')                      AS user,
               get_json_object(details, '$.question')                  AS question,
               CAST(get_json_object(details, '$.latency_s') AS DOUBLE) AS latency_s,
               get_json_object(details, '$.agent_path')                AS agent_path,
               CAST(get_json_object(details, '$.sources') AS INT)      AS sources,
               get_json_object(details, '$.rating')                    AS rating,
               get_json_object(details, '$.error')                     AS error
        FROM {C}.ops.app_logs
        WHERE component = 'chat_app' AND event IN ('chat_request', 'chat_error', 'feedback')""",
    "v_pipeline_events": f"""
        SELECT event_ts, date(event_ts) AS day, component, event, level, details
        FROM {C}.ops.app_logs WHERE component <> 'chat_app'""",
    "v_eval_scores": f"""
        SELECT eval_ts, agent_version, run_id, metric, value, threshold,
               CASE WHEN threshold IS NULL THEN NULL ELSE value >= threshold END AS passed
        FROM {C}.ops.eval_results""",
    "v_job_runs": f"""
        SELECT run_id,
               min(period_start_time)                               AS start_time,
               max(period_end_time)                                 AS end_time,
               round((unix_timestamp(max(period_end_time)) - unix_timestamp(min(period_start_time))) / 60, 1) AS duration_min,
               coalesce(max_by(result_state, period_end_time), 'RUNNING') AS result_state,
               max(trigger_type)                                    AS trigger_type
        FROM system.lakeflow.job_run_timeline
        WHERE job_id = '{job_id}'
        GROUP BY run_id""",
    "v_serving_usage": f"""
        SELECT u.request_time, date(u.request_time) AS day, e.endpoint_name, u.requester, u.status_code,
               u.input_token_count, u.output_token_count
        FROM system.serving.endpoint_usage u
        JOIN (SELECT DISTINCT served_entity_id, endpoint_name FROM system.serving.served_entities) e USING (served_entity_id)
        WHERE e.endpoint_name IN ('{ENDPOINT_NAME}', '{LLM_ENDPOINT}', '{EMBEDDING_ENDPOINT}')""",
    "v_cost": """
        SELECT u.usage_date, u.billing_origin_product AS product, u.sku_name,
               sum(u.usage_quantity)                                        AS dbus,
               round(sum(u.usage_quantity * CAST(p.pricing.default AS DOUBLE)), 2) AS est_list_cost_usd
        FROM system.billing.usage u
        LEFT JOIN system.billing.list_prices p
          ON u.sku_name = p.sku_name AND u.usage_start_time >= p.price_start_time
         AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
        WHERE u.usage_date >= current_date() - 30
        GROUP BY ALL""",
}

available = set()
for name, body in views.items():
    try:
        spark.sql(f"CREATE OR REPLACE VIEW {C}.ops.{name} AS {body}")
        n = spark.table(f"{C}.ops.{name}").count()
        available.add(name)
        print(f"✓ ops.{name:<18} {n} rows")
    except Exception as e:
        print(f"⚠ ops.{name:<18} skipped → {str(e).splitlines()[0][:160]}")

# COMMAND ----------

# MAGIC %md ## 2. Quick look (also useful live in the demo)

# COMMAND ----------

if "v_chat_requests" in available:
    display(spark.sql(f"""
      SELECT count_if(event = 'chat_request')                                    AS requests,
             count_if(event = 'chat_error')                                      AS errors,
             round(avg(latency_s) FILTER (WHERE event = 'chat_request'), 1)      AS avg_latency_s,
             round(percentile(latency_s, 0.95) FILTER (WHERE event = 'chat_request'), 1) AS p95_latency_s,
             count_if(rating = 'up') AS thumbs_up, count_if(rating = 'down') AS thumbs_down
      FROM {C}.ops.v_chat_requests"""))

# COMMAND ----------

if "v_job_runs" in available:
    display(spark.sql(f"SELECT * FROM {C}.ops.v_job_runs ORDER BY start_time DESC LIMIT 10"))

# COMMAND ----------

# MAGIC %md ## 3. Monitoring dashboard

# COMMAND ----------

def ds_if(view, sql):
    return sql if view in available else None

datasets = {k: v for k, v in {
    "kpi_chat": ds_if("v_chat_requests", f"""
        SELECT count_if(event = 'chat_request') AS requests_7d,
               round(avg(latency_s) FILTER (WHERE event = 'chat_request'), 1) AS avg_latency_s,
               round(100 * count_if(event = 'chat_error') / greatest(count_if(event IN ('chat_request', 'chat_error')), 1), 1) AS error_rate_pct,
               round(100 * count_if(rating = 'up') / greatest(count_if(event = 'feedback'), 1), 0) AS thumbs_up_pct
        FROM {C}.ops.v_chat_requests WHERE event_ts >= current_timestamp() - INTERVAL 7 DAYS"""),
    "kpi_eval": ds_if("v_eval_scores", f"""
        SELECT round(max_by(value, eval_ts), 2) AS latest_correctness
        FROM {C}.ops.v_eval_scores WHERE agent_version = 'v2' AND metric = 'correctness'"""),
    "requests_daily": ds_if("v_chat_requests", f"""
        SELECT day, event, count(*) AS events FROM {C}.ops.v_chat_requests GROUP BY ALL"""),
    "latency_daily": ds_if("v_chat_requests", f"""
        SELECT day, 'p50' AS stat, percentile(latency_s, 0.5) AS latency_s FROM {C}.ops.v_chat_requests WHERE event = 'chat_request' GROUP BY day
        UNION ALL
        SELECT day, 'p95', percentile(latency_s, 0.95) FROM {C}.ops.v_chat_requests WHERE event = 'chat_request' GROUP BY day"""),
    "agent_paths": ds_if("v_chat_requests", f"""
        SELECT coalesce(agent_path, 'unknown') AS agent_path, count(*) AS requests
        FROM {C}.ops.v_chat_requests WHERE event = 'chat_request' GROUP BY 1"""),
    "eval_trend": ds_if("v_eval_scores", f"""
        SELECT eval_ts, metric, value FROM {C}.ops.v_eval_scores
        WHERE agent_version = 'v2' AND metric IN ('correctness', 'follows_guidelines', 'cites_source', 'correct_route', 'groundedness')"""),
    "job_runs": ds_if("v_job_runs", f"SELECT start_time, duration_min, result_state FROM {C}.ops.v_job_runs"),
    "serving": ds_if("v_serving_usage", f"""
        SELECT day, endpoint_name, count(*) AS requests FROM {C}.ops.v_serving_usage GROUP BY ALL"""),
    "cost": ds_if("v_cost", f"SELECT usage_date, product, sum(dbus) AS dbus FROM {C}.ops.v_cost GROUP BY ALL"),
}.items() if v}

for name, q in list(datasets.items()):   # validate every query now, drop any that fail
    try:
        spark.sql(q).limit(1).collect()
    except Exception as e:
        print(f"⚠ dataset {name} dropped → {str(e).splitlines()[0][:150]}")
        datasets.pop(name)
print("Datasets:", list(datasets))

# COMMAND ----------

def widget(name, dataset, cols, spec, pos):
    return {"widget": {"name": name, "queries": [{"name": "main_query", "query": {
                "datasetName": dataset, "fields": [{"name": c, "expression": f"`{c}`"} for c in dict.fromkeys(cols)],
                "disaggregated": True}}], "spec": spec},
            "position": dict(zip(("x", "y", "width", "height"), pos))}

def counter(name, dataset, col, title, x, y):
    return widget(name, dataset, [col], {"version": 2, "widgetType": "counter",
                  "encodings": {"value": {"fieldName": col, "displayName": title}},
                  "frame": {"showTitle": True, "title": title}}, (x, y, 1, 2))

def chart(kind, name, dataset, x, y, color, title, x_type, pos):
    enc = {"x": {"fieldName": x, "scale": {"type": x_type}, "displayName": x},
           "y": {"fieldName": y, "scale": {"type": "quantitative"}, "displayName": y}}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    return widget(name, dataset, [x, y] + ([color] if color else []),
                  {"version": 3, "widgetType": kind, "encodings": enc, "frame": {"showTitle": True, "title": title}}, pos)

layout = [{"widget": {"name": "header", "multilineTextboxSpec": {"lines": [
    "# World Bank Assistant - Monitoring",
    f"Chat app requests & feedback (`{C}.ops.app_logs`), agent evaluation scores, pipeline job runs, "
    "serving usage and cost. Per-request traces: MLflow experiment `wb_demo_agent`."]}},
    "position": {"x": 0, "y": 0, "width": 6, "height": 2}}]

kpis = [("kpi_chat", "requests_7d", "Requests (7 days)"), ("kpi_chat", "avg_latency_s", "Avg latency (s)"),
        ("kpi_chat", "error_rate_pct", "Error rate %"), ("kpi_chat", "thumbs_up_pct", "👍 rate %"),
        ("kpi_eval", "latest_correctness", "Eval correctness")]
x = 0
for ds, col, title in kpis:
    if ds in datasets:
        layout.append(counter(f"k_{col}", ds, col, title, x, 2))
        x += 1

charts = [
    ("bar",  "c_requests", "requests_daily", "day", "events", "event", "Chat events per day", "temporal"),
    ("line", "c_latency", "latency_daily", "day", "latency_s", "stat", "Latency p50 / p95 (s)", "temporal"),
    ("bar",  "c_paths", "agent_paths", "agent_path", "requests", None, "Agent path taken", "categorical"),
    ("line", "c_eval", "eval_trend", "eval_ts", "value", "metric", "Evaluation scores over time (v2)", "temporal"),
    ("bar",  "c_jobs", "job_runs", "start_time", "duration_min", "result_state", "Pipeline runs (minutes, by result)", "temporal"),
    ("bar",  "c_serving", "serving", "day", "requests", "endpoint_name", "Serving requests per endpoint", "temporal"),
    ("bar",  "c_cost", "cost", "usage_date", "dbus", "product", "DBUs per day by product (30 days)", "temporal"),
]
slot = 0
for kind, name, ds, xc, yc, color, title, xt in charts:
    if ds in datasets:
        layout.append(chart(kind, name, ds, xc, yc, color, title, xt, ((slot % 2) * 3, 4 + (slot // 2) * 6, 3, 6)))
        slot += 1

serialized = json.dumps({
    "datasets": [{"name": n, "displayName": n.replace("_", " ").title(), "queryLines": [q.strip()]} for n, q in datasets.items()],
    "pages": [{"name": "monitoring", "displayName": "Monitoring", "layout": layout}],
})

# COMMAND ----------

from databricks.sdk.service.dashboards import Dashboard

warehouses = list(w.warehouses.list())
WAREHOUSE_ID = next((wh.id for wh in warehouses if wh.enable_serverless_compute), warehouses[0].id)
parent = f"/Users/{w.current_user.me().user_name}"

for obj in w.workspace.list(parent):
    if str(obj.object_type).endswith("DASHBOARD") and obj.path.split("/")[-1].startswith("World Bank Assistant"):
        try:
            w.lakeview.trash(obj.resource_id)
        except Exception as e:
            print("Could not remove old dashboard:", e)

try:
    dash = w.lakeview.create(dashboard=Dashboard(display_name=DASHBOARD_NAME, parent_path=parent,
                                                 serialized_dashboard=serialized, warehouse_id=WAREHOUSE_ID))
except TypeError:
    dash = w.lakeview.create(display_name=DASHBOARD_NAME, parent_path=parent,
                             serialized_dashboard=serialized, warehouse_id=WAREHOUSE_ID)
w.lakeview.publish(dash.dashboard_id, embed_credentials=True, warehouse_id=WAREHOUSE_ID)
url = f"{w.config.host.rstrip('/')}/dashboardsv3/{dash.dashboard_id}/published"
displayHTML(f'<h3>✅ <a href="{url}" target="_blank">{DASHBOARD_NAME}</a></h3>'
            f'<p>{len(datasets)} datasets, {len(layout) - 1} widgets. Views skipped: {sorted(set(views) - available) or "none"}</p>')
log_event("monitoring", "complete", dashboard_id=dash.dashboard_id, views=sorted(available), datasets=list(datasets))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional extras
# MAGIC * **Alert:** SQL Editor → run `SELECT error_rate_pct FROM` the `kpi_chat` query → **Create alert** (e.g. > 10%).
# MAGIC * **Refresh:** open the dashboard → **Schedule** → daily, so it's current before the demo.
# MAGIC * **Lakehouse Monitoring:** Catalog → `wb_demo.ops.app_logs` → *Quality* → create a monitor to track drift over time.
