# Databricks notebook source
# MAGIC %md
# MAGIC # Step 11 - Deploy the agent with Model Serving (+ AI Gateway)
# MAGIC
# MAGIC 1. **`agents.deploy()`** serves `wb_demo.ai.worldbank_agent@champion` (the version that passed the Step 10 gate)
# MAGIC    on a Model Serving endpoint, with automatic credentials for the resources declared in Step 9, and a
# MAGIC    **Review App** where people can chat with the agent and leave feedback.
# MAGIC 2. **AI Gateway** on the endpoint: **inference tables** (every request/response → Delta in `wb_demo.ops`),
# MAGIC    **usage tracking** (→ `system.serving` tables) and a per-user **rate limit**.
# MAGIC 3. A live test call through the REST API - exactly how the chat app (Step 12) will call it.
# MAGIC
# MAGIC Re-runnable: if the endpoint already serves the champion version, it just re-checks and re-tests.
# MAGIC In the Lakeflow Job (Step 13) this task only deploys when `run_evaluation` reported `eval_passed = True`.
# MAGIC
# MAGIC First deployment: ~15–25 min (container build + start). `SCALE_TO_ZERO=True` stops billing when idle,
# MAGIC but the first request after idling takes minutes - set it to False on demo day.

# COMMAND ----------

# MAGIC %pip install -q "mlflow-skinny[databricks]==3.16.1" "databricks-agents==1.12.0" "openai>=1.60,<3" "protobuf>=6,<7"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import time, json
import mlflow
from mlflow import MlflowClient
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig, AiGatewayRateLimit, AiGatewayRateLimitKey, AiGatewayRateLimitRenewalPeriod,
    AiGatewayUsageTrackingConfig,
)

ENDPOINT_NAME = "wb-demo-agent"
SCALE_TO_ZERO = True          # False on demo day, so the first question is answered in seconds
RATE_LIMIT_PER_USER_PER_MIN = 20
# True → redeploy even if this version is already served. Needed after re-creating a declared resource
# (e.g. the vector index): the endpoint's credentials are issued for the resources as they existed at deploy time.
FORCE_REDEPLOY = False

w = WorkspaceClient()
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

# In the Lakeflow Job, only deploy if the evaluation gate passed (outside a job this defaults to True)
try:
    eval_passed = dbutils.jobs.taskValues.get(taskKey="run_evaluation", key="eval_passed", default=True, debugValue=True)
except Exception:
    eval_passed = True
if not eval_passed:
    log_event("deploy", "skipped_eval_failed", level="WARNING")
    dbutils.notebook.exit("Evaluation gate not passed - nothing deployed.")

champion = client.get_model_version_by_alias(AGENT_MODEL_NAME, "champion")
VERSION = int(champion.version)
print(f"Deploying {AGENT_MODEL_NAME} version {VERSION} (@champion) → endpoint {ENDPOINT_NAME}")
log_event("deploy", "started", model=AGENT_MODEL_NAME, version=VERSION, endpoint=ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md ## 1. Deploy with the Agent Framework

# COMMAND ----------

def served_versions(name: str) -> set[str]:
    try:
        ep = w.serving_endpoints.get(name)
        return {str(e.entity_version) for e in (ep.config.served_entities or [])} if ep.config else set()
    except Exception:
        return set()

if str(VERSION) in served_versions(ENDPOINT_NAME) and not FORCE_REDEPLOY:
    print(f"Endpoint already serves version {VERSION} - skipping deploy")
    deployment = next((d for d in agents.list_deployments() if d.endpoint_name == ENDPOINT_NAME), None)
else:
    deployment = agents.deploy(
        AGENT_MODEL_NAME, VERSION,
        endpoint_name=ENDPOINT_NAME,
        scale_to_zero=SCALE_TO_ZERO,
        workload_size="Small",
        tags={"project": "wb_demo", "model_version": str(VERSION)},
        description="World Bank multi-agent assistant (supervisor + data analyst + research).",
    )
    print("Deployment requested:", deployment.endpoint_name)

# COMMAND ----------

# MAGIC %md ### Wait until the endpoint is ready
# MAGIC Checks every minute for up to 30 min and stops **without failing** - the build continues server-side;
# MAGIC re-run this cell if needed. Progress: **Serving → `wb-demo-agent` → Events / Logs**.

# COMMAND ----------

def endpoint_state():
    ep = w.serving_endpoints.get(ENDPOINT_NAME)
    ready = ep.state.ready.value if ep.state and ep.state.ready else "?"
    update = ep.state.config_update.value if ep.state and ep.state.config_update else "?"
    return ready, update

deadline, ready = time.time() + 30 * 60, False
while time.time() < deadline:
    r, u = endpoint_state()
    print(f"{time.strftime('%H:%M:%S')}  ready={r}  config_update={u}")
    if u == "UPDATE_FAILED":
        raise RuntimeError("Deployment failed - open Serving → wb-demo-agent → Logs (build logs show the cause)")
    if r == "READY" and u == "NOT_UPDATING":
        ready = True
        break
    time.sleep(60)
if not ready:
    print("⏳ Still deploying - it continues in the background. Re-run this cell later.")

# COMMAND ----------

# MAGIC %md ## 2. AI Gateway: inference tables, usage tracking, rate limit
# MAGIC `put_ai_gateway` replaces the whole gateway config, so existing settings are kept and only ours are changed.

# COMMAND ----------

ep = w.serving_endpoints.get(ENDPOINT_NAME)
current = ep.ai_gateway
if current and current.inference_table_config and current.inference_table_config.enabled:
    itc = current.inference_table_config   # already on (e.g. set by agents.deploy) - keep its location
    print(f"Inference table already enabled: {itc.catalog_name}.{itc.schema_name}.{itc.table_name_prefix}_payload")
else:
    itc = AiGatewayInferenceTableConfig(catalog_name=CATALOG, schema_name="ops", table_name_prefix="wb_agent", enabled=True)

try:
    w.serving_endpoints.put_ai_gateway(
        ENDPOINT_NAME,
        inference_table_config=itc,
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        rate_limits=[AiGatewayRateLimit(calls=RATE_LIMIT_PER_USER_PER_MIN, key=AiGatewayRateLimitKey.USER,
                                        renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
        guardrails=current.guardrails if current else None,
        fallback_config=current.fallback_config if current else None,
    )
    INFERENCE_TABLE = f"{itc.catalog_name}.{itc.schema_name}.{itc.table_name_prefix}_payload"
    print(f"✓ AI Gateway: inference table {INFERENCE_TABLE}, usage tracking on, {RATE_LIMIT_PER_USER_PER_MIN} calls/user/min")
except Exception as e:
    INFERENCE_TABLE = None
    print("⚠ AI Gateway config not applied:", str(e).splitlines()[0][:250])
    print("  Set it in the UI: Serving → wb-demo-agent → Edit AI Gateway")

# COMMAND ----------

# MAGIC %md ## 3. Live test through the REST API (what the chat app does)

# COMMAND ----------

def ask_endpoint(question: str) -> dict:
    t0 = time.time()
    resp = w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT_NAME}/invocations",
                           body={"input": [{"role": "user", "content": question}]})
    text = resp["output"][0]["content"][0]["text"]
    co = resp.get("custom_outputs", {})
    print(f"{time.time() - t0:.1f}s   🧭 {co.get('agent_path')}\n\n{text}\n")
    return resp

if ready:
    ask_endpoint("How has India's GDP growth compared to Bangladesh since 2010, and what does the World Bank say "
                 "are the main risks to South Asia's growth?")
else:
    print("Endpoint not ready yet - re-run the wait cell, then this one.")

# COMMAND ----------

# MAGIC %md ## 4. Links and permissions

# COMMAND ----------

host = w.config.host.rstrip("/")
review_url = getattr(deployment, "review_app_url", None) if deployment else None
displayHTML(f"""
<p><a href="{host}/ml/endpoints/{ENDPOINT_NAME}" target="_blank">Serving endpoint {ENDPOINT_NAME}</a> (metrics, logs, AI Gateway)</p>
<p>{'<a href="' + review_url + '" target="_blank">Review App</a> - chat with the agent and give feedback' if review_url else 'Review App: see the endpoint page'}</p>
<p>AI Playground: pick <b>{ENDPOINT_NAME}</b> as the model to chat with it</p>
<p>Inference table: <code>{INFERENCE_TABLE or 'not configured'}</code> (rows appear a few minutes after requests)</p>""")

# The chat app's service principal needs CAN_QUERY - granted in Step 12 when the app exists.
# Analysts group (Step 15): agents.set_permissions(AGENT_MODEL_NAME, [...], agents.PermissionLevel.CAN_QUERY)

log_event("deploy", "complete" if ready else "pending", model=AGENT_MODEL_NAME, version=VERSION,
          endpoint=ENDPOINT_NAME, inference_table=INFERENCE_TABLE)
