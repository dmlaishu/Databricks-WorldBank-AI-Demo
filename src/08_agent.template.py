# Databricks notebook source
# MAGIC %md
# MAGIC # Step 8 - Multi-agent system (MLflow ResponsesAgent + Tracing)
# MAGIC
# MAGIC ```
# MAGIC  user question ─▶ Supervisor (LLM router: splits the question, resolves follow-ups)
# MAGIC                     ├─▶ Data Analyst agent ─▶ UC functions: list_indicators · get_indicator · compare_countries · get_country_profile
# MAGIC                     └─▶ Research agent     ─▶ UC function: search_reports (Vector Search)
# MAGIC                   Supervisor merges both answers  +  source list built from the actual tool calls
# MAGIC ```
# MAGIC
# MAGIC * **Tools = Unity Catalog functions (Step 7).** The agent reads each tool's description and parameters from UC
# MAGIC   at start-up and runs them on a SQL warehouse → governed by grants, visible in the audit log.
# MAGIC * **MLflow Tracing** records every step: supervisor routing, each agent, each tool call, each LLM call (tokens, latency).
# MAGIC * **ResponsesAgent** is the MLflow interface that Model Serving, the Review App and AI Playground understand.
# MAGIC * `custom_outputs` returns the **agent path**, **sources** and **tool calls** - the chat app (Step 12) displays them.
# MAGIC
# MAGIC Libraries: only `mlflow-skinny` + `openai` (protobuf pinned to 6.x so the serverless kernel stays healthy).

# COMMAND ----------

# MAGIC %pip install -q "mlflow-skinny[databricks]==3.16.1" "openai>=1.60,<3" "protobuf>=6,<7"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md ## 1. Agent configuration
# MAGIC Written to `agent_config.yaml` next to this notebook. Step 9 logs it with the model, so the deployed agent
# MAGIC uses exactly these values.

# COMMAND ----------

import os, yaml
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
warehouses = list(w.warehouses.list())
WAREHOUSE_ID = next((wh.id for wh in warehouses if wh.enable_serverless_compute), warehouses[0].id if warehouses else None)
assert WAREHOUSE_ID, "No SQL warehouse found"

agent_config = {"catalog": CATALOG, "llm_endpoint": LLM_ENDPOINT, "warehouse_id": WAREHOUSE_ID,
                "prompt_version": "v2"}   # v1 = minimal baseline prompts, used for comparison in Step 10
with open("agent_config.yaml", "w") as f:
    yaml.safe_dump(agent_config, f)
print(os.getcwd(), agent_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Agent code
# MAGIC Written to `worldbank_agent.py` so MLflow can log it **as code** in Step 9 (models-from-code).
# MAGIC Source of truth in the repo: `src/worldbank_agent.py`.

# COMMAND ----------

# {{AGENT_WRITEFILE_CELL}}

# COMMAND ----------

# MAGIC %md ## 3. Try it - with tracing
# MAGIC Each call below shows an **MLflow trace** under the cell output: expand it to see
# MAGIC Supervisor → agents → tool calls → LLM calls, with inputs, outputs, latency and tokens.

# COMMAND ----------

import sys, importlib.util, time, json
import mlflow

me = w.current_user.me().user_name
mlflow.set_experiment(f"/Users/{me}/wb_demo_agent")

# Load the agent straight from the file written above. (A plain `import` can miss a file created
# mid-session because Python caches directory listings of the workspace filesystem.)
AGENT_FILE = os.path.join(os.getcwd(), "worldbank_agent.py")
assert os.path.exists(AGENT_FILE), f"{AGENT_FILE} not found - run the %%writefile cell above first"
spec = importlib.util.spec_from_file_location("worldbank_agent", AGENT_FILE)
worldbank_agent = importlib.util.module_from_spec(spec)
sys.modules["worldbank_agent"] = worldbank_agent
spec.loader.exec_module(worldbank_agent)
AGENT = worldbank_agent.AGENT
print("Agent loaded from", AGENT_FILE)

def ask(messages):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    t0 = time.time()
    resp = AGENT.predict({"input": messages})
    out = resp.model_dump(exclude_none=True)
    text = out["output"][0]["content"][0]["text"]
    co = out.get("custom_outputs", {})
    print(f"{time.time() - t0:.1f}s   🧭 {co.get('agent_path')}")
    print("routing:", json.dumps(co.get("routing"), indent=1))
    print("tools:", [(c["agent"], c["tool"].split(".")[-1], c.get("rows", c.get("error"))) for c in co.get("tool_calls", [])])
    print("\n" + text)
    return text

# COMMAND ----------

# Numeric question → Data Analyst only
ask("What was Kenya's GDP growth in 2019?")

# COMMAND ----------

# Document question → Research only
ask("What risks does the latest Nigeria Development Update highlight?")

# COMMAND ----------

# Mixed question → both agents, merged by the supervisor
DEMO_Q = "How has India's GDP growth compared to Bangladesh since 2010, and what does the World Bank say are the main risks to South Asia's growth?"
first_answer = ask(DEMO_Q)

# COMMAND ----------

# Multi-turn: the supervisor resolves "what about Pakistan?" using the conversation history
ask([
    {"role": "user", "content": DEMO_Q},
    {"role": "assistant", "content": first_answer},
    {"role": "user", "content": "What about Pakistan?"},
])

# COMMAND ----------

# Out of scope → supervisor answers directly, no tools called
ask("Hi! What can you help me with?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Where to look
# MAGIC * **Traces:** under each cell above, or **Experiments → `wb_demo_agent` → Traces** (filter, compare, share).
# MAGIC   Open the mixed question's trace for the demo: it shows both agents collaborating.
# MAGIC * **Governance:** every tool call ran `SELECT * FROM wb_demo.ai.<function>(...)` on the SQL warehouse -
# MAGIC   see **Query History** (SQL sidebar) and, later, `system.access.audit`.

# COMMAND ----------

log_event("agent", "notebook_test_complete", llm=LLM_ENDPOINT, warehouse_id=WAREHOUSE_ID)
