# Databricks notebook source
# MAGIC %md
# MAGIC # Step 9 - Log the agent with MLflow and register it in Unity Catalog
# MAGIC
# MAGIC 1. **Log as code** - MLflow stores `worldbank_agent.py` + `agent_config.yaml` (written by `08_agent`) in an
# MAGIC    experiment run, with parameters, tags and a sample prediction.
# MAGIC 2. **Declare resources** - the LLM, embedding model, SQL warehouse, the 5 UC functions, the vector index and the
# MAGIC    gold tables. Model Serving uses this list to give the deployed agent short-lived credentials for exactly
# MAGIC    these objects (automatic auth passthrough) - no personal tokens in the endpoint.
# MAGIC 3. **Validate** - reload the logged model and run it, so packaging problems show up now, not at deploy time.
# MAGIC 4. **Register in UC** as `wb_demo.ai.worldbank_agent` - versioned, governed by grants, with lineage.
# MAGIC
# MAGIC Run `08_agent` first (it writes the two files into this folder).

# COMMAND ----------

# MAGIC %pip install -q "mlflow-skinny[databricks]==3.16.1" "openai>=1.60,<3" "protobuf>=6,<7"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os, yaml, mlflow
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksFunction, DatabricksServingEndpoint, DatabricksSQLWarehouse, DatabricksTable, DatabricksVectorSearchIndex,
)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me().user_name
mlflow.set_experiment(f"/Users/{me}/wb_demo_agent")
mlflow.set_registry_uri("databricks-uc")

AGENT_FILE = os.path.join(os.getcwd(), "worldbank_agent.py")
CONFIG_FILE = os.path.join(os.getcwd(), "agent_config.yaml")
for f in (AGENT_FILE, CONFIG_FILE):
    assert os.path.exists(f), f"{f} not found - run 08_agent first"
agent_config = yaml.safe_load(open(CONFIG_FILE))
print(agent_config)

log_event("agent_registry", "started", model=AGENT_MODEL_NAME)

# COMMAND ----------

# MAGIC %md ## 1. Resources the agent needs

# COMMAND ----------

UC_FUNCTIONS = [f"{CATALOG}.ai.{n}" for n in
                ("list_indicators", "get_indicator", "compare_countries", "get_country_profile", "search_reports")]
UC_TABLES = [f"{CATALOG}.gold.{t}" for t in
             ("country_year_facts", "indicator_definitions", "indicator_values_long", "doc_chunks")]

resources = [
    DatabricksServingEndpoint(endpoint_name=agent_config["llm_endpoint"]),
    DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),          # used by vector_search() to embed the query
    DatabricksSQLWarehouse(warehouse_id=agent_config["warehouse_id"]),   # runs the UC function tools
    DatabricksVectorSearchIndex(index_name=VS_INDEX),
    *[DatabricksFunction(function_name=f) for f in UC_FUNCTIONS],
    *[DatabricksTable(table_name=t) for t in UC_TABLES],
]
for r in resources:
    print(type(r).__name__, r.to_dict())

# COMMAND ----------

# MAGIC %md ## 2. Log the agent (models-from-code)

# COMMAND ----------

INPUT_EXAMPLE = {"input": [{"role": "user", "content": "What was Kenya's GDP growth in 2019?"}]}

# The serving container builds a fresh environment, so it gets full MLflow (no serverless kernel constraints there)
PIP_REQUIREMENTS = ["mlflow==3.16.1", "openai>=1.60,<3", "databricks-sdk>=0.50", "pyyaml"]

with mlflow.start_run(run_name="worldbank_agent") as run:
    mlflow.log_params({
        "llm_endpoint": agent_config["llm_endpoint"],
        "embedding_endpoint": EMBEDDING_ENDPOINT,
        "vector_index": VS_INDEX,
        "architecture": "supervisor + data_analyst + research",
        "tools": ",".join(f.split(".")[-1] for f in UC_FUNCTIONS),
    })
    mlflow.set_tags({"project": "wb_demo", "stage": "candidate"})
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT_FILE,
        model_config=CONFIG_FILE,
        resources=resources,
        input_example=INPUT_EXAMPLE,       # also runs the agent once → a trace + an inferred signature
        pip_requirements=PIP_REQUIREMENTS,
    )
    # Lineage: record which governed tables the agent reads (shows on the run and the model version)
    for t in UC_TABLES:
        try:
            mlflow.log_input(mlflow.data.load_delta(table_name=t, name=t), context="agent_source")
        except Exception as e:
            print(f"(lineage for {t} skipped: {str(e).splitlines()[0][:100]})")

print("Run:", run.info.run_id)
print("Model URI:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md ## 3. Validate: reload the logged model and ask the demo question

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
result = loaded.predict({"input": [{"role": "user", "content":
    "How has India's GDP growth compared to Bangladesh since 2010, and what does the World Bank say are the main risks to South Asia's growth?"}]})
print("🧭", result["custom_outputs"]["agent_path"])
print(result["output"][0]["content"][0]["text"])

# COMMAND ----------

# MAGIC %md ## 4. Register in Unity Catalog

# COMMAND ----------

mv = mlflow.register_model(model_info.model_uri, AGENT_MODEL_NAME, tags={"llm": agent_config["llm_endpoint"]})
client = MlflowClient(registry_uri="databricks-uc")

client.update_registered_model(
    AGENT_MODEL_NAME,
    description=("World Bank development-data assistant. Supervisor routes questions to a Data Analyst agent "
                 f"(UC functions over {CATALOG}.gold) and a Research agent (Vector Search over World Bank reports). "
                 "Built in notebooks 07–09; evaluated in 10; served in 11."),
)
client.update_model_version(AGENT_MODEL_NAME, mv.version,
                            description=f"LLM {agent_config['llm_endpoint']}; logged from run {run.info.run_id}")
client.set_registered_model_alias(AGENT_MODEL_NAME, "candidate", mv.version)   # Step 10 promotes it to "champion"
print(f"✅ Registered {AGENT_MODEL_NAME} version {mv.version} (alias: candidate)")

# COMMAND ----------

# MAGIC %md ## 5. Governance on the model

# COMMAND ----------

# Registered models use the EXECUTE privilege: only granted principals can load or serve it
stmt = f"GRANT EXECUTE ON FUNCTION {AGENT_MODEL_NAME} TO `{GROUP_AI_APP}`"
try:
    spark.sql(stmt)
    print("✓", stmt)
except Exception as e:
    print("⚠", stmt, "→", str(e).splitlines()[0][:150])

for k, v in {"layer": "ai", "project": "wb_demo", "architecture": "multi-agent"}.items():
    client.set_registered_model_tag(AGENT_MODEL_NAME, k, v)

host = w.config.host.rstrip("/")
displayHTML(f"""
<p><a href="{host}/explore/data/models/{AGENT_MODEL_NAME.replace('.', '/')}" target="_blank">Model in Catalog Explorer</a>
 (versions, aliases, lineage, permissions)</p>
<p><a href="{host}/ml/experiments/{run.info.experiment_id}/runs/{run.info.run_id}" target="_blank">MLflow run</a>
 (params, logged code, resources, trace of the input example)</p>""")

log_event("agent_registry", "complete", model=AGENT_MODEL_NAME, version=mv.version, run_id=run.info.run_id)
