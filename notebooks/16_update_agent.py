# Databricks notebook source
# MAGIC %md
# MAGIC # 16_update_agent - ship an agent code change in one run
# MAGIC Runs the release steps in order: write and test the agent (08) → log and register a new version (09) →
# MAGIC promote it to `@champion` → deploy `@champion` to Model Serving (11).
# MAGIC
# MAGIC Use it for code changes that don't alter prompts or behaviour (here: how the Research agent calls Vector Search).
# MAGIC For prompt or logic changes, run `10_evaluate` instead of the promote step so the quality gate decides.

# COMMAND ----------

import time

def step(name: str, timeout_min: int):
    t0 = time.time()
    print(f"▶ {name} …")
    result = dbutils.notebook.run(f"./{name}", timeout_min * 60)
    print(f"✓ {name} finished in {(time.time() - t0) / 60:.1f} min {('- ' + result) if result else ''}")

step("08_agent", 30)

# COMMAND ----------

step("09_log_register", 30)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

MODEL = "wb_demo.ai.worldbank_agent"
w = WorkspaceClient()
try:
    info = w.registered_models.get(MODEL, include_aliases=True)
except TypeError:   # older SDK on serverless doesn't take include_aliases (aliases are returned by default)
    info = w.registered_models.get(MODEL)
aliases = {a.alias_name: a.version_num for a in (info.aliases or [])}
candidate = aliases["candidate"]
w.registered_models.set_alias(MODEL, "champion", candidate)
print(f"✓ {MODEL} version {candidate} → @champion (was {aliases.get('champion')})")

# COMMAND ----------

step("11_deploy", 60)
print("Done - open the chat app and try the Document and Mixed examples.")
