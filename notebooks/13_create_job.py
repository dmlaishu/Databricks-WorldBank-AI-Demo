# Databricks notebook source
# MAGIC %md
# MAGIC # Step 13 - Orchestrate everything with a Lakeflow Job (Workflows)
# MAGIC
# MAGIC Creates (or updates) the job **`wb_demo_pipeline`** - the whole lifecycle as one repeatable DAG:
# MAGIC
# MAGIC ```
# MAGIC ingest_indicators ──▶ build_silver_gold_indicators ─────────────────────────┐
# MAGIC                                                                              ├─▶ run_evaluation ─▶ eval_gate ─(true)─▶ deploy_agent
# MAGIC ingest_documents  ──▶ build_silver_gold_documents ──▶ sync_vector_index ──────┘
# MAGIC ```
# MAGIC
# MAGIC * Every task is one of the notebooks you already ran, on **serverless** compute (no clusters to manage).
# MAGIC * **`eval_gate`** is a *condition task*: it reads the `eval_passed` value that `10_evaluate` publishes.
# MAGIC   The agent is only (re)deployed when the new evaluation passes the quality thresholds.
# MAGIC * **Schedule:** weekly (Monday 06:00 UTC), created **paused** so the trial isn't charged until you unpause it.
# MAGIC * **Alerts:** email to you on any failure; retries on the API-dependent ingestion tasks.
# MAGIC * Run history lands in `system.lakeflow.job_run_timeline` (Step 14 monitoring).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
JOB_NAME = "wb_demo_pipeline"
RUN_NOW = False    # True → start a full run immediately (~1–1.5 h: ingestion, parsing, index sync, 48 agent runs)

# Notebooks live next to this one; job paths are workspace paths without the /Workspace prefix
FOLDER = os.getcwd().removeprefix("/Workspace")
def nb(name: str) -> jobs.NotebookTask:
    return jobs.NotebookTask(notebook_path=f"{FOLDER}/{name}", source=jobs.Source.WORKSPACE)
print("Notebook folder:", FOLDER)

# COMMAND ----------

# MAGIC %md ## Task graph

# COMMAND ----------

def task(key, notebook, depends=(), timeout_min=60, retries=0, description=""):
    return jobs.Task(
        task_key=key, description=description, notebook_task=nb(notebook),
        depends_on=[jobs.TaskDependency(task_key=d) for d in depends],
        timeout_seconds=timeout_min * 60, max_retries=retries, min_retry_interval_millis=120_000,
    )

tasks = [
    task("ingest_indicators", "02_ingest_indicators", retries=2, timeout_min=30,
         description="World Bank Indicators API → bronze.indicators_raw (+ country / indicator metadata)"),
    task("ingest_documents", "03_ingest_documents", retries=2, timeout_min=45,
         description="World Bank Documents API → PDFs in the raw_docs volume + bronze.documents_meta"),
    task("build_silver_gold_indicators", "04a_silver_gold_indicators", ["ingest_indicators"], timeout_min=30,
         description="silver.indicator_values → gold.country_year_facts"),
    task("build_silver_gold_documents", "04b_silver_gold_documents", ["ingest_documents"], timeout_min=60,
         description="Parse new PDFs → silver.doc_text → gold.doc_chunks (MERGE, CDF)"),
    task("sync_vector_index", "06_vector_index", ["build_silver_gold_documents"], timeout_min=60,
         description="Incremental Delta Sync of the Vector Search index"),
    task("run_evaluation", "10_evaluate", ["build_silver_gold_indicators", "sync_vector_index"], timeout_min=120,
         description="MLflow GenAI evaluation v1 vs @candidate; publishes eval_passed"),
    jobs.Task(
        task_key="eval_gate", description="Deploy only if the quality gate passed",
        depends_on=[jobs.TaskDependency(task_key="run_evaluation")],
        condition_task=jobs.ConditionTask(op=jobs.ConditionTaskOp.EQUAL_TO,
                                          left="{{tasks.run_evaluation.values.eval_passed}}", right="true"),
    ),
    jobs.Task(
        task_key="deploy_agent", description="agents.deploy(@champion) + AI Gateway + smoke test",
        depends_on=[jobs.TaskDependency(task_key="eval_gate", outcome="true")],
        notebook_task=nb("11_deploy"), timeout_seconds=60 * 60,
    ),
]

settings = dict(
    name=JOB_NAME,
    description="World Bank AI assistant: ingest → lakehouse → vector index → evaluate → deploy (if quality gate passes)",
    tasks=tasks,
    schedule=jobs.CronSchedule(quartz_cron_expression="0 0 6 ? * MON", timezone_id="UTC",
                               pause_status=jobs.PauseStatus.PAUSED),
    email_notifications=jobs.JobEmailNotifications(on_failure=[me], no_alert_for_skipped_runs=True),
    max_concurrent_runs=1,
    timeout_seconds=4 * 60 * 60,
    tags={"project": "wb_demo"},
)

# COMMAND ----------

# MAGIC %md ## Create or update

# COMMAND ----------

existing = [j for j in w.jobs.list(name=JOB_NAME)]
if existing:
    job_id = existing[0].job_id
    w.jobs.reset(job_id, new_settings=jobs.JobSettings(**settings))
    print(f"Updated job {JOB_NAME} ({job_id})")
else:
    job_id = w.jobs.create(**settings).job_id
    print(f"Created job {JOB_NAME} ({job_id})")

host = w.config.host.rstrip("/")
html = f'<h3>✅ <a href="{host}/jobs/{job_id}" target="_blank">Open {JOB_NAME}</a> - Tasks tab shows the DAG</h3>'

if RUN_NOW:
    run = w.jobs.run_now(job_id)
    html += f'<p>▶ Run started: <a href="{host}/jobs/{job_id}/runs/{run.run_id}" target="_blank">run {run.run_id}</a></p>'

displayHTML(html + "<p>Schedule: Mondays 06:00 UTC - <b>paused</b>. Unpause it on the job page when you want weekly runs.</p>")
log_event("job", "created_or_updated", job_id=job_id, tasks=[t.task_key for t in tasks], run_now=RUN_NOW)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo tips
# MAGIC * **Show the DAG:** job page → *Tasks*. Click `eval_gate` to show the condition `eval_passed == true`.
# MAGIC * **Show a run:** *Runs* tab → the Gantt view shows task timings; each task links to its notebook output.
# MAGIC * **Re-run just one part:** open a finished run → **Repair run** → select a task (e.g. `run_evaluation`) -
# MAGIC   only that task and its downstream tasks run again.
# MAGIC * Before the demo, start one full run (RUN_NOW=True or *Run now*) so there is a green run to show.
