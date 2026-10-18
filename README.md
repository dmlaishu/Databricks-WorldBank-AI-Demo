# World Bank AI Assistant on Databricks

A multi-agent assistant that answers questions about global development using World Bank data. It works with two kinds of
data: indicator time series from the World Bank Indicators API (GDP, growth, inflation, poverty, life expectancy and so on)
and around 60 World Bank reports from the Documents & Reports API.

Example question:

> How has India's GDP growth compared to Bangladesh since 2010, and what does the World Bank say are the main risks to
> South Asia's growth?

The first half is answered with SQL over the indicator tables, the second half by searching the reports. A supervisor
agent splits the question, sends each part to the right specialist, and merges the answers with citations.

The project is built end to end on Databricks: ingestion, a Delta lakehouse governed by Unity Catalog, agent tools as
Unity Catalog functions, MLflow tracing and evaluation, Model Serving, a Databricks App for the chat UI, a Lakeflow Job
for orchestration, and monitoring and audit views on top of system tables.

## Architecture

```
 Sources                    Lakehouse (Delta + Unity Catalog, catalog wb_demo)            AI layer                    Consumers
 World Bank Indicators ──▶ bronze ─▶ silver ─▶ gold.country_year_facts ──────────┐
                                                                                  ├─▶ 5 UC functions (tools)  ─▶  Chat app
 World Bank Documents  ──▶ bronze volume (PDFs) ─▶ silver.doc_text ─▶             │   Supervisor agent              SQL dashboards
                           gold.doc_chunks ─▶ Vector Search index ───────────────┘    ├ Data Analyst agent
                                                                                       └ Research agent
 Lakeflow Job: ingest → build → sync index → evaluate → deploy (only if the evaluation passes)
 Monitoring: ops.app_logs, MLflow traces, system tables     Audit: system.access.audit, lineage
```

Schemas: `bronze` (raw), `silver` (cleaned), `gold` (consumption), `ai` (functions, index, model), `ops` (logs,
evaluation results, monitoring and audit views).

## How the agent works

- **Supervisor** - reads the conversation, splits the question into a data part and a research part, and handles
  follow-ups like "what about Pakistan?".
- **Data Analyst agent** - calls four Unity Catalog SQL functions (`list_indicators`, `get_indicator`,
  `compare_countries`, `get_country_profile`). It only reports numbers returned by a tool.
- **Research agent** - calls `search_reports`, which wraps a hybrid (semantic + keyword) Vector Search query, and cites
  the passages it uses.
- The source list at the end of each answer is built from the passages that were actually retrieved and cited, not
  written by the model.
- The agent implements MLflow's `ResponsesAgent` interface and every step is traced (routing, each agent, each tool
  call, each LLM call).

Tools are Unity Catalog functions rather than free-form SQL so that the agent runs under the same permissions and audit
trail as any other user, and can only do what the functions allow.

## Evaluation

24 questions in `ops.eval_set`: numeric questions with ground truth taken from the gold table, document questions with
facts taken from report abstracts, mixed questions, and out-of-scope questions (including a prompt-injection attempt).
They are scored with MLflow judges (correctness, groundedness, retrieval sufficiency, guidelines, relevance, safety), a
custom `cites_source` judge, and code checks for routing and latency.

| Metric | v1 (minimal prompts) | v2 (released) |
|---|---|---|
| Correctness | 0.59 | 0.82 |
| Cites sources | 0.42 | 0.75 |
| Retrieval sufficiency | 0.38 | 0.63 |
| Correct routing | 0.96 | 0.96 |
| Safety | 1.00 | 1.00 |

Evaluation surfaced real issues that were then fixed: rate-limit failures on the model endpoint, a system-prompt leak,
country names that didn't match the World Bank's spelling ("Vietnam" vs "Viet Nam"), `IND` also matching Indonesia,
"latest year" questions answered with an old year, and forecast questions routed to the data agent. A version is only
promoted to the `@champion` alias, and deployed, if it passes the quality thresholds.

Groundedness dropped from v1 to v2. That judge compares the whole answer with the retrieved passages, and v2 answers
include exact figures from the SQL tools that are not in any passage, so it understates the real grounding.

## Governance and audit

| Group | bronze / silver | gold | ai functions | vector index | ops |
|---|---|---|---|---|---|
| `wb_analysts` | no access | read | execute | no access | no access |
| `wb_ai_app` | no access | read | read + execute | read | read + write |

The chat app runs as its own service principal with query access to the endpoint, use of the SQL warehouse, and write
access to the log table. Tested with an analyst account: reading gold works, reading bronze fails with
`INSUFFICIENT_PERMISSIONS`, and `search_reports` fails with a 403 on the vector index. Those attempts show up in
`ops.v_audit_denied`, built on `system.access.audit`.

## Repository layout

```
notebooks/   Databricks notebooks, run in order (source format, import into a workspace)
  00_config                    shared settings and the app logger
  00_verify_workspace          pre-flight checks for a new workspace
  01_setup_unity_catalog       catalog, schemas, volume, log table, grants
  02_ingest_indicators         Indicators API -> bronze
  03_ingest_documents          Documents API -> PDFs in a volume + metadata
  04a_silver_gold_indicators   silver indicator values -> gold.country_year_facts
  04b_silver_gold_documents    PDF text -> chunks (gold.doc_chunks, change data feed on)
  05_sql_dashboard             snapshot view + AI/BI dashboard
  06_vector_index              Vector Search endpoint and Delta Sync index
  07_uc_tools                  the five Unity Catalog functions used as tools
  08_agent                     the multi-agent system (written from src/worldbank_agent.py)
  09_log_register              log with MLflow, register in Unity Catalog
  10_evaluate                  evaluation set, judges, v1 vs v2, quality gate
  11_deploy                    Model Serving deployment
  12_app                       chat app deployment (written from app/)
  13_create_job                Lakeflow Job with the evaluation gate
  14_monitoring                monitoring views and dashboard
  15_audit                     groups, grants, audit and lineage views
src/         agent source and the script that builds notebooks 08 and 12
app/         Streamlit chat app (Databricks App)
sql/         analytics queries for the SQL editor
```

`notebooks/08_agent.py` and `notebooks/12_app.py` embed the code from `src/` and `app/`. After changing those files,
run `python src/build_agent_notebook.py` to regenerate them.

## Running it

Requirements: a Databricks workspace with Unity Catalog, serverless compute, Foundation Model APIs
(`databricks-meta-llama-3-3-70b-instruct` and `databricks-gte-large-en`), Vector Search and Databricks Apps, plus
permission to create a catalog. System tables are needed for the monitoring and audit views.

1. Import the `notebooks/` folder into the workspace (Workspace > Import, or a Git folder).
2. Check `00_config` - catalog name, model endpoint names, countries and indicators.
3. Run the notebooks in order on serverless compute. `06_vector_index` takes a while on the first run because every
   chunk is embedded; `10_evaluate` takes 20-40 minutes.
4. Create the `wb_analysts` and `wb_ai_app` groups as account-level groups (Settings > Identity and access > Groups),
   then run `15_audit` to apply the grants.

## Notes and limitations

- Data covers 20 countries (2000 to the latest published year) and 14 indicators. There are no World Bank country
  reports for the US, and only one each for Mexico, Colombia and Vietnam.
- Chat history in the app lasts for the browser session; every request is also stored in `ops.app_logs`.
- AI Gateway inference tables were not available on the workspace this was built on, so request logging relies on the
  app log table and MLflow traces.
- The evaluation set is small and the document facts were generated from abstracts, so they would benefit from review.
- Typical answer time is around 10 seconds, since a question can involve several model and SQL calls.
