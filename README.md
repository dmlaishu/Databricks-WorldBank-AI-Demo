# World Bank Development Intelligence — Databricks Multi-Agent Demo

A demo-ready Databricks Lakehouse + GenAI project that combines structured World Bank indicators with unstructured development reports. A supervisor routes numerical questions to a SQL agent and document questions to a RAG agent, then combines both for hybrid analysis.

## What this demonstrates

- Databricks Workspace — notebooks and development
- Databricks SQL — governed analytical queries
- Lakehouse + Delta Lake — Bronze/Silver/Gold architecture
- Unity Catalog — catalog/schema governance and lineage
- MLflow — tracing and evaluation
- Workflows — scheduled ingestion, transformation and evaluation
- Model Serving — production endpoint target for the agent
- Evaluation — correctness, groundedness, relevance and latency
- Audit Logs — access/activity query examples
- Application/System Logs — request, route, latency and error logging
- Structured data — World Bank Indicators API
- Unstructured data — World Bank reports/document chunks
- Multi-agent AI — supervisor + SQL agent + RAG agent

## Architecture

```
World Bank Indicators API -----> Bronze -----> Silver -----> Gold Delta tables
                                                          |
                                                          v
                                                     SQL Agent
                                                          |
User -> Databricks App -> Supervisor ---------------------+----> Final answer
                                                          |
World Bank reports -> document chunks -> Vector Search -> RAG Agent
                                                          |
                                                          v
                                                MLflow trace + evaluation

Governance: Unity Catalog
Automation: Databricks Workflows
Serving: Databricks Model Serving
Observability: MLflow + application logs + audit logs
```

## Demo questions

1. **Structured:** "Show India's GDP growth over the last five years."
2. **Unstructured:** "What does the World Bank say about India's economic outlook?"
3. **Hybrid / hero demo:** "Compare India's recent GDP performance with the World Bank's economic outlook."

## Repository layout

```
notebooks/
  01_setup.py
  02_ingest_worldbank.py
  03_bronze_to_silver.py
  04_build_gold.py
  05_ingest_documents.py
  06_evaluation.py
src/
  supervisor.py
  sql_agent.py
  rag_agent.py
  logging_utils.py
sql/
  analytics.sql
  audit_logs.sql
docs/
  architecture.md
  demo-script.md
databricks.yml
app.yaml
requirements.txt
```

## Data model

Catalog: `world_bank_ai`

Schemas:
- `bronze` — raw API/document data
- `silver` — validated and standardized records/chunks
- `gold` — business-ready indicator tables
- `evaluation` — evaluation datasets/results
- `monitoring` — application telemetry

Curated indicators:
- `NY.GDP.MKTP.KD.ZG` — GDP growth (annual %)
- `SP.POP.TOTL` — Population
- `SL.UEM.TOTL.ZS` — Unemployment
- `FP.CPI.TOTL.ZG` — Inflation

Initial countries: India, United States and China.

## Quick start

1. Import/clone this repo into a Databricks Workspace.
2. Run notebooks 01–04 in order to create the governed Lakehouse tables.
3. Add World Bank report text/PDF extraction output and run notebook 05.
4. Configure a Databricks Vector Search index over `silver.document_chunks`.
5. Configure the SQL and RAG agents in `src/`.
6. Enable MLflow tracing and run notebook 06 for evaluation.
7. Deploy the agent through Model Serving and connect the Databricks App.
8. Configure the Workflow from `databricks.yml`.
9. Use `sql/audit_logs.sql` and the monitoring table during the demo.

## Important

This repository is intentionally configuration-driven. Workspace-specific resources such as SQL warehouse IDs, serving endpoint names, Vector Search endpoint/index names and model endpoint names must be supplied for your Databricks workspace. Do not commit PATs, client secrets or other credentials.

## Interview explanation

"I use the Databricks Lakehouse as the central data and AI foundation. Raw World Bank data lands in Bronze Delta tables, validated data moves to Silver, and business-ready metrics are exposed through Gold. Unity Catalog governs the assets. A supervisor routes quantitative questions to a SQL agent and report-based questions to a RAG agent. Hybrid questions use both. MLflow traces and evaluates the AI workflow, Workflows automate the pipeline, Model Serving exposes the agent, and audit/application logs provide operational visibility."
