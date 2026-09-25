# Workspace Setup Guide

This project is now wired as a Streamlit Databricks App. The code is portable, but Databricks resources must be created/selected in your workspace.

## 1. Use Databricks Free Edition or a trial workspace

Free Edition is sufficient for learning/prototyping but has resource limits. Use serverless notebook compute.

## 2. Bring the GitHub repo into Databricks

In the workspace, create/import a Git folder from this repository, then open the project.

## 3. Run the Lakehouse notebooks

Run in this order:

1. `notebooks/01_setup.py`
2. `notebooks/02_ingest_worldbank.py`
3. `notebooks/03_bronze_to_silver.py`
4. `notebooks/04_build_gold.py`

Verify these objects in Catalog Explorer:

- `world_bank_ai.bronze.indicators_raw`
- `world_bank_ai.silver.indicators_clean`
- `world_bank_ai.gold.country_indicators`
- `world_bank_ai.monitoring.app_logs`

## 4. Verify Databricks SQL

Open SQL Editor and run `sql/analytics.sql`. This is the structured-data proof for the interview.

## 5. Load unstructured World Bank report content

Insert approved report text and its source URL into:

`world_bank_ai.bronze.documents_raw`

Then run `notebooks/05_ingest_documents.py` to create:

`world_bank_ai.silver.document_chunks`

Do not claim RAG is live until the index below is actually configured.

## 6. Create Databricks AI Search

Create an AI Search endpoint and a Delta Sync index sourced from:

`world_bank_ai.silver.document_chunks`

Use `chunk_text` as the text/embedding source and keep `document_id`, `title`, `source_url`, and `chunk_id` as metadata. Name the index:

`world_bank_ai.silver.world_bank_document_index`

The app's `src/rag_agent.py` queries this index with hybrid retrieval.

## 7. Create the Databricks App

Create a Databricks App from this source folder. Add a SQL warehouse resource with the key:

`sql-warehouse`

Grant the app service principal CAN USE on the warehouse and SELECT/USE privileges needed for the Unity Catalog objects.

The app starts with:

`streamlit run app.py`

## 8. MLflow evaluation

Run `notebooks/06_evaluation.py`. It logs the router evaluation to MLflow. Once the final LLM synthesis endpoint is connected, extend evaluation to groundedness, relevance, retrieval quality, latency and cost.

## 9. Workflows

The repository's `databricks.yml` defines the ingestion -> Silver -> Gold -> evaluation task chain. Validate/deploy it with the current Databricks CLI or author/deploy the bundle in the workspace.

## 10. Model Serving

For the final production-style demo, create/select a Model Serving endpoint for the LLM/agent and add it to the app as a managed resource with CAN QUERY. The current app deliberately does not invent an endpoint name because the available models/resources differ by workspace.

## 11. Audit and application logs

Run `sql/audit_logs.sql` where system audit tables are enabled. The app also writes request route/status/latency records into `world_bank_ai.monitoring.app_logs`.

## Demo readiness checklist

- Structured World Bank API ingestion works
- Bronze/Silver/Gold Delta tables visible
- Unity Catalog objects visible
- SQL query returns GDP data
- World Bank report chunks visible
- AI Search index online
- Streamlit app can query SQL
- RAG retrieval returns report chunks
- MLflow evaluation run visible
- Workflow visible/runnable
- App logs visible
- Audit query shown where enabled
- Model Serving endpoint connected before claiming end-to-end generated synthesis
