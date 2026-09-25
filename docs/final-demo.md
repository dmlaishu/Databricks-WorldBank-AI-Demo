# Final Demo Walkthrough

## Demo 1 — Structured data
Ask: **Show India's GDP growth over the last five years.**

The supervisor chooses the SQL path. Databricks SQL reads the governed Gold Delta table and the app renders the observations and trend chart.

## Demo 2 — Unstructured data
Ask: **Explain what the World Bank GDP growth indicator measures.**

The supervisor chooses the RAG path. Databricks AI Search retrieves chunks created from World Bank indicator metadata, including the source note and source organization.

## Demo 3 — Multi-agent / hybrid
Ask: **Compare India's recent GDP growth with how the World Bank defines and sources this indicator.**

The supervisor chooses the hybrid path. It combines SQL evidence from the Gold table with unstructured World Bank metadata retrieved through AI Search. When AGENT_ENDPOINT is configured, the Model Serving LLM synthesizes one grounded response.

## What to show on screen
1. Catalog Explorer: bronze, silver, gold, monitoring and evaluation schemas.
2. Bronze: raw API data and raw World Bank metadata.
3. Silver: cleaned indicator rows and document chunks.
4. Gold: country_indicators.
5. SQL Editor: run sql/analytics.sql.
6. AI Search: show world_bank_document_index.
7. App: run the three questions above.
8. MLflow: show router evaluation run/traces.
9. Workflows: show ingestion -> transformation -> Gold -> evaluation.
10. Query monitoring.app_logs and system.access.audit where available.

## 45-second interview explanation
"I built a multi-agent development intelligence chatbot on Databricks using World Bank data. Structured indicator data is ingested through the World Bank API and stored in Bronze, cleaned in Silver, and exposed through Gold Delta tables. I also ingest World Bank indicator documentation as unstructured text and index the chunks with Databricks AI Search. A supervisor routes quantitative questions to Databricks SQL, documentation questions to RAG, and hybrid questions to both. Unity Catalog governs the data, MLflow handles evaluation and tracing, Workflows automate the pipeline, Model Serving provides the LLM endpoint, and application plus audit logs give me observability and governance."
