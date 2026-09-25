# Architecture

## 1. Ingestion
Structured indicators are pulled from the World Bank Indicators API. Unstructured World Bank report text is stored separately.

## 2. Lakehouse
Raw records land in Bronze Delta tables. Validation, casting and deduplication produce Silver tables. Gold tables expose analytics-ready indicators.

## 3. Governance
Unity Catalog organizes the catalog/schemas and provides the governance boundary for data and AI assets.

## 4. Agent layer
The supervisor classifies a request as SQL, RAG or hybrid. SQL answers come from Gold tables. RAG answers come from retrieved report chunks. Hybrid questions use both.

## 5. MLflow and evaluation
MLflow captures experiments/traces and evaluation artifacts. The starter evaluation notebook tests routing; expand it with answer relevance, groundedness, retrieval quality, latency and cost once the serving endpoint is configured.

## 6. Serving and app
Deploy the finalized agent behind Databricks Model Serving and connect it to a Databricks App chat UI.

## 7. Operations
Databricks Workflows runs ingestion -> transformation -> Gold -> evaluation. Application logs capture request IDs, routes, latency and failures. System audit tables provide governance evidence where enabled.
