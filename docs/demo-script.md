# Interview Demo Script

Start with the architecture: "This application combines structured World Bank indicators and unstructured World Bank reports in one governed Databricks Lakehouse."

Show Unity Catalog and explain Bronze, Silver and Gold.

Run the SQL demo: "Show India's GDP growth over the last five years." Explain that the numerical values come from the Gold Delta table through Databricks SQL rather than model memory.

Run the RAG demo: "What does the World Bank say about India's economic outlook?" Show the retrieved document chunks/source links and explain grounding.

Run the hero question: "Compare India's recent GDP performance with the World Bank's economic outlook." Explain that the supervisor invokes both the SQL and RAG paths and combines their evidence.

Open MLflow and show the trace/evaluation run. Then open the Workflow to show automated ingestion/transformation/evaluation.

Finish with application logs and audit logs: "This gives me both operational observability and governance/auditability."

## Four questions to know for every component

1. What is it?
2. Why did I use it?
3. Where is it used in this project?
4. How does data flow through it?
