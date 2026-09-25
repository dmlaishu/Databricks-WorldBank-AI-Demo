import os
import time
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

from src.supervisor import route_question
from src.rag_agent import retrieve_context

CATALOG = os.getenv("CATALOG", "world_bank_ai")
WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID", "")
cfg = Config()

st.set_page_config(page_title="World Bank Development Intelligence", page_icon="🌍", layout="wide")
st.title("🌍 World Bank Development Intelligence")
st.caption("Databricks Lakehouse • Delta Lake • Unity Catalog • SQL • AI Search/RAG • MLflow")

INDICATORS = {
    "GDP growth": "NY.GDP.MKTP.KD.ZG",
    "Population": "SP.POP.TOTL",
    "Unemployment": "SL.UEM.TOTL.ZS",
    "Inflation": "FP.CPI.TOTL.ZG",
}
COUNTRIES = {"India": "IND", "United States": "USA", "China": "CHN"}

def warehouse_http_path():
    explicit = os.getenv("DATABRICKS_HTTP_PATH")
    if explicit:
        return explicit
    if WAREHOUSE_ID:
        return f"/sql/1.0/warehouses/{WAREHOUSE_ID}"
    raise RuntimeError("Add a SQL warehouse resource to the app with key sql-warehouse.")

def connection():
    host = cfg.host.replace("https://", "")
    return sql.connect(
        server_hostname=host,
        http_path=warehouse_http_path(),
        credentials_provider=lambda: cfg.authenticate,
    )

def structured_answer(country_code="IND", indicator_code="NY.GDP.MKTP.KD.ZG", limit=5):
    query = f"""
      SELECT country_name, indicator_name, year, value
      FROM {CATALOG}.gold.country_indicators
      WHERE country_code = ?
        AND indicator_code = ?
      ORDER BY year DESC
      LIMIT {int(limit)}
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [country_code, indicator_code])
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
    return pd.DataFrame(rows, columns=cols)

def log_event(question, route, latency_ms, status, error=None):
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {CATALOG}.monitoring.app_logs
                    VALUES (current_timestamp(), ?, ?, ?, ?, ?, ?)""",
                    [str(time.time_ns()), question, route, float(latency_ms), status, error],
                )
    except Exception:
        pass

with st.sidebar:
    st.header("Demo controls")
    country_name = st.selectbox("Country", list(COUNTRIES))
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    st.markdown("**Suggested questions**")
    st.caption("Show India's GDP growth over the last five years.")
    st.caption("What does the World Bank say about India's economic outlook?")
    st.caption("Compare India's recent GDP performance with the World Bank's economic outlook.")

question = st.chat_input("Ask a World Bank development question...")

if question:
    with st.chat_message("user"):
        st.write(question)

    decision = route_question(question)
    started = time.perf_counter()

    with st.chat_message("assistant"):
        st.caption(f"Supervisor route: **{decision.route.upper()}** — {decision.reason}")
        try:
            if decision.route in ("sql", "hybrid"):
                df = structured_answer(COUNTRIES[country_name], INDICATORS[indicator_name])
                st.subheader("Structured evidence — Databricks SQL")
                if df.empty:
                    st.warning("No structured observations were returned.")
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    chart = df.sort_values("year").set_index("year")["value"]
                    st.line_chart(chart)

            if decision.route in ("rag", "hybrid"):
                st.subheader("Unstructured evidence — Databricks AI Search")
                try:
                    docs = retrieve_context(question, k=5)
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Source {i}: {doc.get('title', 'World Bank report')}**")
                        st.write(doc.get("chunk_text", ""))
                        if doc.get("source_url"):
                            st.caption(doc["source_url"])
                except Exception as rag_error:
                    st.info(
                        "RAG is ready in the code but the workspace AI Search resource/index "
                        f"must be configured before retrieval can run. ({rag_error})"
                    )

            if decision.route == "hybrid":
                st.success(
                    "Hybrid path completed: the supervisor used quantitative Lakehouse data "
                    "and the document-retrieval path. Connect the serving LLM endpoint to synthesize "
                    "these two evidence blocks into one generated response."
                )
            elif decision.route == "sql":
                st.success("Answer grounded in the governed Gold Delta table.")
            else:
                st.success("Answer path grounded in World Bank document retrieval.")

            latency = (time.perf_counter() - started) * 1000
            log_event(question, decision.route, latency, "success")
            st.caption(f"Latency: {latency:.0f} ms")
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            log_event(question, decision.route, latency, "error", str(exc))
            st.error(f"Request failed: {exc}")
