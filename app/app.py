"""World Bank AI Assistant - Databricks App (Streamlit).

Calls the Model Serving endpoint of the multi-agent system and shows, for every answer:
the answer, the agent path taken, the sources (report passages with links, SQL tools with arguments),
and how long it took. Every request is written to wb_demo.ops.app_logs (application log, Step 14)
and to the app's own logs (Apps → wb-demo-chat → Logs).
"""
import json
import logging
import os
import time
import uuid

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

ENDPOINT = os.environ["SERVING_ENDPOINT"]            # injected from the app's serving-endpoint resource
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID")
LOG_TABLE = os.environ.get("LOG_TABLE", "wb_demo.ops.app_logs")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wb-demo-chat")

EXAMPLES = {
    "🔢 Numeric": "What was Kenya's GDP growth in 2019?",
    "📄 Document": "What risks does the latest Nigeria Development Update highlight?",
    "🔀 Mixed": ("How has India's GDP growth compared to Bangladesh since 2010, and what does the World Bank say "
                "are the main risks to South Asia's growth?"),
}


@st.cache_resource
def workspace() -> WorkspaceClient:
    return WorkspaceClient()   # authenticates as the app's service principal


def current_user() -> str:
    try:
        return st.context.headers.get("X-Forwarded-Email") or st.context.headers.get("X-Forwarded-Preferred-Username") or "unknown"
    except Exception:
        return "unknown"


def ask_agent(messages: list[dict]) -> dict:
    return workspace().api_client.do("POST", f"/serving-endpoints/{ENDPOINT}/invocations", body={"input": messages})


def log_request(event: str, level: str = "INFO", **details):
    """App log → stdout (Apps UI) and → Delta table in Unity Catalog (queried by the monitoring dashboard)."""
    log.log(getattr(logging, level), "%s %s", event, json.dumps(details, default=str))
    if not WAREHOUSE_ID:
        return
    try:
        workspace().statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID, wait_timeout="0s",   # fire-and-forget: never slows the chat down
            statement=f"INSERT INTO {LOG_TABLE} VALUES (current_timestamp(), :run_id, 'chat_app', :level, :event, :details)",
            parameters=[StatementParameterListItem(name="run_id", value=st.session_state.session_id),
                        StatementParameterListItem(name="level", value=level),
                        StatementParameterListItem(name="event", value=event),
                        StatementParameterListItem(name="details", value=json.dumps(details, default=str))],
        )
    except Exception as e:
        log.warning("could not write to %s: %s", LOG_TABLE, e)


def split_sources(text: str) -> str:
    """The agent appends a '**Sources**' list; the app renders sources itself from custom_outputs."""
    return text.split("\n\n**Sources**")[0]


def render_details(meta: dict):
    path = meta.get("agent_path") or "-"
    st.caption(f"🧭 **{path}**   ·   ⏱ {meta.get('latency_s', '?')} s")

    sources = meta.get("sources") or []
    docs = [s for s in sources if s.get("type") == "document"]
    tables = [s for s in sources if s.get("type") == "table"]
    if docs or tables:
        with st.expander(f"📚 Sources ({len(docs)} report passages, {len(tables)} data queries)", expanded=True):
            for s in docs:
                title = f"[{s.get('id')}] {s.get('title')} ({s.get('year')}), pp. {s.get('pages')}"
                st.markdown(f"- [{title}]({s['url']})" if s.get("url") else f"- {title}")
            for s in tables:
                st.markdown(f"- 🗄️ {s.get('label')}")

    with st.expander("🔍 How this answer was produced"):
        routing = meta.get("routing") or {}
        st.markdown("**Supervisor routing**")
        st.json({k: v for k, v in routing.items() if v})
        calls = meta.get("tool_calls") or []
        if calls:
            st.markdown("**Tool calls** (Unity Catalog functions)")
            st.dataframe([{"agent": c.get("agent"), "tool": c.get("tool"), "arguments": json.dumps(c.get("args")),
                           "rows": c.get("rows", c.get("error"))} for c in calls], width="stretch")


# ── UI ─────────────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="World Bank AI Assistant", page_icon="🌍", layout="wide")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []          # [{"role", "content", "meta"}]

with st.sidebar:
    st.header("🌍 World Bank AI Assistant")
    st.write("Ask about development indicators for 20 countries (2000 → latest) or about what World Bank "
             "reports say. A supervisor agent routes each question to a **Data Analyst** agent (SQL over "
             "governed gold tables) and/or a **Research** agent (Vector Search over World Bank reports).")
    st.markdown("**Try an example**")
    for label, q in EXAMPLES.items():
        if st.button(label, width="stretch"):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ New conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Endpoint: `{ENDPOINT}` · signed in as {current_user()}")

st.title("World Bank AI Assistant")

def give_feedback(i: int, rating: str):
    st.session_state.history[i]["rated"] = rating
    log_request("feedback", user=current_user(), question=st.session_state.history[i - 1]["content"], rating=rating,
                agent_path=(st.session_state.history[i].get("meta") or {}).get("agent_path"))

for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            render_details(msg["meta"])
            if msg.get("rated"):
                st.caption("Thanks for the feedback " + ("👍" if msg["rated"] == "up" else "👎"))
            else:
                c1, c2, _ = st.columns([1, 1, 12])
                c1.button("👍", key=f"up{i}", on_click=give_feedback, args=(i, "up"))
                c2.button("👎", key=f"down{i}", on_click=give_feedback, args=(i, "down"))

question = st.chat_input("Ask about GDP, poverty, inflation… or what the World Bank says about a country") \
           or st.session_state.pop("pending", None)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Supervisor is routing your question to the specialist agents…"):
            conversation = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history[-8:]]
            t0 = time.time()
            try:
                resp = ask_agent(conversation)
                text = split_sources(resp["output"][0]["content"][0]["text"])
                meta = {**(resp.get("custom_outputs") or {}), "latency_s": round(time.time() - t0, 1)}
                log_request("chat_request", user=current_user(), question=question, latency_s=meta["latency_s"],
                            agent_path=meta.get("agent_path"), sources=len(meta.get("sources") or []),
                            tool_calls=len(meta.get("tool_calls") or []))
            except Exception as e:
                text, meta = f"⚠️ The assistant is unavailable right now ({type(e).__name__}). Please try again.", None
                log_request("chat_error", level="ERROR", user=current_user(), question=question, error=str(e)[:500],
                            latency_s=round(time.time() - t0, 1))
        st.markdown(text)
        if meta:
            render_details(meta)
    st.session_state.history.append({"role": "assistant", "content": text, "meta": meta})
    st.rerun()   # redraw from history so the new answer gets its feedback buttons
