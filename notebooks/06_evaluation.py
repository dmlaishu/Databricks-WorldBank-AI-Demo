# Databricks notebook source
import mlflow
import pandas as pd

mlflow.set_experiment("/Shared/world-bank-ai-evaluation")

eval_data = pd.DataFrame([
    {
        "question": "Show India's GDP growth over the last five years.",
        "expected_route": "sql",
    },
    {
        "question": "What does the World Bank say about India's economic outlook?",
        "expected_route": "rag",
    },
    {
        "question": "Compare India's recent GDP performance with the World Bank's economic outlook.",
        "expected_route": "hybrid",
    },
])

def expected_router(question: str) -> str:
    q = question.lower()
    has_metric = any(x in q for x in ["gdp", "inflation", "population", "unemployment"])
    has_report = any(x in q for x in ["outlook", "report", "say about", "explain"])
    if has_metric and has_report:
        return "hybrid"
    if has_report:
        return "rag"
    return "sql"

eval_data["predicted_route"] = eval_data["question"].map(expected_router)
eval_data["route_correct"] = eval_data["expected_route"] == eval_data["predicted_route"]

with mlflow.start_run(run_name="router-evaluation"):
    accuracy = float(eval_data["route_correct"].mean())
    mlflow.log_metric("route_accuracy", accuracy)
    mlflow.log_table(eval_data, "router_eval.json")

display(eval_data)
