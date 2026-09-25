from dataclasses import dataclass
from typing import Literal

Route = Literal["sql", "rag", "hybrid"]

@dataclass
class RouteDecision:
    route: Route
    reason: str

def route_question(question: str) -> RouteDecision:
    q = question.lower()
    metric_terms = ["gdp", "inflation", "population", "unemployment", "growth", "percent", "year"]
    document_terms = ["outlook", "report", "according to", "world bank say", "explain", "why"]

    needs_sql = any(term in q for term in metric_terms)
    needs_rag = any(term in q for term in document_terms)

    if needs_sql and needs_rag:
        return RouteDecision("hybrid", "Question needs both quantitative data and report context.")
    if needs_rag:
        return RouteDecision("rag", "Question asks for document-grounded context.")
    return RouteDecision("sql", "Question is primarily quantitative/structured.")

def combine_answers(sql_answer: str | None, rag_answer: str | None) -> str:
    sections = []
    if sql_answer:
        sections.append("### Data evidence\n" + sql_answer)
    if rag_answer:
        sections.append("### World Bank report context\n" + rag_answer)
    return "\n\n".join(sections)
