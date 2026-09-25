import os
from openai import OpenAI

ENDPOINT = os.getenv("AGENT_ENDPOINT", "")

def synthesize(question: str, structured_context: str = "", document_context: str = "") -> str:
    """Generate a grounded answer through a Databricks Model Serving endpoint."""
    if not ENDPOINT:
        parts = []
        if structured_context:
            parts.append("Structured evidence:\n" + structured_context)
        if document_context:
            parts.append("World Bank documentation:\n" + document_context)
        return "\n\n".join(parts) or "No evidence was returned."

    client = OpenAI(
        api_key=os.getenv("DATABRICKS_TOKEN", "databricks-oauth"),
        base_url=os.getenv("DATABRICKS_OPENAI_BASE_URL"),
    )
    prompt = f"""Answer the user's question using ONLY the evidence below.
If the evidence is insufficient, say what is missing. Do not invent numbers.
Question: {question}

STRUCTURED EVIDENCE
{structured_context}

UNSTRUCTURED WORLD BANK EVIDENCE
{document_context}
"""
    response = client.chat.completions.create(
        model=ENDPOINT,
        messages=[
            {"role": "system", "content": "You are a grounded World Bank development data assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
