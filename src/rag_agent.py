import os
from databricks.ai_search.client import AISearchClient

VECTOR_INDEX = os.getenv(
    "VECTOR_INDEX",
    "world_bank_ai.silver.world_bank_document_index",
)

_client = None
_index = None

def _get_index():
    global _client, _index
    if _index is None:
        _client = AISearchClient()
        _index = _client.get_index(index_name=VECTOR_INDEX)
    return _index

def retrieve_context(question: str, k: int = 5):
    """Retrieve governed World Bank report chunks from Databricks AI Search."""
    index = _get_index()
    results = index.similarity_search(
        query_text=question,
        columns=["document_id", "title", "source_url", "chunk_id", "chunk_text"],
        num_results=k,
        query_type="hybrid",
    )

    columns = [c["name"] for c in results["manifest"]["columns"]]
    docs = []
    for row in results["result"]["data_array"]:
        item = dict(zip(columns, row[:len(columns)]))
        docs.append(item)
    return docs
