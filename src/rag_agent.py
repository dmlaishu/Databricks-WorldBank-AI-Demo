import os

VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "")
VECTOR_INDEX = os.getenv(
    "VECTOR_INDEX",
    "world_bank_ai.silver.world_bank_document_index",
)

def retrieve_context(question: str, k: int = 5):
    """
    Workspace integration point.

    Configure a Databricks Vector Search endpoint/index over
    world_bank_ai.silver.document_chunks, then call similarity search here.

    Keeping this adapter isolated makes the interview demo easy to explain:
    ingestion/chunking -> governed Delta table -> vector index -> retrieved
    report context -> LLM answer with source URLs.
    """
    if not VECTOR_SEARCH_ENDPOINT:
        raise RuntimeError("VECTOR_SEARCH_ENDPOINT is not configured")
    # Add workspace-specific Vector Search client call after index creation.
    raise NotImplementedError("Configure the workspace Vector Search index first")
