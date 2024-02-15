from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore


document_store = InMemoryDocumentStore()
retriever = EmbeddingRetriever(
    document_store=document_store,
    # embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_model="intfloat/multilingual-e5-large",
    use_gpu=True,
    scale_score=False,
)
