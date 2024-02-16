import os
import logging
import pandas as pd
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


retriever = EmbeddingRetriever(
    # pooling_strategy="reduce_mean",
    document_store=InMemoryDocumentStore(),
    # model_format = "sentence_transformers",
    # embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    # embedding_model="intfloat/multilingual-e5-large",
    embedding_model=(os.path.join(os.getcwd(), "models", "all_sys_paraphrase.transformers")),
    use_gpu=True,
    # scale_score=False,
    scale_score=True,
)


df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons.csv"), sep="\t")

print(df)
questions = list(df["query"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"query": "content"})
df.to_csv(os.path.join(os.getcwd(), "data", "etalons_with_embedding.csv"), sep="\t")
