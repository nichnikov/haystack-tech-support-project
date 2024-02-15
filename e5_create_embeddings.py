import os
import logging
import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from start import document_store, retriever

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons.csv"), sep="\t")

print(df)
questions = list(df["query"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"query": "content"})
df.to_csv(os.path.join(os.getcwd(), "data", "etalons_with_embedding.csv"), sep="\t")

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)
