import os
import pandas as pd
import numpy as np

from ast import literal_eval
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore


document_store = InMemoryDocumentStore()
df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons_with_embedding.csv"), sep="\t")
df["embedding"] = df["embedding"].apply(lambda x: np.array(literal_eval(x)))

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

retriever = EmbeddingRetriever(
    # pooling_strategy="reduce_mean",
    document_store=document_store,
    # model_format = "sentence_transformers",
    # embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    # embedding_model="intfloat/multilingual-e5-large",
    embedding_model=(os.path.join(os.getcwd(), "models", "all_sys_paraphrase.transformers")),
    use_gpu=True,
    # scale_score=False,
    scale_score=True,
)
