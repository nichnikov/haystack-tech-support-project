import os
import logging
import pandas as pd
from haystack.nodes import (EmbeddingRetriever, 
                            FARMReader)
from haystack.document_stores import InMemoryDocumentStore

from haystack.pipelines import (FAQPipeline, 
                                ExtractiveQAPipeline,
                                Pipeline)
from haystack.utils import print_answers
from start import document_store, retriever


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

emb_model = (os.path.join(os.getcwd(), "models", "all_sys_paraphrase.transformers"))
#emb_model="sentence-transformers/all-MiniLM-L6-v2",
# emb_model="intfloat/multilingual-e5-large"

retriever = EmbeddingRetriever(
    # pooling_strategy="reduce_mean",
    document_store=InMemoryDocumentStore(),
    # model_format = "sentence_transformers",
    # embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    # embedding_model="intfloat/multilingual-e5-large",
    embedding_model=emb_model,
    use_gpu=True,
    # scale_score=False,
    scale_score=True,
)

df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons.csv"), sep="\t")
print(df)

questions = list(df["query"].values)
# print("embed_queries:", retriever.embed_queries(queries=questions))
df["embedding"] = retriever.embed_queries(queries=questions).tolist()

df = df.rename(columns={"query": "content"})

document_store = InMemoryDocumentStore()
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
reader = FARMReader(model_name_or_path=emb_model, use_gpu=True)
sh_retriever = EmbeddingRetriever(
    # pooling_strategy="reduce_mean",
    document_store=document_store,
    embedding_model=emb_model,
    use_gpu=True,
    scale_score=True,
)

# pipe = FAQPipeline(retriever=sh_retriever)
pipe = Pipeline()
pipe.add_node(component=sh_retriever, name="Retriever", inputs=["Query"])


ext_query_txs = ["африканский носорог опасный зверь", "москва - столица санкт-петербурга", "пусть бегут неуклюжи пешеходы по лужам, а вода по асфальту рекой"]

query_txs = list(df["content"])[:5] + ext_query_txs
for query_tx in query_txs:
    prediction = pipe.run(query=query_tx, params={"Retriever": {"top_k": 10}})
    print_answers(prediction, details="medium")
