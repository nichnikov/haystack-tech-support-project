import os
import logging
import pandas as pd
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore

from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
# from start import document_store, retriever


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# working:
# emb_model = "sentence-transformers/LaBSE" # working
# emb_model="sentence-transformers/all-MiniLM-L6-v2" # working only English

# emb_model = "sentence-transformers/distiluse-base-multilingual-cased-v1" # 50/50
# emb_model = "sentence-transformers/stsb-xlm-r-multilingual" # not working
emb_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # not working
# emb_model = "sentence-transformers/distiluse-base-multilingual-cased" # not working
# emb_model = (os.path.join(os.getcwd(), "models", "all_sys_paraphrase.transformers")) # not working
# emb_model="intfloat/multilingual-e5-large" # not working

document_store = InMemoryDocumentStore()

retriever = EmbeddingRetriever(
    document_store=document_store,
    model_format = "sentence_transformers",
    embedding_model=emb_model,
    use_gpu=True,
    scale_score=False,
    emb_extraction_layer=-1
)

df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons.csv"), sep="\t")
print(df)

df["question"] = df["question"].apply(lambda x: x.strip())
questions = list(df["question"].values)
# print("embed_queries:", retriever.embed_queries(queries=questions))
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

pipe = FAQPipeline(retriever=retriever)


ext_query_txs = ["я закончил курс и не получил диплом", 
                 "отпишите меня от ваших ебучих рассылок", 
                 "африканский носорог опасный зверь", 
                 "москва - столица санкт-петербурга", 
                 "пусть бегут неуклюжи пешеходы по лужам, а вода по асфальту рекой", 
                 "там вдали за рекой",
                 "электрификация всей страны наша главная задача"]


query_txs = ext_query_txs + list(df["content"])[:10]
for query_tx in query_txs:
    prediction = pipe.run(query=query_tx, params={"Retriever": {"top_k": 1}})
    print_answers(prediction, details="medium")
