import os
import pandas as pd

from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
from start import document_store, retriever

pipe = FAQPipeline(retriever=retriever)

df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons.csv"), sep="\t")

ext_query_txs = ["африканский носорог опасный зверь", "что за жопа", "отпишите меня от ваших рассылок", "пусть бегут неуклюжи пешеходы по лужам, а вода по асфальту рекой"]
query_txs = list(df["query"])[:5] + ext_query_txs
for query_tx in query_txs:
    prediction = pipe.run(query=query_tx, params={"Retriever": {"top_k": 1}})
    print_answers(prediction, details="medium")