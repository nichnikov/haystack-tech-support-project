import os
import pandas as pd
import numpy as np
from ast import literal_eval
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
from start import document_store, retriever

pipe = FAQPipeline(retriever=retriever)

df = pd.read_csv(os.path.join(os.getcwd(), "data", "etalons_with_embedding.csv"), sep="\t")
df["embedding"] = df["embedding"].apply(lambda x: np.array(literal_eval(x)))

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

# for query_tx in list(df["content"])[:5]:
query_tx = "что за жопа"
prediction = pipe.run(query=query_tx, params={"Retriever": {"top_k": 1}})
print(query_tx)
print_answers(prediction, details="medium")