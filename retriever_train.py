import os
from start import retriever

retriever.train(
    save_dir=os.path.join(os.getcwd(), "trained_models")
)