from sentence_transformers import (
    SentenceTransformer
)

import faiss
import pickle
import json
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

with open(
    "backend/catalog.json",
    "r"
) as f:

    catalog = json.load(f)

texts = []

for item in catalog:

    text = f"""
    {item['name']}
    {item['description']}
    """

    texts.append(text)

embeddings = model.encode(texts)

embeddings = np.array(
    embeddings
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

faiss.write_index(
    index,
    "backend/faiss_index.bin"
)

with open(
    "backend/metadata.pkl",
    "wb"
) as f:

    pickle.dump(catalog, f)

print("FAISS index built successfully.")