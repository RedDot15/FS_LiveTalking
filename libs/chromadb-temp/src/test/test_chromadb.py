from __future__ import annotations

import numpy as np

from chromadb_temp import ChromaDB
from chromadb_temp import ChromaDBInput
from chromadb_temp import ChromaDBSetting

chroma_settings = ChromaDBSetting(
    host="localhost",
    port=8000,
    model_name="gemini-2.0-flash"
)
chromadb = ChromaDB(chromadb_setting=chroma_settings)

# --- sample documents ---
documents = [
    "Mars, often called the 'Red Planet', has captured the imagination of scientists and space enthusiasts alike.",
    "The Hubble Space Telescope has provided us with breathtaking images of distant galaxies and nebulae.",
    "The concept of a black hole, where gravity is so strong that nothing can escape it, was first theorized by Albert Einstein's theory of general relativity.",
    "The Renaissance was a pivotal period in history that saw a flourishing of art, science, and culture in Europe.",
    "The Industrial Revolution marked a significant shift in human society, leading to urbanization and technological advancements.",
]

metadatas = [
    {'source': "Space"},
    {'source': "Space"},
    {'source': "Space"},
    {'source': "History"},
    {'source': "History"}
]
ids = [str(i) for i in range(1, len(documents) + 1)]

embedding_dim = 768
fake_vectors = np.random.rand(len(documents), embedding_dim).tolist()

collection_status = False
while not collection_status:
    try:
        chromadb.store_vector(
            document_collections='test001',
            vectors=fake_vectors,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
        collection_status = True
    except Exception as e:
        raise e

query_vec = np.random.rand(1, embedding_dim).tolist()

results = chromadb.process(
    inputs=ChromaDBInput(
        query=query_vec,
        topk=3,
        document_collections='test001'
    )   
)

print("\nTop 3 matches:")
for i, doc in enumerate(results.results):
    print(f"{i+1}. {doc}")