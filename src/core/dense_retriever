import os

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def retriever(query: str, law_name: str, limit: int = 20):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client = QdrantClient("localhost", port=6333)
    model = SentenceTransformer(
        "intfloat/multilingual-e5-small", token=token, device=device
    )

    query_embeddings = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name=law_name, query=query_embeddings, limit=limit
    ).points

    return [(str(r.payload["chunk_id"]), r.score) for r in results]
