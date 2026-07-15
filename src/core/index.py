import json
import os
import re

import pandas as pd
import pdfplumber
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def chunking(file_name: str, law_name: str):
    texts = []
    with pdfplumber.open(file_name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            texts.append(text)

    law = "".join(texts)

    parts = re.split("Madde", law)

    chunks = []
    for i, chunk in enumerate(parts):
        chunk = chunk.strip()
        chunks.append(
            {
                "chunk_id": f"{law_name}_{i+1}",
                "text": chunk,
                "metadata": {"madde_no": i + 1, "char_count": len(chunk)},
            }
        )

    with open(
        os.getcwd() + f"/data/{law_name}/chunks/chunks.jsonl", "w", encoding="utf-8"
    ) as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def indexing(law_name: str):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    chunks_path = os.getcwd() + f"/data/{law_name}/chunks/chunks.jsonl"

    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name, token=hf_token, device=device)

    client = QdrantClient("localhost", port=6333)

    if not client.collection_exists(law_name):
        client.create_collection(
            collection_name=law_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    chunks = pd.read_json(chunks_path, lines=True)
    chunk_ids = chunks["chunk_id"].tolist()
    texts = chunks["text"].tolist()
    metadatas = chunks["metadata"].tolist()

    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    points = [
        PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={"chunk_id": chunk_id, "text": text_val, **metadata_val},
        )
        for i, (chunk_id, text_val, metadata_val, emb) in enumerate(
            zip(chunk_ids, texts, metadatas, embeddings)
        )
    ]
    client.upsert(collection_name=law_name, points=points)
