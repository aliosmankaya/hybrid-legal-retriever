from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="legal_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)


def index_chunks(chunks: list[dict], model):
    embeddings = model.encode([c["text"] for c in chunks])
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload=chunk["metadata"])
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name="legal_docs", points=points)
