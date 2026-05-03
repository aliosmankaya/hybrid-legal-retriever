from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from .bm25_retriever import bm25_search


def dense_search(
    query: str, model: SentenceTransformer, client: QdrantClient, top_k: int = 20
) -> list[tuple[str, float]]:
    query_emb = model.encode([query])[0].tolist()
    results = client.search(
        collection_name="legal_docs", query_vector=query_emb, limit=top_k
    )
    return [(str(r.id), r.score) for r in results]


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[dict, float]],
    chunks: list[dict],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    rrf_scores = {}

    for rank, (chunk_id, _) in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    for rank, (chunk, _) in enumerate(sparse_results):
        chunk_id = chunk["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    id_to_chunk = {c["chunk_id"]: c for c in chunks}

    return [
        {**id_to_chunk[cid], "rrf_score": rrf_scores[cid]}
        for cid in sorted_ids
        if cid in id_to_chunk
    ]


def hybrid_search(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    bm25,
    chunks: list[dict],
    top_k: int = 20,
) -> list[dict]:
    dense_results = dense_search(query, model, client, top_k=top_k)
    sparse_results = bm25_search(query, bm25, chunks, top_k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results, chunks, top_k=top_k)
