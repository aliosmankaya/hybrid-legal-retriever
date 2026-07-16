from .bm25_retriever import retriever as bm25_retriever
from .dense_retriever import retriever as dense_retriever
from .helper import load_chunks


def reciprocal_rank_fusion(bm25_results, dense_results, chunks, limit, k: int = 60):
    rrf_scores = {}
    for rank, (chunk_id, _) in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:limit]
    id_to_chunk = {c["chunk_id"]: c for c in chunks}

    return [
        {**id_to_chunk[cid], "rrf_score": rrf_scores[cid]}
        for cid in sorted_ids
        if cid in id_to_chunk
    ]


def retriever(query: str, law_name: str, limit: int):
    bm25_results = bm25_retriever(query=query, law_name=law_name, limit=limit)
    dense_results = dense_retriever(query=query, law_name=law_name, limit=limit)
    chunks = load_chunks(law_name=law_name)

    return reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        chunks=chunks,
        limit=limit,
    )
