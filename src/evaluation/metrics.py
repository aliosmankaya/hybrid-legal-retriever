def hit_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> int:
    """İlk k sonuç içinde relevant chunk var mı? 1 veya 0."""
    return int(relevant_id in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    """Relevant chunk kaçıncı sırada? 1/rank döndür."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def evaluate_pipeline(
    pipeline_fn,  # chunk_id listesi döndüren fonksiyon
    dataset: list[dict],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    hit_scores = {k: [] for k in k_values}
    mrr_scores = []

    for item in dataset:
        query = item["question"]
        relevant_id = item["relevant_chunk_id"]

        retrieved_ids = pipeline_fn(query)  # chunk_id listesi

        # MRR
        mrr_scores.append(reciprocal_rank(retrieved_ids, relevant_id))

        # Hit@k
        for k in k_values:
            hit_scores[k].append(hit_at_k(retrieved_ids, relevant_id, k))

    results = {"MRR": round(sum(mrr_scores) / len(mrr_scores), 4)}
    for k in k_values:
        results[f"Hit@{k}"] = round(sum(hit_scores[k]) / len(hit_scores[k]), 4)

    return results
